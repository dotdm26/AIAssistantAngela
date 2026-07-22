import os
import asyncio
import re
import json
import inspect
from typing import List, Union, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, util
from src.utils.config import (
    COMMAND_MEMORY_MAX_PROMPT_LEN,
    GOOGLE_API_KEY,
    LOCAL_EMBEDDING_MODEL,
    HYBRID_TOP_K,
    HYBRID_MIN_PROMPT_CHARS,
    HYBRID_EXCLUDE_RECENT_COUNT,
    HYBRID_MIN_SCORE,
    HYBRID_MIN_SCORE_MEMORY,
    MAX_TOOL_ROUNDS,
    TOOL_FILTER_MIN_SCORE,
)

# Avoid HF Xet/CAS path issues in some environments (401 Unauthorized on public repos).
# os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.conversation_store import ConversationStore
from src.utils.embeddings import LocalNomicEmbeddings, detect_embedding_dimension
from src.utils.prompts import build_system_prompt, configure_formatting
from src.utils.trackers import (
    estimate_token_count,
    log_hybrid_filter_skipped,
    log_token_estimate_input,
    log_token_estimate_output,
    log_token_usage,
    log_tool_filter_fallback,
    log_tool_filter_selected,
    log_tool_router_decision,
    log_tool_router_fallback,
)
from src.tools.general_tools import general_tools_list
from src.tools.web_search_tools import web_search_tools
from src.tools.commands import (
    commands_list,
    detect_command_registration,
    detect_command_lookup,
    is_command_candidate,
)
from src.tools.calendar_tools import calendar_tools_list
from src.tools.gmail_tools import (
    gmail_label_tools_list,
    gmail_read_tools_list,
    gmail_write_tools_list,
)

load_dotenv()

MEMORY_INTENT_RE = re.compile(
    r"\b(remind|remember|earlier|previous|before|last time|you said|we said|continue|recap|summary|what did i)\b"
)
TOOL_INTENT_RE = re.compile(
    r"\b(calendar|event|schedule|meeting|reminder|time|date|gmail|email|mail|inbox|message|messages|label|labels|draft|run|execute|use command|check|search|web|browse|crawl|extract|url|/\w+)\b"
)
GMAIL_INTENT_RE = re.compile(
    r"\b(gmail|email|mail|inbox|message|messages|label|labels|draft|send|recipient|subject)\b"
)
GMAIL_READ_INTENT_RE = re.compile(
    r"\b(gmail|email|mail|inbox|message|messages|search|find|list|read)\b"
)
GMAIL_LABEL_INTENT_RE = re.compile(
    r"\b(label|labels|tag|tags|categorize|category)\b"
)
GMAIL_WRITE_INTENT_RE = re.compile(
    r"\b(send|draft|compose|write|reply|forward|recipient|subject)\b"
)
CALENDAR_INTENT_RE = re.compile(
    r"\b(calendar|event|events|schedule|meeting|reminder|appointment)\b"
)
WEB_SEARCH_INTENT_RE = re.compile(
    r"\b(search|web|webpage|internet|browse|look up|lookup|find online|crawl|extract|url|website|news|headline|headlines)\b"
)
WEB_DEEPEN_INTENT_RE = re.compile(
    r"\b(continue|further|deeper|deep|more|latest|current|up to date|update|updated|refresh)\b"
)
DOMAIN_PATTERN_RE = re.compile(r"\b(?:https?://)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?:/[^\s]*)?\b")

general_tool_names = {tool.name: tool for tool in general_tools_list}
commands_tool_names = {tool.name: tool for tool in commands_list}
calendar_tool_names = {tool.name: tool for tool in calendar_tools_list}
gmail_tools_list = gmail_read_tools_list + gmail_label_tools_list + gmail_write_tools_list
gmail_tool_names = {tool.name: tool for tool in gmail_tools_list}
web_search_tool_names = {tool.name: tool for tool in web_search_tools}

tools_list = general_tools_list + commands_list + calendar_tools_list + gmail_tools_list + web_search_tools

READ_ONLY_TOOL_NAMES = {
    "use_command",
    "get_time",
    "list_calendars",
    "get_upcoming_calendar_events",
    "get_calendar_events_between",
    "gmail_list_messages",
    "gmail_search_messages",
    "gmail_list_labels",
}


def _tool_accepts_arg(tool_obj, arg_name: str) -> bool:
    try:
        fn = getattr(tool_obj, "func", None)
        if not callable(fn):
            return False
        signature = inspect.signature(fn)
        return arg_name in signature.parameters
    except Exception:
        return False

# avoid sending empty messages to the LLM, which can cause errors
def _extract_text(content):
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        return "".join(text_parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _is_command_candidate(text: str) -> bool:
    return is_command_candidate(text, max_len=COMMAND_MEMORY_MAX_PROMPT_LEN)


def _looks_like_memory_key(text: str) -> bool:
    stripped = (text or "").strip().lower()
    if not stripped or len(stripped) > 64:
        return False
    return bool(re.fullmatch(r"[a-z0-9_-]{4,}", stripped) and re.search(r"\d", stripped))


def _is_explicit_memory_intent(text: str) -> bool:
    return bool(MEMORY_INTENT_RE.search((text or "").strip().lower()))


def _hybrid_score_threshold(text: str) -> float:
    
    return HYBRID_MIN_SCORE_MEMORY if (_looks_like_memory_key(text) or _is_explicit_memory_intent(text)) else HYBRID_MIN_SCORE


def _format_hybrid_context(results) -> str:
    if not results:
        return ""

    lines = ["Relevant past conversations:"]
    for user_message, agent_response, text_rank, semantic_score, hybrid_score in results:
        user_text = _extract_text(user_message)
        agent_text = _extract_text(agent_response)
        lines.append(
            f"- User: {user_text}\n  Assistant: {agent_text}\n  Scores: text={text_rank:.3f}, semantic={semantic_score:.3f}, hybrid={hybrid_score:.3f}"
        )

    return "\n".join(lines)


def _tool_is_read_only(tool_name: str) -> bool:
    return tool_name in READ_ONLY_TOOL_NAMES


def _extract_domains_from_text(text: str) -> List[str]:
    domains = []
    for match in DOMAIN_PATTERN_RE.finditer((text or "").lower()):
        domain = match.group(1).strip(". ")
        if domain and domain not in domains:
            domains.append(domain)
    return domains

class AIAgent:
    def __init__(self):
        
        self.llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
        self.llm_general = self.llm.bind_tools(general_tools_list)
        self.llm_commands = self.llm.bind_tools(commands_list)
        self.llm_calendar = self.llm.bind_tools(calendar_tools_list)
        self.llm_gmail_read = self.llm.bind_tools(gmail_read_tools_list)
        self.llm_gmail_label = self.llm.bind_tools(gmail_label_tools_list)
        self.llm_gmail_write = self.llm.bind_tools(gmail_write_tools_list)
        self.llm_web_search = self.llm.bind_tools(web_search_tools)
        self.tool_filter = ToolFilter()
        self._bound_tools_cache = {}
        self.embeddings = LocalNomicEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        self.embedding_dim = detect_embedding_dimension(self.embeddings)
        self.conversation_history = {}
        self.tools_by_name = {**general_tool_names, **commands_tool_names, **calendar_tool_names, **gmail_tool_names, **web_search_tool_names}

        self.store = ConversationStore(
            database_url=os.getenv("DATABASE_URL"),
            embedding_dim=self.embedding_dim,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
            hybrid_exclude_recent_count=HYBRID_EXCLUDE_RECENT_COUNT,
            command_memory_max_prompt_len=COMMAND_MEMORY_MAX_PROMPT_LEN,
        )

        # Agent configuration
        self.system_prompt = build_system_prompt(os.getenv("EXTRA_INSTRUCTIONS"))

    def _get_llm_with_tools_for_prompt(self, prompt: str):
        """Choose a domain-specific tool-bound LLM to keep tool lists small and fast."""
        prompt_text = _extract_text(prompt).strip().lower()

        if GMAIL_INTENT_RE.search(prompt_text):
            if GMAIL_LABEL_INTENT_RE.search(prompt_text):
                return self.llm_gmail_label
            if GMAIL_WRITE_INTENT_RE.search(prompt_text):
                return self.llm_gmail_write
            if GMAIL_READ_INTENT_RE.search(prompt_text):
                return self.llm_gmail_read
            return self.llm_gmail_read

        if CALENDAR_INTENT_RE.search(prompt_text):
            return self.llm_calendar

        if WEB_SEARCH_INTENT_RE.search(prompt_text):
            return self.llm_web_search

        return self.tool_filter.get_bound_llm_for_prompt(
            prompt,
            llm=self.llm,
            bound_tools_cache=self._bound_tools_cache,
            max_rounds=MAX_TOOL_ROUNDS,
        )

    def should_enable_tools(self, prompt: str) -> bool:
        """Public helper for callers to detect whether this prompt is likely to trigger tools."""
        return self.tool_filter.has_tool_intent(prompt)

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """Retrieve conversation history for context."""
        return self.store.get_conversation_history(session_id, limit=limit)

    async def _handle_direct_command_intent(self, prompt: str, prompt_text: str, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None

        registration = detect_command_registration(prompt)
        if registration:
            command_key, command_response = registration
            await asyncio.to_thread(
                self.store.record_command_memory,
                session_id,
                command_key,
                command_response,
            )
            return f"Registered command '{command_key}'."

        if not _is_command_candidate(prompt_text):
            return None

        lookup_key = detect_command_lookup(prompt)
        command_key = lookup_key or prompt_text
        command_response = self.store.resolve_command_memory(
            session_id,
            command_key,
        )
        if command_response:
            print(
                f"[command_memory] resolved exact command | prompt={command_key!r}"
            )
            return command_response

        stripped_prompt = prompt.strip()
        if lookup_key and (stripped_prompt.startswith("/") or " " in stripped_prompt):
            return f"I could not find a registered command for '{lookup_key}'."

        return None

    async def _check_retrieval_context(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        prompt: str,
        prompt_text: str,
        session_id: Optional[str],
    ) -> str:
        if not session_id:
            return ""

        is_session_memory_alias = bool(
            _is_command_candidate(prompt_text)
            and self.store.has_exact_user_message(session_id, prompt_text, min_count=2)
        )
        should_use_hybrid_search = (
            len(prompt_text) >= HYBRID_MIN_PROMPT_CHARS
            or _looks_like_memory_key(prompt_text)
            or is_session_memory_alias
            or _is_explicit_memory_intent(prompt_text)
        )
        if not should_use_hybrid_search:
            return ""

        retrieval_context = ""
        try:
            retrieval_limit = HYBRID_TOP_K
            if _looks_like_memory_key(prompt_text) or is_session_memory_alias:
                retrieval_limit = max(HYBRID_TOP_K, 10)

            related_conversations = await self.store.hybrid_search_conversations(
                prompt,
                session_id,
                embedding_provider=self.generate_embedding,
                limit=retrieval_limit,
            )

            top_hybrid_score = max((row[4] for row in related_conversations), default=0.0)
            threshold = _hybrid_score_threshold(prompt_text)
            if is_session_memory_alias:
                threshold = min(threshold, HYBRID_MIN_SCORE_MEMORY)

            if top_hybrid_score >= threshold:
                retrieval_context = _format_hybrid_context(related_conversations)
                messages.append(SystemMessage(content=retrieval_context))
            else:
                log_hybrid_filter_skipped(top_hybrid_score, threshold)
        except Exception as exc:
            print(f"Hybrid search failed: {exc}")

        return retrieval_context

    async def _invoke_model(
        self,
        sanitized_messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        session_id: Optional[str],
        prompt: str,
        enable_tools: bool = True,
    ):
        if not enable_tools:
            if hasattr(self.llm, "ainvoke"):
                return await self.llm.ainvoke(sanitized_messages)
            return await asyncio.to_thread(self.llm.invoke, sanitized_messages)

        llm_with_tools = self._get_llm_with_tools_for_prompt(prompt)
        prompt_text = _extract_text(prompt).strip().lower()
        is_web_prompt = bool(WEB_SEARCH_INTENT_RE.search(prompt_text))
        is_web_deepen_prompt = bool(is_web_prompt and WEB_DEEPEN_INTENT_RE.search(prompt_text))
        conversation_for_model = list(sanitized_messages)
        max_tool_rounds = MAX_TOOL_ROUNDS
        if is_web_deepen_prompt:
            max_tool_rounds = max(max_tool_rounds, 4)
        elif is_web_prompt:
            max_tool_rounds = max(max_tool_rounds, 3)
        response = None
        saw_tool_call = False
        for _ in range(max_tool_rounds):
            if hasattr(llm_with_tools, "ainvoke"):
                response = await llm_with_tools.ainvoke(conversation_for_model)
            else:
                response = await asyncio.to_thread(llm_with_tools.invoke, conversation_for_model)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            saw_tool_call = True

            conversation_for_model.append(response)
            can_parallelize = all(_tool_is_read_only(tool_call.get("name", "")) for tool_call in tool_calls)
            if can_parallelize:
                async def _invoke_one(tool_call):
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                    tool_fn = self.tools_by_name.get(tool_name)

                    if not tool_fn:
                        return ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id)

                    if not isinstance(tool_args, dict):
                        try:
                            tool_args = json.loads(tool_args) if isinstance(tool_args, str) else {}
                        except Exception:
                            tool_args = {}

                    if "session_id" not in tool_args and session_id and _tool_accepts_arg(tool_fn, "session_id"):
                        tool_args["session_id"] = session_id

                    try:
                        tool_output = await asyncio.to_thread(tool_fn.invoke, tool_args)
                    except Exception as exc:
                        tool_output = f"Tool execution error: {exc}"

                    return ToolMessage(content=str(tool_output), tool_call_id=tool_id)

                tool_messages = await asyncio.gather(*[_invoke_one(tool_call) for tool_call in tool_calls])
                conversation_for_model.extend(tool_messages)
                continue

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                tool_fn = self.tools_by_name.get(tool_name)

                if not tool_fn:
                    tool_output = f"Unknown tool: {tool_name}"
                else:
                    if not isinstance(tool_args, dict):
                        try:
                            tool_args = json.loads(tool_args) if isinstance(tool_args, str) else {}
                        except Exception:
                            tool_args = {}

                    if "session_id" not in tool_args and session_id and _tool_accepts_arg(tool_fn, "session_id"):
                        tool_args["session_id"] = session_id

                    try:
                        tool_output = tool_fn.invoke(tool_args)
                    except Exception as exc:
                        tool_output = f"Tool execution error: {exc}"

                conversation_for_model.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_id,
                    )
                )

        # Hard fallback for explicit web prompts when model skips tool calls.
        # This guarantees at least one live web lookup before responding.
        if is_web_prompt and not saw_tool_call:
            forced_tool_output = ""
            forced_tool = self.tools_by_name.get("search_web")
            if forced_tool is not None:
                force_args = {
                    "query": prompt,
                    "search_depth": "advanced" if is_web_deepen_prompt else "basic",
                    "include_answer": True,
                }
                domains = _extract_domains_from_text(prompt_text)
                if domains:
                    force_args["include_domains"] = domains

                try:
                    forced_tool_output = await asyncio.to_thread(forced_tool.invoke, force_args)
                except Exception as exc:
                    forced_tool_output = f"Tool execution error: {exc}"

            forced_synthesis_messages = list(sanitized_messages)
            forced_synthesis_messages.append(
                SystemMessage(
                    content=(
                        "You must ground your answer in the provided live web search result. "
                        "If the result is insufficient, say what is missing and suggest a narrower follow-up query."
                    )
                )
            )
            forced_synthesis_messages.append(
                HumanMessage(
                    content=(
                        f"User request:\n{prompt}\n\n"
                        f"Live web search result:\n{forced_tool_output}"
                    )
                )
            )
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(forced_synthesis_messages)
            else:
                response = await asyncio.to_thread(self.llm.invoke, forced_synthesis_messages)

        # If we exhausted rounds while the model was still requesting tools,
        # force one synthesis pass to avoid returning a tool-call payload.
        if response is not None and (getattr(response, "tool_calls", None) or []):
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(conversation_for_model)
            else:
                response = await asyncio.to_thread(self.llm.invoke, conversation_for_model)

        if response is None:
            raise RuntimeError("No model response received")

        return response

    async def generate_reply(
        self,
        history: List[Union[HumanMessage, AIMessage, SystemMessage]],
        prompt: str,
        session_id: Optional[str] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        messages = list(history)
        prompt_text = _extract_text(prompt).lower()

        direct_command_response = await self._handle_direct_command_intent(prompt, prompt_text, session_id)
        if direct_command_response is not None:
            return direct_command_response

        retrieval_context = await self._check_retrieval_context(messages, prompt, prompt_text, session_id)

        messages.append(HumanMessage(content=prompt))
        sanitized_messages = []
        for message in messages:
            content = getattr(message, "content", None)
            text = _extract_text(content)
            if text:
                sanitized_messages.append(message)

        estimated_input_tokens = estimate_token_count(sanitized_messages, _extract_text)
        log_token_estimate_input(estimated_input_tokens, len(sanitized_messages), bool(retrieval_context))

        enable_tools = self.tool_filter.should_invoke_with_tools(prompt, min_score=TOOL_FILTER_MIN_SCORE)
        response = await self._invoke_model(
            sanitized_messages,
            session_id,
            prompt,
            enable_tools=enable_tools,
        )

        log_token_usage(response)

        if hasattr(response, "content"):
            output = _extract_text(response.content)
            log_token_estimate_output(output)
            return output

        output = _extract_text(response)
        log_token_estimate_output(output)
        return output

    async def generate_embedding(self, text: str):
        """Generate an embedding for the given text."""
        if hasattr(self.embeddings, "aembed_query"):
            return await self.embeddings.aembed_query(text)
        return await asyncio.to_thread(self.embeddings.embed_query, text)

    async def store_conversation(self, session_id: str, user_message: str, agent_response: str):
        """Store a conversation in the database, including its embedding for semantic search."""
        combined_text = f"{user_message}\n\n{agent_response}"
        embedding = await self.generate_embedding(combined_text)
        await asyncio.to_thread(
            self.store.save_conversation,
            session_id,
            user_message,
            agent_response,
            embedding,
        )

    def close(self):
        """Clean up resources."""
        self.store.close()


class ToolFilter:
    """Filter to determine if a tool should be invoked based on the prompt."""

    def __init__(self):
        self.llm = SentenceTransformer("all-MiniLM-L6-v2")

        self.tool_library = tools_list
        self.tool_embeddings = self.llm.encode([tool.description for tool in self.tool_library], convert_to_tensor=True)

    def has_tool_intent(self, prompt: str) -> bool:
        text = _extract_text(prompt).strip().lower()
        if not text:
            return False

        if detect_command_registration(text) or detect_command_lookup(text):
            return True

        return bool(TOOL_INTENT_RE.search(text))

    def should_invoke_with_tools(self, prompt: str, min_score: float) -> bool:
        """Route to tool-bound model only when intent and relevance both indicate tool use."""
        text = _extract_text(prompt).strip().lower()

        if detect_command_registration(text) or detect_command_lookup(text):
            return True

        if WEB_SEARCH_INTENT_RE.search(text) or GMAIL_INTENT_RE.search(text) or CALENDAR_INTENT_RE.search(text):
            return True

        if not self.has_tool_intent(prompt):
            return False

        try:
            top_score = self.get_top_score(prompt)
            decision = top_score >= min_score
            log_tool_router_decision(top_score, min_score, decision)
            return decision
        except Exception as exc:
            # If the filter fails, keep existing behavior for explicit tool intents.
            log_tool_router_fallback(exc)
            return True

    def get_bound_llm_for_prompt(self, prompt: str, llm, bound_tools_cache: dict, max_rounds: int = MAX_TOOL_ROUNDS):
        """Bind only top relevant tools for this prompt, with safe fallbacks and cache."""
        selected_tools = self.tool_library
        selected_tool_names = [tool.name for tool in self.tool_library]

        prompt_text = _extract_text(prompt).strip().lower()
        force_gmail_tools = bool(GMAIL_INTENT_RE.search(prompt_text))
        force_calendar_tools = bool(CALENDAR_INTENT_RE.search(prompt_text))
        force_web_tools = bool(WEB_SEARCH_INTENT_RE.search(prompt_text))

        try:
            filtered_tools = self.get_relevant_tools(prompt, max_rounds=max_rounds)
            if filtered_tools:
                selected_tools = filtered_tools
                selected_tool_names = [tool.name for tool in filtered_tools]

            if force_gmail_tools:
                by_name = {tool.name: tool for tool in selected_tools}
                for tool in gmail_tools_list:
                    by_name.setdefault(tool.name, tool)
                selected_tools = list(by_name.values())
                selected_tool_names = [tool.name for tool in selected_tools]

            if force_calendar_tools:
                by_name = {tool.name: tool for tool in selected_tools}
                for tool in calendar_tools_list:
                    by_name.setdefault(tool.name, tool)
                selected_tools = list(by_name.values())
                selected_tool_names = [tool.name for tool in selected_tools]

            if force_web_tools:
                by_name = {tool.name: tool for tool in selected_tools}
                for tool in web_search_tools:
                    by_name.setdefault(tool.name, tool)
                selected_tools = list(by_name.values())
                selected_tool_names = [tool.name for tool in selected_tools]
        except Exception as exc:
            log_tool_filter_fallback(exc)

        cache_key = tuple(sorted(selected_tool_names))
        if cache_key not in bound_tools_cache:
            bound_tools_cache[cache_key] = llm.bind_tools(selected_tools)

        log_tool_filter_selected(selected_tool_names)
        return bound_tools_cache[cache_key]

    def get_relevant_tools(self, prompt, max_rounds: int = MAX_TOOL_ROUNDS):
        """Filter tools based on the prompt."""
        prompt_embedding = self.llm.encode(prompt, convert_to_tensor=True)
        cosine_scores = util.cos_sim(prompt_embedding, self.tool_embeddings)[0]
        k = max(1, min(max_rounds, len(self.tool_library)))
        top_tools = cosine_scores.topk(k)
        return [self.tool_library[i] for i in top_tools.indices]

    def get_top_score(self, prompt) -> float:
        """Return top cosine similarity between prompt and tool descriptions."""
        if not self.tool_library:
            return 0.0

        prompt_embedding = self.llm.encode(prompt, convert_to_tensor=True)
        cosine_scores = util.cos_sim(prompt_embedding, self.tool_embeddings)[0]
        return float(cosine_scores.max().item())

    def get_relevant_tool_names(self, prompt, max_rounds: int = MAX_TOOL_ROUNDS):
        """Filter tools based on the prompt."""
        return [tool.name for tool in self.get_relevant_tools(prompt, max_rounds)]

