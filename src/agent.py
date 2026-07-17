import os
import asyncio
import re
import json
import inspect
from typing import List, Union, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Avoid HF Xet/CAS path issues in some environments (401 Unauthorized on public repos).
# os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.config import (
    COMMAND_MEMORY_MAX_PROMPT_LEN,
    COMMAND_MEMORY_MIN_SHARE,
    COMMAND_MEMORY_MIN_USAGE,
    GOOGLE_API_KEY,
    LOCAL_EMBEDDING_MODEL,
    HYBRID_TOP_K,
    HYBRID_MIN_PROMPT_CHARS,
    HYBRID_EXCLUDE_RECENT_COUNT,
    HYBRID_MIN_SCORE,
    HYBRID_MIN_SCORE_MEMORY,
)
from src.conversation_store import ConversationStore
from src.embeddings import LocalNomicEmbeddings, detect_embedding_dimension
from src.prompts import build_system_prompt, configure_formatting
from src.tools.commands import (
    save_command,
    use_command,
    process_command,
    detect_command_registration,
    detect_command_lookup,
    is_command_candidate,
)
from src.tools.calendar_tools import (
    get_time,
    list_calendars,
    get_upcoming_calendar_events,
    get_calendar_events_between,
)

load_dotenv()

MEMORY_INTENT_RE = re.compile(
    r"\b(remind|remember|earlier|previous|before|last time|you said|we said|continue|recap|summary|what did i)\b"
)

tool_list = [
    save_command,
    use_command,
    process_command,
    get_time,
    list_calendars,
    get_upcoming_calendar_events,
    get_calendar_events_between
]

tool_names = {tool.name: tool for tool in tool_list}


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


def _estimate_token_count(messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> int:
    # Fast approximation for observability: ~4 chars per token for English-heavy text.
    total_chars = 0
    for message in messages:
        total_chars += len(_extract_text(getattr(message, "content", "")))
    return max(1, total_chars // 4)


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

#class AngelaResponse(BaseModel):
#    reply: str = Field(
#        description=configure_formatting()
#    )

class AIAgent:
    def __init__(self):
        
        self.llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
        self.llm_with_tools = self.llm.bind_tools(tool_list)
        self.embeddings = LocalNomicEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        self.embedding_dim = detect_embedding_dimension(self.embeddings)
        self.conversation_history = {}
        self.tools_by_name = tool_names

        self.store = ConversationStore(
            database_url=os.getenv("DATABASE_URL"),
            embedding_dim=self.embedding_dim,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
            hybrid_exclude_recent_count=HYBRID_EXCLUDE_RECENT_COUNT,
            command_memory_max_prompt_len=COMMAND_MEMORY_MAX_PROMPT_LEN,
        )

        # Agent configuration
        self.system_prompt = build_system_prompt(os.getenv("EXTRA_INSTRUCTIONS"))

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
            min_usage=COMMAND_MEMORY_MIN_USAGE,
            min_share=COMMAND_MEMORY_MIN_SHARE,
        )
        if command_response:
            print(
                f"[command_memory] resolved exact command | prompt={command_key!r} "
                f"min_usage={COMMAND_MEMORY_MIN_USAGE} min_share={COMMAND_MEMORY_MIN_SHARE:.2f}"
            )
            return command_response

        stripped_prompt = prompt.strip()
        if lookup_key and (stripped_prompt.startswith("/") or " " in stripped_prompt):
            return f"I could not find a registered command for '{lookup_key}'."

        return None

    async def _maybe_add_retrieval_context(
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
                print(
                    f"[hybrid_filter] skipped retrieval context | "
                    f"top_score={top_hybrid_score:.3f} < threshold={threshold:.3f}"
                )
        except Exception as exc:
            print(f"Hybrid search failed: {exc}")

        return retrieval_context

    async def _invoke_model_with_tools(
        self,
        sanitized_messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        session_id: Optional[str],
    ):
        conversation_for_model = list(sanitized_messages)
        max_tool_rounds = 3
        response = None
        for _ in range(max_tool_rounds):
            if hasattr(self.llm_with_tools, "ainvoke"):
                response = await self.llm_with_tools.ainvoke(conversation_for_model)
            else:
                response = await asyncio.to_thread(self.llm_with_tools.invoke, conversation_for_model)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            conversation_for_model.append(response)
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

        retrieval_context = await self._maybe_add_retrieval_context(messages, prompt, prompt_text, session_id)

        messages.append(HumanMessage(content=prompt))
        sanitized_messages = []
        for message in messages:
            content = getattr(message, "content", None)
            text = _extract_text(content)
            if text:
                sanitized_messages.append(message)

        estimated_input_tokens = _estimate_token_count(sanitized_messages)
        print(
            f"[token_estimate] input~{estimated_input_tokens} tokens | "
            f"messages={len(sanitized_messages)} | hybrid_used={bool(retrieval_context)}"
        )

        response = await self._invoke_model_with_tools(sanitized_messages, session_id)

        if hasattr(response, "usage_metadata"):
            print(f"[token_usage] {response.usage_metadata}")
        elif hasattr(response, "response_metadata"):
            usage = getattr(response, "response_metadata", {}).get("usage_metadata")
            if usage:
                print(f"[token_usage] {usage}")

        if hasattr(response, "content"):
            output = _extract_text(response.content)
            print(f"[token_estimate] output~{max(1, len(output) // 4)} tokens")
            return output

        output = _extract_text(response)
        print(f"[token_estimate] output~{max(1, len(output) // 4)} tokens")
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
        if _is_command_candidate(_extract_text(user_message)):
            await asyncio.to_thread(
                self.store.record_command_memory,
                session_id,
                user_message,
                agent_response,
            )

    def close(self):
        """Clean up resources."""
        self.store.close()
