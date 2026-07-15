import os
import asyncio
import re

# Avoid HF Xet/CAS path issues in some environments (401 Unauthorized on public repos).
# os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from typing import List, Union, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
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
from conversation_store import ConversationStore
from embeddings import LocalNomicEmbeddings, detect_embedding_dimension
from prompts import build_system_prompt

load_dotenv()


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


class AIAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
        self.embeddings = LocalNomicEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        self.embedding_dim = detect_embedding_dimension(self.embeddings)
        self.conversation_history = {}

        self.store = ConversationStore(
            database_url=os.getenv("DATABASE_URL"),
            embedding_dim=self.embedding_dim,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
            hybrid_exclude_recent_count=HYBRID_EXCLUDE_RECENT_COUNT,
            command_memory_max_prompt_len=COMMAND_MEMORY_MAX_PROMPT_LEN,
        )

        # Agent configuration
        self.system_prompt = build_system_prompt(os.getenv("EXTRA_INSTRUCTIONS"))

    def _format_hybrid_context(self, results) -> str:
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

    def _should_use_hybrid_search(self, prompt: str, session_id: Optional[str] = None) -> bool:
        text = _extract_text(prompt).lower()
        if len(text) >= HYBRID_MIN_PROMPT_CHARS:
            return True

        if self._looks_like_memory_key(text):
            return True

        if session_id and self._is_session_memory_alias(session_id, text):
            return True

        # For short prompts, only retrieve when the user implies memory/reference intent.
        return self._is_explicit_memory_intent(text)

    def _is_explicit_memory_intent(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b(remind|remember|earlier|previous|before|last time|you said|we said|continue|recap|summary|what did i)\b",
                text,
            )
        )

    def _get_hybrid_score_threshold(self, text: str) -> float:
        if self._looks_like_memory_key(text) or self._is_explicit_memory_intent(text):
            return HYBRID_MIN_SCORE_MEMORY
        return HYBRID_MIN_SCORE

    def _is_session_memory_alias(self, session_id: str, text: str) -> bool:
        stripped = text.strip()
        if not stripped or " " in stripped or len(stripped) > COMMAND_MEMORY_MAX_PROMPT_LEN:
            return False

        # Reused short prompts in the same session often represent custom commands.
        return self.store.has_exact_user_message(session_id, stripped, min_count=2)

    def _looks_like_memory_key(self, text: str) -> bool:
        """Detect short key-like prompts (e.g., pingpong123) that should force retrieval."""
        stripped = text.strip()
        if not stripped or len(stripped) > 64:
            return False

        # Single-token alphanumeric keys with digits are common user-defined memory triggers.
        return bool(re.fullmatch(r"[a-z0-9_-]{4,}", stripped) and re.search(r"\d", stripped))

    def _is_command_candidate(self, text: str) -> bool:
        stripped = text.strip().lower()
        if not stripped or len(stripped) > COMMAND_MEMORY_MAX_PROMPT_LEN or " " in stripped:
            return False
        return bool(re.fullmatch(r"[a-z0-9_-]{2,}", stripped))

    def _estimate_token_count(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> int:
        # Fast approximation for observability: ~4 chars per token for English-heavy text.
        total_chars = 0
        for message in messages:
            total_chars += len(_extract_text(getattr(message, "content", "")))
        return max(1, total_chars // 4)

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """Retrieve conversation history for context."""
        return self.store.get_conversation_history(session_id, limit=limit)

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
        if session_id and self._is_command_candidate(prompt_text):
            command_response = self.store.resolve_command_memory(
                session_id,
                prompt_text,
                min_usage=COMMAND_MEMORY_MIN_USAGE,
                min_share=COMMAND_MEMORY_MIN_SHARE,
            )
            if command_response:
                print(
                    f"[command_memory] resolved exact command | prompt={prompt_text!r} "
                    f"min_usage={COMMAND_MEMORY_MIN_USAGE} min_share={COMMAND_MEMORY_MIN_SHARE:.2f}"
                )
                return command_response

        retrieval_context = ""
        if session_id and self._should_use_hybrid_search(prompt, session_id=session_id):
            try:
                retrieval_limit = HYBRID_TOP_K
                is_memory_alias = self._is_session_memory_alias(session_id, prompt_text)
                if self._looks_like_memory_key(prompt_text) or is_memory_alias:
                    retrieval_limit = max(HYBRID_TOP_K, 10)

                related_conversations = await self.store.hybrid_search_conversations(
                    prompt,
                    session_id,
                    embedding_provider=self.generate_embedding,
                    limit=retrieval_limit,
                )

                top_hybrid_score = max((row[4] for row in related_conversations), default=0.0)
                threshold = self._get_hybrid_score_threshold(prompt_text)
                if is_memory_alias:
                    threshold = min(threshold, HYBRID_MIN_SCORE_MEMORY)

                if top_hybrid_score >= threshold:
                    retrieval_context = self._format_hybrid_context(related_conversations)
                    messages.append(SystemMessage(content=retrieval_context))
                else:
                    print(
                        f"[hybrid_filter] skipped retrieval context | "
                        f"top_score={top_hybrid_score:.3f} < threshold={threshold:.3f}"
                    )
            except Exception as exc:
                print(f"Hybrid search failed: {exc}")

        messages.append(HumanMessage(content=prompt))
        sanitized_messages = []
        for message in messages:
            content = getattr(message, "content", None)
            text = _extract_text(content)
            if text:
                sanitized_messages.append(message)

        estimated_input_tokens = self._estimate_token_count(sanitized_messages)
        print(
            f"[token_estimate] input~{estimated_input_tokens} tokens | "
            f"messages={len(sanitized_messages)} | hybrid_used={bool(retrieval_context)}"
        )

        if hasattr(self.llm, "ainvoke"):
            response = await self.llm.ainvoke(sanitized_messages)
        else:
            response = await asyncio.to_thread(self.llm.invoke, sanitized_messages)

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
        if self._is_command_candidate(_extract_text(user_message)):
            await asyncio.to_thread(
                self.store.record_command_memory,
                session_id,
                user_message,
                agent_response,
            )

    def close(self):
        """Clean up resources."""
        self.store.close()
