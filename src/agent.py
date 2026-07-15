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
    GOOGLE_API_KEY,
    LOCAL_EMBEDDING_MODEL,
    HYBRID_TOP_K,
    HYBRID_MIN_PROMPT_CHARS,
    HYBRID_EXCLUDE_RECENT_COUNT,
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

    def _should_use_hybrid_search(self, prompt: str) -> bool:
        text = _extract_text(prompt).lower()
        if len(text) >= HYBRID_MIN_PROMPT_CHARS:
            return True

        # For short prompts, only retrieve when the user implies memory/reference intent.
        return bool(
            re.search(
                r"\b(remind|remember|earlier|previous|before|last time|you said|we said|continue|recap|summary|what did i)\b",
                text,
            )
        )

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

        retrieval_context = ""
        if session_id and self._should_use_hybrid_search(prompt):
            try:
                related_conversations = await self.store.hybrid_search_conversations(
                    prompt,
                    session_id,
                    embedding_provider=self.generate_embedding,
                    limit=HYBRID_TOP_K,
                )
                retrieval_context = self._format_hybrid_context(related_conversations)
                messages.append(SystemMessage(content=retrieval_context))
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

    def close(self):
        """Clean up resources."""
        self.store.close()
