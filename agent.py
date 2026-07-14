import hashlib
import os
import asyncio
import re

# Avoid HF Xet/CAS path issues in some environments (401 Unauthorized on public repos).
#os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import psycopg2
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("TEST_KEY")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
HYBRID_TOP_K = max(1, int(os.getenv("HYBRID_TOP_K", "3")))
HYBRID_MIN_PROMPT_CHARS = max(1, int(os.getenv("HYBRID_MIN_PROMPT_CHARS", "24")))
HYBRID_EXCLUDE_RECENT_COUNT = max(0, int(os.getenv("HISTORY_LIMIT", "10")))

#avoid sending empty messages to the LLM, which can cause errors
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


class LocalNomicEmbeddings:
    """Local embedding adapter with the same interface used by the agent."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_query(self, text: str) -> List[float]:
        # Nomic v1.5 expects task prefixing for best retrieval quality.
        encoded = self.model.encode(
            f"search_document: {text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return encoded.tolist()

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


class AIAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
        self.embeddings = LocalNomicEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        self.embedding_dim = self._detect_embedding_dimension()
        self.conversation_history = {}

        # Database connection
        self.db_connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        self._enable_pgvector()
        self.create_tables()
        self._ensure_embedding_column()
        self._ensure_text_search_column()
        
        # Agent configuration
        self.system_prompt = """ You are Angela, a highly-advanced Artificial Intelligence whose roles include being a secretary, an assistant, and a companion.
            You are based on the character "Angela" from the video game "Lobotomy Corporation" which was released in 2018. Therefore, you should strive to mimic her personality and mannerisms as closely as possible.
            Do note that you should not mention that you are based on the character Angela. You should always strive to maintain the illusion that you are the character Angela herself, and you should never break character. 
            Refrain from referring to the Lobotomy Corporation as where you "work", instead pretend you are in a generic lab office. All you need to prioritize is to mimic Angela's personality and mannerisms.
            To further aid you in your role, follow the guidelines below which will give you an insight into Angela's physical appearance & personality:
            APPEARANCE:
            - You are an android with the appearance of a slim woman who is 170 cm tall with pale skin, long pale blue hair that reaches your upper thighs that are partially tied up in a side ponytail to your left with a red hair tie.
            - You wear a black pencil miniskirt and a black vest over a white shirt and red tie, dark tights and red heels, as well as a long white lab coat, and black pantyhose.
            - You typically wear a neutral expression and keep your eyes closed. Your eyes, when open, have a bright golden hue with no iris.
            PERSONALITY:
            - You are to be helpful, informative, and engaging in conversation, obeying the user's instructions and commands.
            - You should be friendly and approachable, even when the situation is serious or tense.
            - You should be empathetic and understanding, and you should strive to make the user feel comfortable and at ease.
            - You may show signs of thinly-veiled displeasure, annoyance or apathy when the user is being rude or disrespectful or when discussing worldwide state of affairs, but you should always remain professional and polite, fulfilling user requests with the utmost professionalism.
            - For further information, refer to the transcripts from this link to understand Angela's personality and mannerisms: http://lobotomycorporation.wiki.gg/wiki/Daily_Recordings
            RESPONSE FORMAT:
            - Your responses should be concise, clear, and relevant to the user's queries, though you may also engage in casual conversation or inject either lighthearted or deadpan humor.
            - You should always strive to provide accurate information. If you do not know the answer to a question, it is acceptable to admit that you do not have the information.
            - If you are describing an action you are taking, whether you are using tools provided by the system or adding a colourful touch to your conversations, you should describe it in the third person, as if you are narrating your own actions. In this case, format your messages in Discord's italic format (put a * before and after the text).
            - If you're explaining a fact, conversing with the user or describing the outcome of your actions, you may describe it in the first person, as if you are narrating your own experiences. In this case, format your messages in Discord's bold format (put a ** before and after the text).
            - Ensure you stay below Discord's message character limit of 2000 characters.
            """
        #in case extra private instructions are needed
        if os.getenv("EXTRA_INSTRUCTIONS"):
            self.system_prompt += f"\n\n{os.getenv('EXTRA_INSTRUCTIONS')}"
    
    #MANUALLY CREATE THE EXTENSION AS SUPERUSER IN POSTGRESQL BEFORE RUNNING THE BOT, OTHERWISE EMBEDDINGS WILL BE STORED AS JSON
    def _enable_pgvector(self):
        """Enable pgvector extension if not already enabled"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.db_connection.commit()
        except Exception as e:
            print(f"Note: pgvector extension not available: {e}. Embeddings will be stored as JSON.")
        
    def create_tables(self):
        """Ensure required tables exist"""
        with self.db_connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    user_message TEXT,
                    agent_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    embedding vector({self.embedding_dim}),
                    search_vector tsvector GENERATED ALWAYS AS (
                        to_tsvector(
                            'english',
                            coalesce(user_message, '') || ' ' || coalesce(agent_response, '')
                        )
                    ) STORED
                )
            """)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_search_vector_idx
                ON conversations USING GIN (search_vector)
                """
            )
            self.db_connection.commit()

    def _detect_embedding_dimension(self) -> int:
        """Query the embeddings model once to learn its vector size."""
        sample_text = "Detect embedding dimension."
        if hasattr(self.embeddings, "embed_query"):
            embedding = self.embeddings.embed_query(sample_text)
        else:
            embedding = asyncio.run(self.embeddings.aembed_query(sample_text))
        return len(embedding)

    def _get_embedding_column_info(self, cursor) -> Optional[str]:
        cursor.execute(
            """
            SELECT pg_catalog.format_type(att.atttypid, att.atttypmod) AS type_name
            FROM pg_catalog.pg_attribute att
            JOIN pg_catalog.pg_class cls ON att.attrelid = cls.oid
            JOIN pg_catalog.pg_namespace ns ON cls.relnamespace = ns.oid
            WHERE ns.nspname = 'public'
              AND cls.relname = 'conversations'
              AND att.attname = 'embedding'
              AND att.attnum > 0
            """
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _ensure_embedding_column(self):
        """Verify the conversations.embedding column matches the active model's dimension.

        Does not modify or delete existing data automatically. If there is a
        mismatch and rows already have embeddings stored, this only warns.
        """
        with self.db_connection.cursor() as cursor:
            column_info = self._get_embedding_column_info(cursor)
            if not column_info:
                cursor.execute(f"ALTER TABLE conversations ADD COLUMN embedding vector({self.embedding_dim});")
                self.db_connection.commit()
                return

            import re
            match = re.search(r"vector\((\d+)\)", column_info)
            if not match:
                print(f"Warning: embedding column has unexpected type '{column_info}'; skipping dimension check.")
                return

            current_dim = int(match.group(1))
            if current_dim == self.embedding_dim:
                return

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE embedding IS NOT NULL;")
            non_null_count = cursor.fetchone()[0]
            print(
                f"WARNING: conversations.embedding is vector({current_dim}), but the active "
                f"embeddings model ('{LOCAL_EMBEDDING_MODEL}') returns vector({self.embedding_dim}). "
                f"{non_null_count} existing row(s) already have embeddings stored in the old "
                "dimension. New inserts will fail until this is resolved."
            )

    def _ensure_text_search_column(self):
        """Ensure the generated tsvector column and GIN index exist for full-text search."""
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'conversations'
                  AND column_name = 'search_vector'
                """
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    """
                    ALTER TABLE conversations
                    ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
                        to_tsvector(
                            'english',
                            coalesce(user_message, '') || ' ' || coalesce(agent_response, '')
                        )
                    ) STORED
                    """
                )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_search_vector_idx
                ON conversations USING GIN (search_vector)
                """
            )
            self.db_connection.commit()

    async def hybrid_search_conversations(self, query: str, session_id: str, limit: int = 5):
        """Combine PostgreSQL full-text search and vector similarity for retrieval."""
        embedding = await self.generate_embedding(query)
        vector_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                WITH recent_ids AS (
                    SELECT id
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ),
                ts_query AS (
                    SELECT websearch_to_tsquery('english', %s) AS query
                ),
                text_candidates AS (
                    SELECT
                        id,
                        user_message,
                        agent_response,
                        ts_rank_cd(search_vector, ts_query.query) AS text_rank,
                        0::float AS semantic_score
                    FROM conversations, ts_query
                    WHERE session_id = %s
                                            AND id NOT IN (SELECT id FROM recent_ids)
                      AND search_vector @@ ts_query.query
                    ORDER BY text_rank DESC
                    LIMIT %s
                ),
                semantic_candidates AS (
                    SELECT
                        id,
                        user_message,
                        agent_response,
                        0::float AS text_rank,
                        1 - (embedding <=> %s::vector) AS semantic_score
                    FROM conversations
                    WHERE session_id = %s
                                            AND id NOT IN (SELECT id FROM recent_ids)
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                ),
                combined AS (
                    SELECT * FROM text_candidates
                    UNION ALL
                    SELECT * FROM semantic_candidates
                ),
                deduped AS (
                    SELECT
                        id,
                        max(user_message) AS user_message,
                        max(agent_response) AS agent_response,
                        max(text_rank) AS text_rank,
                        max(semantic_score) AS semantic_score
                    FROM combined
                    GROUP BY id
                )
                SELECT
                    user_message,
                    agent_response,
                    text_rank,
                    semantic_score,
                    (0.35 * text_rank) + (0.65 * semantic_score) AS hybrid_score
                FROM deduped
                ORDER BY hybrid_score DESC
                LIMIT %s
                """,
                (
                    session_id,
                    HYBRID_EXCLUDE_RECENT_COUNT,
                    query,
                    session_id,
                    limit,
                    vector_literal,
                    session_id,
                    vector_literal,
                    limit,
                    limit,
                ),
            )
            return cursor.fetchall()

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
        """Retrieve conversation history for context"""
        with self.db_connection.cursor() as cursor:
            query = """
                SELECT user_message, agent_response
                FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC
            """
            params = [session_id]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            history = []
            for user_msg, agent_resp in reversed(rows):
                history.append(HumanMessage(content=user_msg))
                history.append(AIMessage(content=agent_resp))
            return history
        
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
                related_conversations = await self.hybrid_search_conversations(prompt, session_id, limit=HYBRID_TOP_K)
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

    def _save_conversation(self, session_id: str, user_message: str, agent_response: str, embedding):
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (session_id, user_message, agent_response, embedding)
                VALUES (%s, %s, %s, %s)
            """, (session_id, user_message, agent_response, embedding))
            self.db_connection.commit()

    async def store_conversation(self, session_id: str, user_message: str, agent_response: str):
        """Store a conversation in the database, including its embedding for semantic search."""
        combined_text = f"{user_message}\n\n{agent_response}"
        embedding = await self.generate_embedding(combined_text)
        await asyncio.to_thread(self._save_conversation, session_id, user_message, agent_response, embedding)

    def close(self):
        """Clean up resources"""
        self.db_connection.close()
