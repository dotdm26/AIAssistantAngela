import re
from typing import Awaitable, Callable, List, Optional

import psycopg2
from langchain_core.messages import AIMessage, HumanMessage


class ConversationStore:
    def __init__(
        self,
        database_url: str,
        embedding_dim: int,
        local_embedding_model: str,
        hybrid_exclude_recent_count: int,
        command_memory_max_prompt_len: int,
    ):
        self.embedding_dim = embedding_dim
        self.local_embedding_model = local_embedding_model
        self.hybrid_exclude_recent_count = hybrid_exclude_recent_count
        self.command_memory_max_prompt_len = command_memory_max_prompt_len
        self.db_connection = psycopg2.connect(database_url)

        self._enable_pgvector()
        self.create_tables()
        self._ensure_embedding_column()
        self._ensure_text_search_column()
        self._ensure_command_memory_table()
        self._rebuild_command_memory_cache()

    # MANUALLY CREATE THE EXTENSION AS SUPERUSER IN POSTGRESQL BEFORE RUNNING THE BOT,
    # OTHERWISE EMBEDDINGS WILL BE STORED AS JSON.
    def _enable_pgvector(self):
        """Enable pgvector extension if not already enabled."""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.db_connection.commit()
        except Exception as e:
            print(f"Note: pgvector extension not available: {e}. Embeddings will be stored as JSON.")

    def create_tables(self):
        """Ensure required tables exist."""
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                f"""
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
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_search_vector_idx
                ON conversations USING GIN (search_vector)
                """
            )
            self.db_connection.commit()

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
        """Verify the conversations.embedding column matches the active model's dimension."""
        with self.db_connection.cursor() as cursor:
            column_info = self._get_embedding_column_info(cursor)
            if not column_info:
                cursor.execute(f"ALTER TABLE conversations ADD COLUMN embedding vector({self.embedding_dim});")
                self.db_connection.commit()
                return

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
                f"embeddings model ('{self.local_embedding_model}') returns vector({self.embedding_dim}). "
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

    def _ensure_command_memory_table(self):
        """Store stable short command-response mappings separately from general chat history."""
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS command_memory (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    trigger_text TEXT NOT NULL,
                    normalized_trigger TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (session_id, normalized_trigger, response_text)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS command_memory_lookup_idx
                ON command_memory (session_id, normalized_trigger, last_seen_at DESC)
                """
            )
            self.db_connection.commit()

    def _rebuild_command_memory_cache(self):
        """Backfill command memory from existing conversations so old commands resolve immediately."""
        with self.db_connection.cursor() as cursor:
            cursor.execute("DELETE FROM command_memory")
            cursor.execute(
                """
                INSERT INTO command_memory (
                    session_id,
                    trigger_text,
                    normalized_trigger,
                    response_text,
                    first_seen_at,
                    last_seen_at
                )
                SELECT
                    session_id,
                    min(trim(user_message)) AS trigger_text,
                    lower(trim(user_message)) AS normalized_trigger,
                    agent_response AS response_text,
                    min(created_at) AS first_seen_at,
                    max(created_at) AS last_seen_at
                FROM conversations
                WHERE coalesce(trim(user_message), '') <> ''
                  AND position(' ' in trim(user_message)) = 0
                  AND char_length(trim(user_message)) <= %s
                  AND trim(user_message) ~ '^[A-Za-z0-9_-]{2,}$'
                  AND coalesce(agent_response, '') <> ''
                GROUP BY session_id, lower(trim(user_message)), agent_response
                """,
                (self.command_memory_max_prompt_len,),
            )
            self.db_connection.commit()

    async def hybrid_search_conversations(
        self,
        query: str,
        session_id: str,
        embedding_provider: Callable[[str], Awaitable[List[float]]],
        limit: int = 5,
    ):
        """Combine PostgreSQL full-text search and vector similarity for retrieval."""
        embedding = await embedding_provider(query)
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
                exact_candidates AS (
                    SELECT
                        id,
                        user_message,
                        agent_response,
                        1.25::float AS text_rank,
                        0::float AS semantic_score
                    FROM conversations
                    WHERE session_id = %s
                      AND id NOT IN (SELECT id FROM recent_ids)
                      AND (
                          lower(coalesce(user_message, '')) LIKE lower('%%' || %s || '%%')
                          OR lower(coalesce(agent_response, '')) LIKE lower('%%' || %s || '%%')
                      )
                    ORDER BY created_at DESC
                    LIMIT %s
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
                    SELECT * FROM exact_candidates
                    UNION ALL
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
                ),
                normalized AS (
                    SELECT
                        user_message,
                        agent_response,
                        text_rank,
                        semantic_score,
                        CASE
                            WHEN max(text_rank) OVER () = min(text_rank) OVER ()
                                THEN CASE WHEN max(text_rank) OVER () > 0 THEN 1::float ELSE 0::float END
                            ELSE (text_rank - min(text_rank) OVER ())
                                / NULLIF(max(text_rank) OVER () - min(text_rank) OVER (), 0)
                        END AS normalized_text_rank,
                        CASE
                            WHEN max(semantic_score) OVER () = min(semantic_score) OVER ()
                                THEN CASE WHEN max(semantic_score) OVER () > 0 THEN 1::float ELSE 0::float END
                            ELSE (semantic_score - min(semantic_score) OVER ())
                                / NULLIF(max(semantic_score) OVER () - min(semantic_score) OVER (), 0)
                        END AS normalized_semantic_score
                    FROM deduped
                )
                SELECT
                    user_message,
                    agent_response,
                    text_rank,
                    semantic_score,
                    (0.35 * normalized_text_rank) + (0.65 * normalized_semantic_score) AS hybrid_score
                FROM normalized
                ORDER BY hybrid_score DESC
                LIMIT %s
                """,
                (
                    session_id,
                    self.hybrid_exclude_recent_count,
                    query,
                    session_id,
                    query,
                    query,
                    limit,
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

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None):
        """Retrieve conversation history for context."""
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

    def has_exact_user_message(
        self,
        session_id: str,
        prompt: str,
        min_count: int = 2,
    ) -> bool:
        """Check whether a normalized prompt has appeared often enough to act as a session command alias."""
        normalized_prompt = (prompt or "").strip().lower()
        if not normalized_prompt:
            return False

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM conversations
                WHERE session_id = %s
                  AND lower(trim(coalesce(user_message, ''))) = %s
                """,
                (session_id, normalized_prompt),
            )
            return cursor.fetchone()[0] >= min_count

    def resolve_command_memory(
        self,
        session_id: str,
        prompt: str,
    ) -> Optional[str]:
        """Return a stable command response when one response clearly dominates a short trigger."""
        normalized_prompt = (prompt or "").strip().lower()
        if not normalized_prompt:
            return None

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                WITH ranked AS (
                    SELECT
                        response_text,
                        ROW_NUMBER() OVER (
                            ORDER BY last_seen_at DESC, id DESC
                        ) AS rank_index
                    FROM command_memory
                    WHERE session_id = %s
                      AND normalized_trigger = %s
                )
                SELECT response_text
                FROM ranked
                WHERE rank_index = 1
                """,
                (session_id, normalized_prompt),
            )
            row = cursor.fetchone()

        return row[0] if row else None

    def record_command_memory(self, session_id: str, user_message: str, agent_response: str):
        normalized_trigger = (user_message or "").strip().lower()
        if not normalized_trigger or not agent_response:
            return

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO command_memory (
                    session_id,
                    trigger_text,
                    normalized_trigger,
                    response_text,
                    first_seen_at,
                    last_seen_at
                )
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id, normalized_trigger, response_text)
                DO UPDATE SET
                    trigger_text = EXCLUDED.trigger_text,
                    last_seen_at = CURRENT_TIMESTAMP
                """,
                (session_id, user_message, normalized_trigger, agent_response),
            )
            self.db_connection.commit()

    def save_conversation(self, session_id: str, user_message: str, agent_response: str, embedding):
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (session_id, user_message, agent_response, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_message, agent_response, embedding),
            )
            self.db_connection.commit()

    def close(self):
        self.db_connection.close()
