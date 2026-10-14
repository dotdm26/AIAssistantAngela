"""SQLite conversation storage for the local desktop installation.

The public methods intentionally mirror ConversationStore so the agent can
switch backends without changing retrieval or persistence behavior.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage


class SQLiteConversationStore:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path,
        embedding_dim: int,
        local_embedding_model: str,
        hybrid_exclude_recent_count: int,
        rag_half_life_days: float = 30.0,
    ):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.local_embedding_model = local_embedding_model
        self.hybrid_exclude_recent_count = max(0, int(hybrid_exclude_recent_count))
        self.rag_half_life_days = max(1.0, float(rag_half_life_days))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    embedding_json TEXT,
                    embedding_model TEXT
                );

                CREATE INDEX IF NOT EXISTS conversations_session_created_idx
                    ON conversations(session_id, created_at DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                    user_message,
                    agent_response,
                    content='conversations',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, user_message, agent_response)
                    VALUES (new.id, new.user_message, new.agent_response);
                END;

                CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, user_message, agent_response)
                    VALUES ('delete', old.id, old.user_message, old.agent_response);
                END;
                """
            )
            applied = self._connection.execute(
                "SELECT version FROM schema_migrations WHERE version = ?",
                (self.SCHEMA_VERSION,),
            ).fetchone()
            if not applied:
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
                )

    async def hybrid_search_conversations(
        self,
        query: str,
        session_id: str,
        embedding_provider: Callable[[str], Awaitable[List[float]]],
        limit: int = 5,
    ):
        embedding = await embedding_provider(query)
        return await asyncio.to_thread(
            self._hybrid_search_sync,
            query,
            session_id,
            embedding,
            max(1, limit),
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _hybrid_search_sync(self, query: str, session_id: str, embedding: list[float], limit: int):
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, user_message, agent_response, created_at, embedding_json
                FROM conversations
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()

        recent_ids = {row["id"] for row in rows[: self.hybrid_exclude_recent_count]}
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        now = datetime.now(timezone.utc).timestamp()
        scored = []
        for row in rows:
            if row["id"] in recent_ids:
                continue
            combined = f"{row['user_message']} {row['agent_response']}".lower()
            matched_terms = sum(1 for term in query_terms if term in combined)
            text_score = 1.25 if query.lower() in combined and query.strip() else 0.0
            if query_terms:
                text_score = max(text_score, matched_terms / len(query_terms))
            saved_embedding = json.loads(row["embedding_json"] or "null")
            semantic_score = max(0.0, self._cosine_similarity(embedding, saved_embedding))
            if not text_score and not semantic_score:
                continue
            age_days = max(0.0, (now - float(row["created_at"])) / 86400)
            decay = math.exp(-math.log(2) * age_days / self.rag_half_life_days)
            hybrid_score = (0.45 * min(1.0, text_score) + 0.55 * semantic_score) * decay
            scored.append(
                (
                    row["user_message"],
                    row["agent_response"],
                    float(text_score),
                    float(semantic_score),
                    float(hybrid_score),
                )
            )

        scored.sort(key=lambda item: item[4], reverse=True)
        return scored[:limit]

    async def get_conversation_history(self, session_id: str, limit: Optional[int] = None):
        return await asyncio.to_thread(self._get_conversation_history_sync, session_id, limit)

    def _get_conversation_history_sync(self, session_id: str, limit: Optional[int] = None):
        query = """
            SELECT user_message, agent_response
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at DESC
        """
        params: list[object] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))

        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        history = []
        for row in reversed(rows):
            history.extend(
                [
                    HumanMessage(content=row["user_message"]),
                    AIMessage(content=row["agent_response"]),
                ]
            )
        return history

    def save_conversation(self, session_id: str, user_message: str, agent_response: str, embedding):
        if embedding is not None and len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match configured dimension {self.embedding_dim}."
            )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO conversations(
                    session_id, user_message, agent_response, created_at,
                    embedding_json, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_message,
                    agent_response,
                    datetime.now(timezone.utc).timestamp(),
                    json.dumps(list(embedding)) if embedding is not None else None,
                    self.local_embedding_model,
                ),
            )

    def close(self):
        with self._lock:
            self._connection.close()

