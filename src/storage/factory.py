"""Select a conversation storage backend from application configuration."""

from __future__ import annotations

from src.runtime.config import AppConfig
from src.storage.sqlite_store import SQLiteConversationStore


def create_conversation_store(config: AppConfig, embedding_dim: int):
    if config.storage.backend == "sqlite":
        return SQLiteConversationStore(
            database_path=config.sqlite_path,
            embedding_dim=embedding_dim,
            local_embedding_model=config.memory.embedding_model,
            hybrid_exclude_recent_count=config.hybrid_exclude_recent_count,
            rag_half_life_days=config.rag_half_life_days,
        )

    if not config.storage.database_url:
        raise ValueError("PostgreSQL storage requires a database URL.")

    # Keep the existing production backend intact for the server/Docker mode.
    from src.conversation_store import ConversationStore

    return ConversationStore(
        database_url=config.storage.database_url,
        embedding_dim=embedding_dim,
        local_embedding_model=config.memory.embedding_model,
        hybrid_exclude_recent_count=config.hybrid_exclude_recent_count,
        rag_half_life_days=config.rag_half_life_days,
    )

