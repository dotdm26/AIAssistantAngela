"""Persistent, non-secret Angela configuration.

Secrets deliberately live in :mod:`src.runtime.secrets` instead of this file.
The configuration format is intentionally small and portable so it can be used
by the worker, the desktop controller, and future installers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def default_data_dir() -> Path:
    configured = os.getenv("ANGELA_DATA_DIR")
    if configured:
        return Path(configured).expanduser()

    if sys.platform == "win32":
        root = os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(root) / "Angela"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Angela"
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    return Path(xdg_data_home or (Path.home() / ".local" / "share")) / "angela"


class LLMConfig(BaseModel):
    provider: str = "google"
    model: str = "gemini-3.1-flash-lite"


class DiscordConfig(BaseModel):
    channel_id: str | None = None
    startup_user_id: str | None = None
    allowed_users: list[str] = Field(default_factory=list)
    respond_to_all_messages: bool = True


class MemoryConfig(BaseModel):
    history_limit: int = 10
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"


class StorageConfig(BaseModel):
    backend: str = "sqlite"
    database_url: str | None = None


class AppConfig(BaseModel):
    data_dir: Path = Field(default_factory=default_data_dir)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    news_feed_url: str = "http://feeds.bbci.co.uk/news/rss.xml"
    extra_instructions: str = ""
    hybrid_top_k: int = 5
    hybrid_min_prompt_chars: int = 24
    hybrid_exclude_recent_count: int = 4
    hybrid_min_score: float = 0.25
    hybrid_min_score_memory: float = 0.15
    rag_half_life_days: float = 30.0
    max_tool_rounds: int = 3
    tool_filter_min_score: float = 0.28

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "angela.db"

    def validate_runtime(self, *, discord_token: str | None, google_api_key: str | None) -> list[str]:
        problems: list[str] = []
        if not discord_token:
            problems.append("Discord bot token is not configured.")
        if self.llm.provider == "google" and not google_api_key:
            problems.append("Google/Gemini API key is not configured.")
        if self.storage.backend not in {"sqlite", "postgres"}:
            problems.append("Storage backend must be 'sqlite' or 'postgres'.")
        if self.storage.backend == "postgres" and not self.storage.database_url:
            problems.append("PostgreSQL storage requires storage.database_url or DATABASE_URL.")
        return problems


def _env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from disk, then apply supported development env overrides."""
    load_dotenv()
    data: dict[str, Any] = {}
    config_path = path or (default_data_dir() / "config.json")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"Angela configuration must be a JSON object: {config_path}")
        data = loaded

    llm = dict(data.get("llm") or {})
    discord = dict(data.get("discord") or {})
    memory = dict(data.get("memory") or {})
    storage = dict(data.get("storage") or {})

    llm.setdefault("provider", _env_or_none("LLM_PROVIDER") or "google")
    llm.setdefault("model", _env_or_none("LLM_MODEL") or "gemini-3.1-flash-lite")
    discord.setdefault("channel_id", _env_or_none("DISCORD_CHANNEL_ID"))
    discord.setdefault("startup_user_id", _env_or_none("USER_ID"))
    discord.setdefault(
        "allowed_users",
        [value for value in (_env_or_none("user1"), _env_or_none("user2")) if value],
    )
    memory.setdefault("history_limit", int(os.getenv("HISTORY_LIMIT", "10")))
    memory.setdefault("embedding_model", os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"))
    storage.setdefault(
        "backend",
        os.getenv("ANGELA_STORAGE_BACKEND", "postgres" if os.getenv("DATABASE_URL") else "sqlite"),
    )
    storage.setdefault("database_url", _env_or_none("DATABASE_URL"))

    data["llm"] = llm
    data["discord"] = discord
    data["memory"] = memory
    data["storage"] = storage
    data.setdefault("data_dir", str(config_path.parent))
    data.setdefault("news_feed_url", os.getenv("NEWS_FEED_URL", "http://feeds.bbci.co.uk/news/rss.xml"))
    data.setdefault("extra_instructions", os.getenv("EXTRA_INSTRUCTIONS", ""))

    for field_name, env_name, cast in (
        ("hybrid_top_k", "HYBRID_TOP_K", int),
        ("hybrid_min_prompt_chars", "HYBRID_MIN_PROMPT_CHARS", int),
        ("hybrid_exclude_recent_count", "HYBRID_EXCLUDE_RECENT_COUNT", int),
        ("hybrid_min_score", "HYBRID_MIN_SCORE", float),
        ("hybrid_min_score_memory", "HYBRID_MIN_SCORE_MEMORY", float),
        ("rag_half_life_days", "RAG_HALF_LIFE_DAYS", float),
        ("max_tool_rounds", "MAX_TOOL_ROUNDS", int),
        ("tool_filter_min_score", "TOOL_FILTER_MIN_SCORE", float),
    ):
        if field_name not in data and os.getenv(env_name) is not None:
            data[field_name] = cast(os.environ[env_name])

    return AppConfig.model_validate(data)


def save_config(config: AppConfig) -> Path:
    """Atomically persist non-secret configuration."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    path = config.config_path
    temporary_path = path.with_suffix(".json.tmp")
    payload = json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)
    temporary_path.write_text(payload + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path

