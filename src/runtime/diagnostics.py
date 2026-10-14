"""Safe local diagnostics; returned data must not contain secret values."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from src.runtime.config import AppConfig


def run_diagnostics(config: AppConfig, *, discord_configured: bool, llm_configured: bool) -> list[dict]:
    checks = []

    def add(name: str, state: str, message: str):
        checks.append({"name": name, "state": state, "message": message})

    try:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        probe = config.data_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("data directory", "ready", str(config.data_dir))
    except Exception as exc:
        add("data directory", "failed", str(exc))

    free_bytes = shutil.disk_usage(config.data_dir).free
    add("disk space", "ready" if free_bytes > 250 * 1024 * 1024 else "degraded", f"{free_bytes // (1024 * 1024)} MiB free")
    add("Discord credentials", "ready" if discord_configured else "failed", "Configured" if discord_configured else "Not configured")
    add("LLM credentials", "ready" if llm_configured else "failed", "Configured" if llm_configured else "Not configured")
    add("SQLite support", "ready" if importlib.util.find_spec("sqlite3") else "failed", "Available")

    if config.storage.backend == "postgres":
        add("PostgreSQL configuration", "ready" if config.storage.database_url else "failed", "Configured" if config.storage.database_url else "Missing database URL")
    else:
        add("SQLite configuration", "ready", str(config.sqlite_path))
    return checks

