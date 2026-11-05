"""Schema migration helpers for portable Angela configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CURRENT_CONFIG_SCHEMA = 1


def migrate_config_data(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Upgrade a persisted config dictionary and report whether it changed.

    The first local-runtime release did not persist a schema version and some
    development configurations used top-level fields.  Keep those files
    readable while giving future releases an explicit migration point.
    """
    migrated = deepcopy(data)
    raw_version = migrated.get("schema_version", 0)
    try:
        version = int(raw_version or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Angela configuration schema_version must be an integer.") from exc

    if version > CURRENT_CONFIG_SCHEMA:
        raise ValueError(
            f"Angela configuration schema {version} is newer than supported schema "
            f"{CURRENT_CONFIG_SCHEMA}. Upgrade Angela before opening this file."
        )

    if version < 1:
        # Migrate the short-lived prototype format into the structured local
        # runtime format. setdefault preserves fields from newer/manual files.
        discord = dict(migrated.get("discord") or {})
        if migrated.get("channel_id") is not None:
            discord.setdefault("channel_id", migrated["channel_id"])
        if discord:
            migrated["discord"] = discord

        storage = dict(migrated.get("storage") or {})
        if migrated.get("database_url") is not None:
            storage.setdefault("database_url", migrated["database_url"])
            storage.setdefault("backend", "postgres")
        if storage:
            migrated["storage"] = storage

        memory = dict(migrated.get("memory") or {})
        if migrated.get("embedding_model") is not None:
            memory.setdefault("embedding_model", migrated["embedding_model"])
        if memory:
            migrated["memory"] = memory

        migrated["schema_version"] = 1
        migrated = {key: value for key, value in migrated.items() if key not in {"channel_id", "database_url", "embedding_model"}}
        return migrated, True

    return migrated, False
