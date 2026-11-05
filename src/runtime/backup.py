"""Portable backup and restore for non-secret Angela application data."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from src.runtime.config import AppConfig
from src.runtime.migrations import CURRENT_CONFIG_SCHEMA, migrate_config_data


BACKUP_FORMAT_VERSION = 1


def _write_sqlite_snapshot(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def backup_data(
    config: AppConfig,
    destination: str | Path,
    *,
    include_logs: bool = False,
    overwrite: bool = False,
) -> Path:
    """Create a zip backup without exporting credentials from the keyring."""
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Backup already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = config.model_dump(mode="json")
    manifest: dict[str, Any] = {
        "format_version": BACKUP_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_schema_version": CURRENT_CONFIG_SCHEMA,
        "contains_secrets": False,
        "files": ["config.json"],
    }

    temporary_destination = destination_path.with_name(destination_path.name + ".tmp")
    if temporary_destination.exists():
        temporary_destination.unlink()

    with tempfile.TemporaryDirectory(prefix="angela-backup-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        database_source = config.sqlite_path
        database_snapshot = temporary_root / "angela.db"
        if database_source.exists():
            _write_sqlite_snapshot(database_source, database_snapshot)
            manifest["files"].append("angela.db")

        log_files: list[tuple[Path, str]] = []
        if include_logs:
            logs_directory = config.data_dir / "logs"
            for log_path in logs_directory.rglob("*") if logs_directory.exists() else ():
                if log_path.is_file():
                    archive_name = str(Path("logs") / log_path.relative_to(logs_directory)).replace("\\", "/")
                    log_files.append((log_path, archive_name))
                    manifest["files"].append(archive_name)

        with zipfile.ZipFile(temporary_destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            archive.writestr("config.json", json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
            if database_snapshot.exists():
                archive.write(database_snapshot, "angela.db")
            for log_path, archive_name in log_files:
                archive.write(log_path, archive_name)

    temporary_destination.replace(destination_path)
    return destination_path


def _validate_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe backup entry: {name}")
    if name not in {"manifest.json", "config.json", "angela.db"} and not name.startswith("logs/"):
        raise ValueError(f"Unsupported backup entry: {name}")


def _load_backup_config(archive: zipfile.ZipFile, data_dir: Path) -> dict[str, Any] | None:
    if "config.json" not in archive.namelist():
        return None
    raw = json.loads(archive.read("config.json").decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Backup config.json must contain an object.")
    migrated, _ = migrate_config_data(raw)
    migrated["data_dir"] = str(data_dir)
    return AppConfig.model_validate(migrated).model_dump(mode="json")


def restore_data(
    archive_path: str | Path,
    data_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Restore a backup, preserving a safety backup before replacement."""
    archive_path = Path(archive_path).expanduser().resolve()
    target_directory = Path(data_dir).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Backup does not exist: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        for name in archive.namelist():
            _validate_archive_name(name)
        if "manifest.json" not in archive.namelist():
            raise ValueError("Backup is missing manifest.json.")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
            raise ValueError("Unsupported Angela backup format.")
        restored_config = _load_backup_config(archive, target_directory)
        entries = set(archive.namelist())

        existing_files = [
            path for path in (target_directory / "config.json", target_directory / "angela.db") if path.exists()
        ]
        if existing_files and not overwrite:
            raise FileExistsError(
                "Angela data already exists. Use --force to restore over it: "
                + ", ".join(str(path) for path in existing_files)
            )

        target_directory.mkdir(parents=True, exist_ok=True)
        safety_backup = target_directory / "pre-restore-backup.zip"
        if existing_files and overwrite:
            current_config = AppConfig(data_dir=target_directory)
            if (target_directory / "config.json").exists():
                current_config_data = json.loads((target_directory / "config.json").read_text(encoding="utf-8"))
                current_config_data["data_dir"] = str(target_directory)
                current_config = AppConfig.model_validate(current_config_data)
            backup_data(current_config, safety_backup, overwrite=True)

        with tempfile.TemporaryDirectory(prefix="angela-restore-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            if "angela.db" in entries:
                with archive.open("angela.db") as source, (temporary_root / "angela.db").open("wb") as target:
                    shutil.copyfileobj(source, target)
                sqlite3.connect(temporary_root / "angela.db").close()

            if restored_config is not None:
                (temporary_root / "config.json").write_text(
                    json.dumps(restored_config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            for name in entries:
                if not name.startswith("logs/"):
                    continue
                output = temporary_root / name
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)

            for name in ("config.json", "angela.db"):
                source = temporary_root / name
                if source.exists():
                    source.replace(target_directory / name)
            logs_source = temporary_root / "logs"
            if logs_source.exists():
                shutil.copytree(logs_source, target_directory / "logs", dirs_exist_ok=True)

    return target_directory
