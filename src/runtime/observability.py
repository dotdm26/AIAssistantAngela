"""Structured logs and component health for the worker and desktop UI."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable


_SECRET_PATTERN = re.compile(r"(?i)(token|api[_-]?key|authorization|password|secret)([=:]\s*)([^\s,;]+)")


def redact(value: str) -> str:
    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(data_dir: Path) -> logging.Logger:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("angela")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        log_dir / "angela.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)
    return logger


@dataclass
class ComponentStatus:
    name: str
    state: str = "starting"
    message: str | None = None
    checked_at: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class HealthRegistry:
    def __init__(self, on_change: Callable[[ComponentStatus], None] | None = None):
        self._statuses: dict[str, ComponentStatus] = {}
        self._on_change = on_change

    def update(self, name: str, state: str, message: str | None = None) -> ComponentStatus:
        status = ComponentStatus(
            name=name,
            state=state,
            message=message,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
        self._statuses[name] = status
        if self._on_change:
            self._on_change(status)
        return status

    def snapshot(self) -> list[dict]:
        return [status.as_dict() for status in self._statuses.values()]

