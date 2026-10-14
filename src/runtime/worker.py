"""Command-line worker used by the desktop supervisor and server deployments."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys

from src.runtime.config import load_config
from src.runtime.observability import configure_logging
from src.runtime.runtime import AngelaRuntime
from src.runtime.secrets import get_secret_store


def emit_event(event_type: str, **payload) -> None:
    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


async def run_worker(config_path=None) -> int:
    config = load_config(config_path)
    logger = configure_logging(config.data_dir)
    secrets = get_secret_store(desktop=os.getenv("ANGELA_DESKTOP") == "1")
    runtime = AngelaRuntime(
        config=config,
        secrets=secrets,
        health_callback=lambda status: emit_event("health", **status),
        logger=logger,
    )
    loop = asyncio.get_running_loop()

    async def request_stop() -> None:
        await runtime.stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, lambda: asyncio.create_task(request_stop()))
        except (NotImplementedError, RuntimeError):
            pass

    emit_event("state", state="starting")
    try:
        await runtime.run()
        emit_event("state", state="stopped")
        return 0
    except Exception as exc:
        logger.error("Angela worker stopped with an error: %s", exc)
        emit_event("state", state="failed", message=str(exc))
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Angela Discord worker")
    parser.add_argument("--config", dest="config_path", default=None)
    args = parser.parse_args(argv)
    return asyncio.run(run_worker(args.config_path))


if __name__ == "__main__":
    sys.exit(main())

