"""Frozen-compatible Angela entry point.

The same source entry point is used by development, the GUI executable, and
the console worker executable produced by PyInstaller.
"""

from __future__ import annotations

import sys
from pathlib import Path


VERSION = "0.1.0"


def _is_frozen_worker() -> bool:
    return getattr(sys, "frozen", False) and Path(sys.executable).stem.lower() == "angela-worker"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--version" in arguments:
        print(VERSION)
        return 0
    if "--diagnose" in arguments:
        from src.runtime.config import load_config
        from src.runtime.diagnostics import run_diagnostics
        from src.runtime.secrets import get_secret_store

        config = load_config()
        secrets = get_secret_store(desktop=True)
        results = run_diagnostics(
            config,
            discord_configured=bool(secrets.get("discord_token")),
            llm_configured=bool(secrets.get("google_api_key")),
        )
        for result in results:
            print(f"{result['state'].upper()}: {result['name']} — {result['message']}")
        return 0 if all(result["state"] != "failed" for result in results) else 1
    if "--backup" in arguments or "--restore" in arguments:
        from src.runtime.cli import main as cli_main

        if "--backup" in arguments:
            index = arguments.index("--backup")
            return cli_main(["backup", *arguments[index + 1 :]])
        index = arguments.index("--restore")
        return cli_main(["restore", *arguments[index + 1 :]])
    if "--worker" in arguments or _is_frozen_worker():
        from src.runtime.worker import main as worker_main

        return worker_main([argument for argument in arguments if argument != "--worker"])

    from src.desktop.app import main as desktop_main

    desktop_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
