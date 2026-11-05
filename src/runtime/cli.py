"""Command-line operations used by development and packaged builds."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.backup import backup_data, restore_data
from src.runtime.config import default_data_dir, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Angela", description="Angela local runtime utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup_parser = subparsers.add_parser("backup", help="Create a non-secret Angela data backup")
    backup_parser.add_argument("destination", type=Path)
    backup_parser.add_argument("--data-dir", type=Path)
    backup_parser.add_argument("--include-logs", action="store_true")
    backup_parser.add_argument("--force", action="store_true", help="Replace an existing archive")

    restore_parser = subparsers.add_parser("restore", help="Restore a backup into an Angela data directory")
    restore_parser.add_argument("archive", type=Path)
    restore_parser.add_argument("--data-dir", type=Path)
    restore_parser.add_argument("--force", action="store_true", help="Replace existing application data")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser() if args.data_dir else default_data_dir()
    if args.command == "backup":
        config = load_config(data_dir / "config.json")
        destination = backup_data(
            config,
            args.destination,
            include_logs=args.include_logs,
            overwrite=args.force,
        )
        print(f"Backup written to {destination}")
        return 0

    restored = restore_data(args.archive, data_dir, overwrite=args.force)
    print(f"Backup restored to {restored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
