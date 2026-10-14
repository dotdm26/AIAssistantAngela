"""Backward-compatible entry point for the Angela worker."""

from src.runtime.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
