"""Command-line entry point for the Vigil v0.1 live demo."""

from __future__ import annotations

from typing import Sequence

from vigil.demo import main as demo_main


def main(argv: Sequence[str] | None = None) -> int:
    """Forward command-line arguments to the live demo."""
    return demo_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
