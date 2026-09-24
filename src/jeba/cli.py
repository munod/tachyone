"""``jeba`` command-line entry point.

Runtime wiring lands in later milestones (M2+). At M0 this is an importable stub so
the ``jeba`` console script resolves without importing optional dependencies.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jeba import __version__

_PRESETS = ("router", "guard", "moderation", "triage", "email")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``jeba`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="jeba",
        description="Local-first System One decision engine (Jev /v1/systemone compatible).",
    )
    parser.add_argument("text", nargs="?", help="state text to evaluate")
    parser.add_argument("--preset", choices=_PRESETS, help="answer a ready-made question set")
    parser.add_argument("--predict", action="store_true", help="run inference (loads a backend)")
    parser.add_argument("--version", action="version", version=f"jeba {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``jeba`` console script."""
    build_parser().parse_args(argv)
    print("jeba CLI is not wired yet; inference arrives in M2. See docs/roadmap.md.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
