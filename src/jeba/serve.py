"""``jeba-serve`` HTTP entry point.

The FastAPI application (``/v1/systemone`` + extensions) lands in M2 behind the
``serve`` extra. This module must stay importable without FastAPI so the console
script resolves on a base install; FastAPI is imported lazily inside ``main``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``jeba-serve`` console script."""
    del argv  # parsed after the server is implemented (M2)
    print(
        "jeba-serve is not implemented yet (M2). "
        "Install the server extra with: uv sync --extra serve",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
