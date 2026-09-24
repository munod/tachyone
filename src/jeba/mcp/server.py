"""``jeba-mcp-server`` stdio entry point (M5).

The MCP server lands in M5 behind the ``mcp`` extra; this stub stays importable
without that dependency.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``jeba-mcp-server`` console script."""
    del argv  # parsed after the MCP server is implemented (M5)
    print(
        "jeba-mcp-server is not implemented yet (M5). "
        "Install the MCP extra with: uv sync --extra mcp",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
