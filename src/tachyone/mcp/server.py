"""MCP stdio server exposing tachyone as a tool for agent hosts.

The prediction logic is plain Python (:func:`predict_payload`) so it is testable without the
``mcp`` extra; the stdio server is built lazily and only needs ``mcp`` when actually run
(OPS-02, SERVE-04).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from tachyone.backends import build_backend
from tachyone.backends.base import Backend
from tachyone.config import Config
from tachyone.wire import answer, parse_request

_MCP_HINT = "the mcp extra is required for the MCP server: uv sync --extra mcp"


def predict_payload(
    backend: Backend, state: str, questions: dict[str, Any], model: str = "tachyone-latest"
) -> dict[str, Any]:
    """Answer ``questions`` for ``state`` and return the canonical response payload."""
    request = parse_request({"state": state, "model": model, "questions": questions})
    import asyncio

    response = asyncio.run(answer(request, backend))
    return response.model_dump(mode="json")


def create_server(backend: Backend) -> Any:
    """Build an MCP server exposing ``tachyone_predict`` over stdio."""
    try:
        from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(_MCP_HINT) from exc

    server = FastMCP("tachyone")

    @server.tool()
    def tachyone_predict(
        state: str, questions: dict[str, Any], model: str = "tachyone-latest"
    ) -> dict[str, Any]:
        """Answer typed choice/score/noul questions about ``state`` and return the answers."""
        return predict_payload(backend, state, questions, model)

    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``tachyone-mcp-server`` console script."""
    del argv
    try:
        backend = build_backend(Config.from_env())
        server = create_server(backend)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
