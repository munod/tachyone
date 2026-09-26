"""LangChain/LangGraph adapter returning canonical tachyone primitives.

The prediction helper is plain Python so it is testable without the extra; the ``Runnable``
wrapper is built lazily and only needs ``langchain-core`` when used (OPS-03, EXT-02).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from tachyone.backends.base import Backend
from tachyone.wire import answer, parse_request

_LANGCHAIN_HINT = "the langchain extra is required: uv sync --extra langchain"


def predict(
    backend: Backend,
    state: str,
    questions: Mapping[str, Any],
    *,
    model: str = "tachyone-latest",
) -> dict[str, Any]:
    """Return the canonical ``/v1/systemone`` response payload for one state."""
    request = parse_request({"state": state, "model": model, "questions": dict(questions)})
    response = asyncio.run(answer(request, backend))
    return response.model_dump(mode="json")


def create_runnable(backend: Backend, *, model: str = "tachyone-latest") -> Any:
    """Wrap the backend as a LangChain ``Runnable``.

    The runnable accepts ``{"state": str, "questions": {...}, "model"?: str}`` and returns
    the canonical response payload.
    """
    try:
        from langchain_core.runnables import RunnableLambda  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(_LANGCHAIN_HINT) from exc

    def invoke(payload: Mapping[str, Any]) -> dict[str, Any]:
        return predict(
            backend,
            payload["state"],
            payload["questions"],
            model=str(payload.get("model", model)),
        )

    return RunnableLambda(invoke)


__all__ = ["create_runnable", "predict"]
