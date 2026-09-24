"""The pluggable backend seam.

Every engine (LLM, local encoder, ONNX) implements :class:`Backend`. ``wire.py`` couples
only to this interface, so swapping backends never changes the ``/v1/systemone`` shape.
See ``docs/architecture.md`` and ``ADR-0002``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jeba.primitives import Answer, Question, State

if TYPE_CHECKING:
    from jeba.wire import Usage


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """What a backend returns: typed answers keyed by question id, plus token usage."""

    answers: dict[str, Answer]
    usage: Usage


@runtime_checkable
class Backend(Protocol):
    """Stable inference interface; implementations must not leak into the wire."""

    name: str

    async def predict(
        self,
        questions: dict[str, Question],
        *,
        state: State,
        model: str,
        return_details: bool = False,
    ) -> PredictionResult:
        """Answer every question against ``state`` and report token usage."""
        ...


__all__ = ["Backend", "PredictionResult"]
