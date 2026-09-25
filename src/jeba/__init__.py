"""jeba — local-first, multilingual System One decision engine.

jeba answers typed ``choice`` / ``score`` / ``noul`` questions against a state and
speaks the TypeSafe Jev ``/v1/systemone`` wire contract as a drop-in.

This package's base install is importable without torch/transformers (NFR-R01);
heavy backends and transports live behind optional extras. See ``docs/architecture.md``.
"""

from __future__ import annotations

from jeba.client import JebaAPIError, JebaClient, JebaConnectionError, JebaError
from jeba.handoff import HandoffReport, HandoffSignal, Uncertainty, assess, assess_response

__version__ = "0.2.0"
__all__ = [
    "HandoffReport",
    "HandoffSignal",
    "JebaAPIError",
    "JebaClient",
    "JebaConnectionError",
    "JebaError",
    "Uncertainty",
    "__version__",
    "assess",
    "assess_response",
]
