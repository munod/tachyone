"""tachyone — local-first, multilingual System One decision engine.

tachyone answers typed ``choice`` / ``score`` / ``noul`` questions against a state and
speaks the TypeSafe Jev ``/v1/systemone`` wire contract as a drop-in.

This package's base install is importable without torch/transformers (NFR-R01);
heavy backends and transports live behind optional extras. See ``docs/architecture.md``.
"""

from __future__ import annotations

from tachyone.client import TachyoneAPIError, TachyoneClient, TachyoneConnectionError, TachyoneError
from tachyone.handoff import HandoffReport, HandoffSignal, Uncertainty, assess, assess_response

__version__ = "0.3.0"
__all__ = [
    "HandoffReport",
    "HandoffSignal",
    "TachyoneAPIError",
    "TachyoneClient",
    "TachyoneConnectionError",
    "TachyoneError",
    "Uncertainty",
    "__version__",
    "assess",
    "assess_response",
]
