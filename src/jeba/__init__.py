"""jeba — local-first, multilingual System One decision engine.

jeba answers typed ``choice`` / ``score`` / ``noul`` questions against a state and
speaks the TypeSafe Jev ``/v1/systemone`` wire contract as a drop-in.

This package's base install is importable without torch/transformers (NFR-R01);
heavy backends and transports live behind optional extras. See ``docs/architecture.md``.
"""

from __future__ import annotations

__version__ = "0.0.1"
__all__ = ["__version__"]
