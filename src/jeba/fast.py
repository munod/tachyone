"""Optional accelerated execution path (TileLang / CUDA graphs).

The fast path is opt-in and must degrade gracefully: when the ``fast`` extra or a supported
CUDA device is missing, callers get the stock object back unchanged (NFR-C05, OPS-01). The
router override is additive and never changes the canonical response (EXT-03).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from jeba.router import ENGLISH, MULTILINGUAL, RouteDecision, Router

#: Checkpoint ids to which a forced language maps.
_ENGLISH_CODES = frozenset({"en", "eng", "english"})


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def cuda_available() -> bool:
    """Whether a CUDA device is usable (torch may be absent)."""
    if not _module_available("torch"):
        return False
    try:
        import torch  # pyright: ignore[reportMissingImports]

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def tilelang_available() -> bool:
    """Whether the TileLang fast path can run (extra installed and CUDA present)."""
    return _module_available("tilelang") and cuda_available()


@dataclass(frozen=True, slots=True)
class Acceleration:
    """Outcome of trying to accelerate an object."""

    target: Any
    accelerated: bool
    reason: str


def maybe_accelerate(target: object, *, device: str = "auto") -> Acceleration:
    """Return ``target`` accelerated when possible, otherwise unchanged with a reason."""
    if device not in {"auto", "cuda"}:
        return Acceleration(
            target=target, accelerated=False, reason=f"device {device!r} has no fast path"
        )
    if not _module_available("tilelang"):
        return Acceleration(target=target, accelerated=False, reason="tilelang not installed")
    if not cuda_available():
        return Acceleration(target=target, accelerated=False, reason="no CUDA device")
    # The TileLang kernels plug in here; until then the stock target is already optimal.
    return Acceleration(
        target=target, accelerated=False, reason="no accelerated kernels registered"
    )


def router_override(
    router: Router, *, model: str | None = None, lang: str | None = None
) -> RouteDecision | None:
    """Build an explicit routing decision (EXT-03), or ``None`` when nothing is forced."""
    if model and model in router.checkpoints:
        return RouteDecision(
            checkpoint_id=model,
            detected_script="override",
            detected_language=lang,
            reason="explicit model override",
        )
    if lang:
        primary = lang.replace("_", "-").split("-")[0].lower()
        checkpoint_id = ENGLISH if primary in _ENGLISH_CODES else MULTILINGUAL
        return RouteDecision(
            checkpoint_id=checkpoint_id,
            detected_script="override",
            detected_language=primary,
            reason="explicit language override",
        )
    return None


__all__ = [
    "Acceleration",
    "cuda_available",
    "maybe_accelerate",
    "router_override",
    "tilelang_available",
]
