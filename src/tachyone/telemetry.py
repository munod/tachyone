"""Telemetry guard: tachyone sends nothing (ADR-0011).

This module exists so the policy is explicit and future-proof: there is no endpoint, no
install id, and no network I/O. ``TACHYONE_TELEMETRY`` and ``DO_NOT_TRACK`` are reserved so that
any future telemetry must be opt-out and can never block offline use (NFR-S05, OPS-08).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Reserved variables for a future opt-out telemetry implementation.
TELEMETRY_ENV = "TACHYONE_TELEMETRY"
DO_NOT_TRACK_ENV = "DO_NOT_TRACK"


def telemetry_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Always ``False``: tachyone has no telemetry."""
    del env
    return False


def do_not_track(env: Mapping[str, str] | None = None) -> bool:
    """Whether the user set ``DO_NOT_TRACK`` (honored regardless of the policy)."""
    source = os.environ if env is None else env
    return source.get(DO_NOT_TRACK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def endpoint() -> str | None:
    """Always ``None``: there is no telemetry endpoint."""
    return None


def record_event(name: str, **fields: object) -> None:
    """No-op: nothing is recorded or sent."""
    del name, fields
