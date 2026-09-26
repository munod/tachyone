"""Test doubles for the backend seam.

The implementations live in :mod:`tachyone.backends.fake` so the same deterministic backend is
available to the server/CLI runtime (``--backend fake``) and to the test suite. This module
re-exports them for local imports.
"""

from __future__ import annotations

from tachyone.backends.fake import (
    FakeBackend,
    ScriptedBackend,
    answer_question,
    estimate_usage,
)

__all__ = [
    "FakeBackend",
    "ScriptedBackend",
    "answer_question",
    "estimate_usage",
]
