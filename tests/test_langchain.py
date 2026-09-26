"""Tests for the LangChain adapter (M5-T4, OPS-03 / EXT-02)."""

from __future__ import annotations

import importlib.util

import pytest

from tachyone.backends.fake import FakeBackend
from tachyone.integrations.langchain import create_runnable, predict

_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "technical": "bugs"},
    },
}


def test_predict_returns_canonical_payload() -> None:
    payload = predict(FakeBackend(), "please refund", _QUESTIONS)
    assert set(payload) == {"model", "answers", "usage"}
    assert set(payload["answers"]) == {"urgent", "team"}


def test_create_runnable_without_extra_raises() -> None:
    if importlib.util.find_spec("langchain_core") is not None:
        pytest.skip("langchain_core is installed")
    with pytest.raises(RuntimeError, match="langchain extra"):
        create_runnable(FakeBackend())


def test_runnable_invocation() -> None:
    pytest.importorskip("langchain_core")
    runnable = create_runnable(FakeBackend())
    result = runnable.invoke({"state": "please refund", "questions": _QUESTIONS})
    assert set(result["answers"]) == {"urgent", "team"}
