"""Tests for the MCP tool logic and server (M5-T3, OPS-02)."""

from __future__ import annotations

import pytest

from jeba.backends.fake import FakeBackend
from jeba.mcp.server import create_server, predict_payload

_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "technical": "bugs"},
    },
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
}


def test_predict_payload_returns_canonical_shape() -> None:
    payload = predict_payload(FakeBackend(), "please refund", _QUESTIONS)
    assert set(payload) == {"model", "answers", "usage"}
    assert set(payload["answers"]) == {"department", "urgent"}


def test_predict_payload_rejects_invalid_questions() -> None:
    from jeba.wire import UnprocessableEntity

    with pytest.raises(UnprocessableEntity):
        predict_payload(FakeBackend(), "x", {"q": {"type": "score", "criteria": ["only"]}})


def test_create_server_without_extra_raises() -> None:
    import importlib.util

    if importlib.util.find_spec("mcp") is not None:
        pytest.skip("mcp is installed")
    with pytest.raises(RuntimeError, match="mcp extra"):
        create_server(FakeBackend())


def test_create_server_exposes_tool() -> None:
    pytest.importorskip("mcp")
    server = create_server(FakeBackend())
    assert server is not None
