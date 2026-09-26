"""Unit tests for the SDK client (M2-T5, SERVE-05 / WIRE-05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tachyone
from tachyone.client import TachyoneAPIError, TachyoneClient, TachyoneConnectionError
from tachyone.primitives import NoulQuestion

_FIXTURES = Path(__file__).parent / "fixtures"
_REQUEST: dict[str, Any] = json.loads(
    (_FIXTURES / "systemone_request_multi.json").read_text(encoding="utf-8")
)
_RESPONSE_TEXT = (_FIXTURES / "systemone_response_multi.json").read_text(encoding="utf-8")


class _Transport:
    """Records calls and replays queued ``(status, body)`` responses."""

    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, data: bytes, headers: dict[str, str], timeout: float
    ) -> tuple[int, str]:
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


def _client(transport: _Transport, **kwargs: Any) -> TachyoneClient:
    return TachyoneClient(transport=transport, sleep=lambda _d: None, **kwargs)


def test_client_is_exported() -> None:
    assert tachyone.TachyoneClient is TachyoneClient


def test_system_one_posts_canonical_payload() -> None:
    transport = _Transport([(200, _RESPONSE_TEXT)])
    response = _client(transport).system_one(
        _REQUEST["state"],
        {
            "is_urgent": NoulQuestion(instructions="urgent?"),
        },
    )
    assert "is_urgent" in response.answers
    sent = json.loads(transport.calls[0]["data"])
    assert sent["model"] == "tachyone-latest"
    assert sent["questions"]["is_urgent"]["type"] == "noul"
    assert transport.calls[0]["url"].endswith("/v1/systemone")


def test_api_key_is_sent_as_bearer() -> None:
    transport = _Transport([(200, _RESPONSE_TEXT)])
    _client(transport, api_key="secret").system_one(
        {"body": "x"}, {"q": NoulQuestion(instructions="q?")}
    )
    assert transport.calls[0]["headers"]["authorization"] == "Bearer secret"


def test_retries_on_429_then_succeeds() -> None:
    transport = _Transport([(429, "{}"), (200, _RESPONSE_TEXT)])
    delays: list[float] = []
    client = TachyoneClient(transport=transport, sleep=delays.append, backoff=0.01, max_retries=3)
    response = client.system_one({"body": "x"}, {"q": NoulQuestion(instructions="q?")})
    assert set(response.answers)
    assert len(transport.calls) == 2
    assert len(delays) == 1
    assert delays[0] > 0


def test_raises_after_exhausting_retries() -> None:
    transport = _Transport([(529, "{}"), (529, "{}")])
    client = _client(transport, max_retries=1)
    with pytest.raises(TachyoneAPIError) as excinfo:
        client.system_one({"body": "x"}, {"q": NoulQuestion(instructions="q?")})
    assert excinfo.value.status_code == 529
    assert len(transport.calls) == 2


def test_non_retryable_error_carries_body() -> None:
    transport = _Transport([(422, '{"error": {"code": "unprocessable_entity"}}')])
    with pytest.raises(TachyoneAPIError) as excinfo:
        _client(transport).system_one({"body": "x"}, {"q": NoulQuestion(instructions="q?")})
    assert excinfo.value.status_code == 422
    assert excinfo.value.body["error"]["code"] == "unprocessable_entity"


def test_connection_error_propagates() -> None:
    def transport(
        _url: str, _data: bytes, _headers: dict[str, str], _timeout: float
    ) -> tuple[int, str]:
        raise TachyoneConnectionError("down")

    with pytest.raises(TachyoneConnectionError):
        TachyoneClient(transport=transport).system_one(
            {"body": "x"}, {"q": NoulQuestion(instructions="q?")}
        )
