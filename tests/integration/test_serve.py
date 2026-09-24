"""Integration tests for the HTTP surface (M2-T3, SERVE-01 / WIRE-01 / WIRE-04).

Skipped cleanly when the ``serve`` extra is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from jeba.backends.fake import FakeBackend
from jeba.config import Config
from jeba.serve import create_app

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_REQUEST: dict[str, Any] = json.loads(
    (_FIXTURES / "systemone_request_multi.json").read_text(encoding="utf-8")
)


def _client(**env: str) -> TestClient:
    return TestClient(create_app(Config.from_env(env), FakeBackend()))


def test_systemone_returns_canonical_shape() -> None:
    response = _client().post("/v1/systemone", json=_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    assert set(body["answers"]) == set(_REQUEST["questions"])
    assert body["model"] == _REQUEST["model"]


def test_systemone_rejects_invalid_body_with_422() -> None:
    response = _client().post("/v1/systemone", json={"model": "m", "questions": {}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unprocessable_entity"


def test_systemone_rejects_malformed_json_with_422() -> None:
    response = _client().post(
        "/v1/systemone", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_primitive_limit_violation_maps_to_422() -> None:
    payload = {
        "state": "x",
        "model": "m",
        "questions": {"q": {"type": "score", "instructions": "rate", "criteria": ["only-one"]}},
    }
    response = _client().post("/v1/systemone", json=payload)
    assert response.status_code == 422


def test_auth_required_when_key_configured() -> None:
    client = _client(JEBA_API_KEY="secret")
    assert client.post("/v1/systemone", json=_REQUEST).status_code == 401
    assert (
        client.post(
            "/v1/systemone", json=_REQUEST, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = client.post("/v1/systemone", json=_REQUEST, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
