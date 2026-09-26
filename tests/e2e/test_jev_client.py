"""End-to-end: an existing Jev client repointed at a live tachyone server.

Binds a real ephemeral port and drives the server with :class:`TachyoneClient`, proving the
SDK + transport + wire + backend path end to end (NFR-X01, NFR-X02). Skipped without the
``serve`` extra.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn

from tachyone.backends.fake import FakeBackend
from tachyone.client import TachyoneAPIError, TachyoneClient
from tachyone.config import Config
from tachyone.serve import create_app
from tachyone.wire import parse_request

pytestmark = pytest.mark.e2e

_REQUEST: dict[str, object] = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "systemone_request_multi.json").read_text(
        encoding="utf-8"
    )
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def running_server(**env: str) -> Iterator[str]:
    port = _free_port()
    app = create_app(Config.from_env(env), FakeBackend())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_repointed_jev_client_round_trip() -> None:
    request = parse_request(_REQUEST)
    with running_server() as base_url:
        response = TachyoneClient(base_url).system_one(
            request.state, request.questions, model=request.model
        )
    dumped = response.model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}
    assert set(response.answers) == set(request.questions)
    for question_id, question in request.questions.items():
        assert response.answers[question_id].type == question.type


def test_authenticated_round_trip() -> None:
    request = parse_request(_REQUEST)
    with running_server(TACHYONE_API_KEY="secret") as base_url:
        anonymous = TachyoneClient(base_url)
        with pytest.raises(TachyoneAPIError) as excinfo:
            anonymous.system_one(request.state, request.questions)
        assert excinfo.value.status_code == 401
        authorized = TachyoneClient(base_url, api_key="secret")
        assert set(authorized.system_one(request.state, request.questions).answers) == set(
            request.questions
        )
