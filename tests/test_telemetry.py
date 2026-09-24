"""Tests for the telemetry guard (M5-T6, OPS-08 / NFR-S05)."""

from __future__ import annotations

import socket

import pytest

from jeba.telemetry import (
    DO_NOT_TRACK_ENV,
    TELEMETRY_ENV,
    do_not_track,
    endpoint,
    record_event,
    telemetry_enabled,
)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {TELEMETRY_ENV: "1"},
        {TELEMETRY_ENV: "0", DO_NOT_TRACK_ENV: "1"},
        {DO_NOT_TRACK_ENV: "true"},
    ],
)
def test_telemetry_is_always_disabled(env: dict[str, str]) -> None:
    assert telemetry_enabled(env) is False
    assert endpoint() is None


def test_do_not_track_is_honored() -> None:
    assert do_not_track({"DO_NOT_TRACK": "1"}) is True
    assert do_not_track({"DO_NOT_TRACK": "0"}) is False
    assert do_not_track({}) is False


def test_record_event_is_a_noop_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("telemetry attempted network access")

    monkeypatch.setattr(socket, "socket", _no_socket)
    assert record_event("predict", backend="fake", ok=True) is None
