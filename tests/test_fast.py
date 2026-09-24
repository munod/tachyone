"""Tests for the optional fast path and router override (M5-T2, NFR-C05 / EXT-03)."""

from __future__ import annotations

import importlib.util

import pytest

from jeba.fast import cuda_available, maybe_accelerate, router_override, tilelang_available
from jeba.router import ENGLISH, MULTILINGUAL, Router

pytestmark = pytest.mark.contract


def test_fast_path_unavailable_without_tilelang() -> None:
    if importlib.util.find_spec("tilelang") is not None:
        pytest.skip("tilelang is installed; the fallback assertion does not apply")
    assert tilelang_available() is False


def test_maybe_accelerate_falls_back_without_raising() -> None:
    sentinel = object()
    result = maybe_accelerate(sentinel, device="auto")
    assert result.target is sentinel
    assert result.accelerated is False
    assert result.reason


def test_maybe_accelerate_rejects_non_cuda_devices() -> None:
    sentinel = object()
    result = maybe_accelerate(sentinel, device="cpu")
    assert result.target is sentinel
    assert result.accelerated is False


def test_cuda_available_is_boolean() -> None:
    assert cuda_available() in (True, False)


def test_router_override_forces_model() -> None:
    decision = router_override(Router(), model=ENGLISH)
    assert decision is not None
    assert decision.checkpoint_id == ENGLISH
    assert decision.reason == "explicit model override"


def test_router_override_maps_language() -> None:
    router = Router()
    english = router_override(router, lang="en_US.UTF-8")
    portuguese = router_override(router, lang="pt")
    assert english is not None and english.checkpoint_id == ENGLISH
    assert portuguese is not None and portuguese.checkpoint_id == MULTILINGUAL


def test_router_override_returns_none_without_force() -> None:
    assert router_override(Router()) is None
    assert router_override(Router(), model="unknown") is None
