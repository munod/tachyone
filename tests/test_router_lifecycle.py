"""Unit tests for checkpoint lifecycle management (M3-T2, ROUTE-04)."""

from __future__ import annotations

import pytest

from jeba.router import ENGLISH, MULTILINGUAL, CheckpointInfo, Router


class _Loader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, info: CheckpointInfo) -> dict[str, str]:
        self.calls.append(info.id)
        return {"id": info.id}


def test_get_loads_and_caches() -> None:
    loader = _Loader()
    router = Router(loader=loader)
    first = router.get(ENGLISH)
    second = router.get(ENGLISH)
    assert first is second
    assert loader.calls == [ENGLISH]
    assert router.loaded == (ENGLISH,)


def test_preload_loads_all_registered() -> None:
    router = Router(loader=_Loader())
    router.preload()
    assert set(router.loaded) == {ENGLISH, MULTILINGUAL}


def test_preload_subset() -> None:
    router = Router(loader=_Loader())
    router.preload([ENGLISH])
    assert router.loaded == (ENGLISH,)


def test_max_loaded_one_evicts_lru() -> None:
    loader = _Loader()
    router = Router(loader=loader, max_loaded=1)
    router.get(ENGLISH)
    router.get(MULTILINGUAL)
    assert router.loaded == (MULTILINGUAL,)
    router.get(ENGLISH)
    assert router.loaded == (ENGLISH,)
    assert loader.calls == [ENGLISH, MULTILINGUAL, ENGLISH]


def test_recent_use_is_not_evicted() -> None:
    router = Router(loader=_Loader(), max_loaded=2)
    router.get(ENGLISH)
    router.get(MULTILINGUAL)
    router.get(ENGLISH)  # touch English -> multilingual is now LRU
    router.attach("jeba-third", object())
    assert MULTILINGUAL not in router.loaded
    assert ENGLISH in router.loaded


def test_max_loaded_zero_is_unlimited() -> None:
    router = Router(loader=_Loader(), max_loaded=0)
    router.preload()
    assert set(router.loaded) == {ENGLISH, MULTILINGUAL}


def test_attach_avoids_loader() -> None:
    loader = _Loader()
    router = Router(loader=loader)
    sentinel = object()
    router.attach(ENGLISH, sentinel)
    assert router.get(ENGLISH) is sentinel
    assert loader.calls == []


def test_unload_one_and_all() -> None:
    router = Router(loader=_Loader())
    router.preload()
    router.unload(ENGLISH)
    assert ENGLISH not in router.loaded
    router.unload()
    assert router.loaded == ()


def test_get_without_loader_raises() -> None:
    with pytest.raises(RuntimeError, match="no loader"):
        Router().get(ENGLISH)


def test_get_unknown_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="unknown checkpoint"):
        Router(loader=_Loader()).get("nope")
