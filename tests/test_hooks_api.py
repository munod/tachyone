"""Hook API contract tests (M3-T6, EXT-01 / EXT-02).

The hook surface is part of the public API (like Laya's ``tests/test_hooks_api.py``): hooks
must fire on every event, must never alter the canonical shape, and a raising hook must be
contained by ``on_error``.
"""

from __future__ import annotations

from typing import Any

import pytest

from jeba.agent import Agent
from jeba.hooks import (
    ON_ERROR,
    ON_EVICT,
    ON_LOAD,
    ON_PREDICT_END,
    ON_PREDICT_START,
    ON_ROUTE,
    HookContext,
    Hooks,
)
from jeba.primitives import NoulAnswer, NoulQuestion, Question, State
from jeba.router import ENGLISH, MULTILINGUAL, CheckpointInfo, Router

pytestmark = pytest.mark.contract


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.contexts: dict[str, HookContext] = {}

    def _record(self, context: HookContext) -> None:
        self.events.append(context.event)
        self.contexts[context.event] = context

    def on_predict_start(self, context: HookContext) -> None:
        self._record(context)

    def on_predict_end(self, context: HookContext) -> None:
        self._record(context)

    def on_route(self, context: HookContext) -> None:
        self._record(context)

    def on_load(self, context: HookContext) -> None:
        self._record(context)

    def on_evict(self, context: HookContext) -> None:
        self._record(context)

    def on_error(self, context: HookContext) -> None:
        self._record(context)


def _runner(states: list[State], _questions: dict[str, Question]) -> list[dict[str, Any]]:
    return [{"q": NoulAnswer(noul=0.5)} for _ in states]


_QUESTIONS: dict[str, Question] = {"q": NoulQuestion(instructions="Is it so?")}


def test_every_event_dispatches_to_the_object_method() -> None:
    recorder = _Recorder()
    hooks = Hooks([recorder])
    for event in (ON_PREDICT_START, ON_PREDICT_END, ON_ROUTE, ON_LOAD, ON_EVICT, ON_ERROR):
        hooks.emit(event)
    assert recorder.events == [
        ON_PREDICT_START,
        ON_PREDICT_END,
        ON_ROUTE,
        ON_LOAD,
        ON_EVICT,
        ON_ERROR,
    ]


def test_plain_callable_listens_on_predict_end() -> None:
    seen: list[str] = []
    hooks = Hooks([lambda context: seen.append(context.event)])
    hooks.emit(ON_PREDICT_START)
    hooks.emit(ON_PREDICT_END)
    assert seen == [ON_PREDICT_END]


def test_raising_hook_triggers_on_error_and_does_not_propagate() -> None:
    calls: list[str] = []

    class Boom:
        def on_predict_end(self, _context: HookContext) -> None:
            calls.append("end")
            raise RuntimeError("boom")

        def on_error(self, context: HookContext) -> None:
            calls.append(f"error:{type(context.error).__name__}")

    hooks = Hooks([Boom()])
    hooks.emit(ON_PREDICT_END)  # must not raise
    assert calls == ["end", "error:RuntimeError"]


def test_error_in_on_error_is_swallowed() -> None:
    class Bad:
        def on_error(self, _context: HookContext) -> None:
            raise RuntimeError("nested")

    Hooks([Bad()]).emit(ON_ERROR)  # must not raise


def test_agent_emits_predict_events() -> None:
    recorder = _Recorder()
    Agent(_runner, hooks=Hooks([recorder])).predict("hello", _QUESTIONS)
    assert recorder.events == [ON_PREDICT_START, ON_PREDICT_END]
    end = recorder.contexts[ON_PREDICT_END]
    assert end.answers and "q" in end.answers[0]
    assert end.elapsed_ms is not None


def test_agent_emits_error_and_reraises() -> None:
    def failing(_states: list[State], _questions: dict[str, Question]) -> list[dict[str, Any]]:
        raise ValueError("runner failed")

    recorder = _Recorder()
    agent = Agent(failing, hooks=Hooks([recorder]))
    with pytest.raises(ValueError, match="runner failed"):
        agent.predict("hello", _QUESTIONS)
    assert ON_ERROR in recorder.events


def test_router_emits_route_load_and_evict() -> None:
    recorder = _Recorder()

    def loader(info: CheckpointInfo) -> object:
        return {"id": info.id}

    router = Router(loader=loader, max_loaded=1, hooks=Hooks([recorder]))
    router.route("Please refund")
    router.get(ENGLISH)
    router.get(MULTILINGUAL)  # evicts English
    assert recorder.events == [ON_ROUTE, ON_LOAD, ON_EVICT, ON_LOAD]
    assert recorder.contexts[ON_ROUTE].checkpoint_id == ENGLISH
