"""Prediction and lifecycle hooks (additive extension).

Hooks observe (and may log/measure) every decision without changing the canonical wire shape
(EXT-01, EXT-02). A hook is either an object exposing any of the event methods below, or a
plain callable treated as an ``on_predict_end`` listener. A raising hook triggers ``on_error``
and serving continues; errors inside ``on_error`` are swallowed so hooks can never break a
request.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tachyone.primitives import Answer, Question, State

if TYPE_CHECKING:
    from tachyone.router import RouteDecision

ON_PREDICT_START = "on_predict_start"
ON_PREDICT_END = "on_predict_end"
ON_ROUTE = "on_route"
ON_LOAD = "on_load"
ON_EVICT = "on_evict"
ON_ERROR = "on_error"

EVENTS: tuple[str, ...] = (
    ON_PREDICT_START,
    ON_PREDICT_END,
    ON_ROUTE,
    ON_LOAD,
    ON_EVICT,
    ON_ERROR,
)


@dataclass(slots=True)
class HookContext:
    """State passed to hooks for a single event."""

    event: str
    state: State | None = None
    states: list[State] | None = None
    questions: dict[str, Question] | None = None
    model: str | None = None
    checkpoint_id: str | None = None
    route: RouteDecision | None = None
    answers: list[dict[str, Answer]] | None = None
    elapsed_ms: float | None = None
    error: Exception | None = None


@dataclass
class Hooks:
    """A registry that dispatches an event to every registered hook."""

    hooks: list[Any] = field(default_factory=list)

    def __init__(self, hooks: Iterable[Any] | None = None) -> None:
        self.hooks = list(hooks) if hooks is not None else []

    def register(self, hook: Any) -> None:
        self.hooks.append(hook)

    def emit(self, event: str, **fields: Any) -> HookContext:
        """Dispatch ``event`` to all hooks; never raises because of a hook."""
        context = HookContext(event=event, **fields)
        for hook in list(self.hooks):
            try:
                self._dispatch(hook, event, context)
            except Exception as exc:
                if event != ON_ERROR:
                    self._dispatch_error(exc, context)
        return context

    @staticmethod
    def _dispatch(hook: Any, event: str, context: HookContext) -> None:
        method = getattr(hook, event, None)
        if callable(method):
            method(context)
        elif callable(hook) and event == ON_PREDICT_END:
            hook(context)

    def _dispatch_error(self, exc: Exception, context: HookContext) -> None:
        error_context = HookContext(
            event=ON_ERROR,
            state=context.state,
            states=context.states,
            questions=context.questions,
            model=context.model,
            checkpoint_id=context.checkpoint_id,
            error=exc,
        )
        for hook in list(self.hooks):
            method = getattr(hook, ON_ERROR, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method(error_context)


__all__ = [
    "EVENTS",
    "ON_ERROR",
    "ON_EVICT",
    "ON_LOAD",
    "ON_PREDICT_END",
    "ON_PREDICT_START",
    "ON_ROUTE",
    "HookContext",
    "Hooks",
]
