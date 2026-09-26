"""Batch orchestration for the local encoder.

An :class:`Agent` owns a ``runner`` — the function that performs one forward pass over a list
of states and returns aligned answers (the encoder backend supplies the torch one; tests
supply a fake). The agent adds the parts the wire needs: single vs batch prediction, chunking
to bound memory, and length-sorted batching that preserves input order (BACK-07, NFR-P05).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from tachyone.hooks import ON_ERROR, ON_PREDICT_END, ON_PREDICT_START, Hooks
from tachyone.primitives import Answer, Question, State
from tachyone.router import state_text

#: Encodes a list of states against the same questions, returning one answer map per state.
type Runner = Callable[[list[State], dict[str, Question]], list[dict[str, Answer]]]


class Agent:
    """Runs single-pass inference with bounded, optionally length-sorted batching."""

    def __init__(
        self,
        runner: Runner,
        *,
        max_batch_size: int | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._hooks = hooks

    def _emit(self, event: str, **fields: object) -> None:
        if self._hooks is not None:
            self._hooks.emit(event, **fields)

    def predict(self, state: State, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer one state (a single forward pass)."""
        self._emit(ON_PREDICT_START, state=state, questions=questions)
        start = time.perf_counter()
        try:
            answers = self._runner([state], questions)[0]
        except Exception as exc:
            self._emit(ON_ERROR, state=state, questions=questions, error=exc)
            raise
        self._emit(
            ON_PREDICT_END,
            state=state,
            questions=questions,
            answers=[answers],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        return answers

    def predict_batch(
        self,
        states: Sequence[State],
        questions: dict[str, Question],
        *,
        batch_size: int | None = None,
        sort_by_length: bool = True,
    ) -> list[dict[str, Answer]]:
        """Answer many states, returning results aligned to the input order."""
        items = list(states)
        if not items:
            return []
        size = batch_size or self._max_batch_size or len(items)
        size = max(1, size)

        order = list(range(len(items)))
        if sort_by_length and 1 < size < len(items):
            order.sort(key=lambda index: len(state_text(items[index])))

        self._emit(ON_PREDICT_START, states=items, questions=questions)
        start = time.perf_counter()
        results: list[dict[str, Answer] | None] = [None] * len(items)
        try:
            for chunk_start in range(0, len(order), size):
                chunk = order[chunk_start : chunk_start + size]
                chunk_results = self._runner([items[index] for index in chunk], questions)
                if len(chunk_results) != len(chunk):
                    raise ValueError("runner returned a different number of results than states")
                for position, index in enumerate(chunk):
                    results[index] = chunk_results[position]
        except Exception as exc:
            self._emit(ON_ERROR, states=items, questions=questions, error=exc)
            raise
        ordered = [result for result in results if result is not None]
        self._emit(
            ON_PREDICT_END,
            states=items,
            questions=questions,
            answers=ordered,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
        return ordered

    #: Alias mirroring the extension endpoint name.
    predict_many = predict_batch


__all__ = ["Agent", "Runner"]
