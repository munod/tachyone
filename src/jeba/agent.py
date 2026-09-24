"""Batch orchestration for the local encoder.

An :class:`Agent` owns a ``runner`` — the function that performs one forward pass over a list
of states and returns aligned answers (the encoder backend supplies the torch one; tests
supply a fake). The agent adds the parts the wire needs: single vs batch prediction, chunking
to bound memory, and length-sorted batching that preserves input order (BACK-07, NFR-P05).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jeba.primitives import Answer, Question, State
from jeba.router import state_text

#: Encodes a list of states against the same questions, returning one answer map per state.
type Runner = Callable[[list[State], dict[str, Question]], list[dict[str, Answer]]]


class Agent:
    """Runs single-pass inference with bounded, optionally length-sorted batching."""

    def __init__(self, runner: Runner, *, max_batch_size: int | None = None) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size

    def predict(self, state: State, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer one state (a single forward pass)."""
        return self._runner([state], questions)[0]

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

        results: list[dict[str, Answer] | None] = [None] * len(items)
        for start in range(0, len(order), size):
            chunk = order[start : start + size]
            chunk_results = self._runner([items[index] for index in chunk], questions)
            if len(chunk_results) != len(chunk):
                raise ValueError("runner returned a different number of results than states")
            for position, index in enumerate(chunk):
                results[index] = chunk_results[position]
        return [result for result in results if result is not None]

    #: Alias mirroring the extension endpoint name.
    predict_many = predict_batch


__all__ = ["Agent", "Runner"]
