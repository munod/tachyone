"""Unit tests for agent batching (M3-T4, BACK-07 / NFR-P05)."""

from __future__ import annotations

from jeba.agent import Agent
from jeba.primitives import Answer, NoulAnswer, NoulQuestion, Question, State


class _Recorder:
    """Fake runner: records the states of each forward pass, answers one noul per state."""

    def __init__(self) -> None:
        self.batches: list[list[State]] = []

    def __call__(
        self, states: list[State], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        self.batches.append(list(states))
        return [{"q": NoulAnswer(noul=min(1.0, len(state) / 10))} for state in states]


_QUESTIONS: dict[str, Question] = {"q": NoulQuestion(instructions="Is it long?")}


def test_predict_runs_one_forward_pass() -> None:
    runner = _Recorder()
    result = Agent(runner).predict({"body": "hi"}, _QUESTIONS)
    assert runner.batches == [[{"body": "hi"}]]
    assert result["q"].noul == 0.1  # type: ignore[union-attr]


def test_predict_batch_aligns_results_to_input_order() -> None:
    runner = _Recorder()
    states: list[State] = ["aaaa", "a", "aa"]
    results = Agent(runner).predict_batch(states, _QUESTIONS, sort_by_length=True)
    assert [r["q"].noul for r in results] == [0.4, 0.1, 0.2]  # type: ignore[union-attr]


def test_predict_batch_chunks_by_batch_size() -> None:
    runner = _Recorder()
    results = Agent(runner).predict_batch(
        ["a", "b", "c", "d", "e"], _QUESTIONS, batch_size=2, sort_by_length=False
    )
    assert [len(batch) for batch in runner.batches] == [2, 2, 1]
    assert len(results) == 5


def test_sort_by_length_reorders_runner_input_but_not_results() -> None:
    runner = _Recorder()
    results = Agent(runner).predict_batch(["cccc", "a", "bb"], _QUESTIONS, batch_size=2)
    # sorted by length across batches: a(1), bb(2), cccc(4)
    assert runner.batches == [["a", "bb"], ["cccc"]]
    assert [r["q"].noul for r in results] == [0.4, 0.1, 0.2]  # type: ignore[union-attr]


def test_empty_states_short_circuits() -> None:
    runner = _Recorder()
    assert Agent(runner).predict_batch([], _QUESTIONS) == []
    assert runner.batches == []


def test_predict_many_is_an_alias() -> None:
    runner = _Recorder()
    assert len(Agent(runner).predict_many(["a", "b"], _QUESTIONS)) == 2
