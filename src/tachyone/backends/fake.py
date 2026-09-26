"""Deterministic, model-free backends.

``FakeBackend`` derives a stable answer from any question, so the whole wire cycle works
offline without an API key. It backs ``--backend fake`` demos, CI end-to-end tests, and the
placeholder until the local encoder lands in M3. ``ScriptedBackend`` replays fixed answers,
which is how the golden fixtures pin the documented response shape.
"""

from __future__ import annotations

import json

from tachyone.backends.base import PredictionResult
from tachyone.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
)
from tachyone.wire import Usage

#: Probability mass placed on the selected option/level by :class:`FakeBackend`.
_SELECTED_MASS = 0.5


def _distribution[K](keys: list[K], selected: K) -> dict[K, float]:
    if len(keys) == 1:
        return {keys[0]: 1.0}
    rest = (1.0 - _SELECTED_MASS) / (len(keys) - 1)
    return {key: (_SELECTED_MASS if key == selected else rest) for key in keys}


def answer_question(question: Question) -> Answer:
    """Return a deterministic answer for any question (no model involved)."""
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=0.5)
    if isinstance(question, ChoiceQuestion):
        options = list(question.criteria)
        selected = options[0]
        probabilities = _distribution(options, selected)
        return ChoiceAnswer(
            choice=selected,
            probabilities=probabilities,
            confidence=max(probabilities.values()),
        )
    if isinstance(question, ScoreQuestion):
        indices = list(range(len(question.criteria)))
        selected = indices[len(indices) // 2]
        probabilities = _distribution(indices, selected)
        expected = sum(index * probability for index, probability in probabilities.items())
        return ScoreAnswer(
            score=expected,
            legend=dict(enumerate(question.criteria)),
            probabilities=probabilities,
            confidence=max(probabilities.values()),
        )
    raise TypeError(f"unsupported question type: {type(question)!r}")


def estimate_usage(state: State, questions: dict[str, Question]) -> Usage:
    """A deterministic, model-free token estimate."""
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False)
    return Usage(
        input_tokens=max(1, len(payload) // 4),
        output_tokens=max(1, len(questions)),
    )


class FakeBackend:
    """Answers every question deterministically; useful for offline tests and demos."""

    name = "fake"

    def __init__(self, usage: Usage | None = None) -> None:
        self._usage = usage

    async def predict(
        self,
        questions: dict[str, Question],
        *,
        state: State,
        model: str,
        return_details: bool = False,
    ) -> PredictionResult:
        answers = {question_id: answer_question(q) for question_id, q in questions.items()}
        usage = self._usage if self._usage is not None else estimate_usage(state, questions)
        return PredictionResult(answers=answers, usage=usage)


class ScriptedBackend:
    """Replays a fixed, pre-validated set of answers for the requested question ids."""

    name = "scripted"

    def __init__(self, answers: dict[str, Answer], usage: Usage) -> None:
        self._answers = answers
        self._usage = usage

    async def predict(
        self,
        questions: dict[str, Question],
        *,
        state: State,
        model: str,
        return_details: bool = False,
    ) -> PredictionResult:
        return PredictionResult(
            answers={question_id: self._answers[question_id] for question_id in questions},
            usage=self._usage,
        )


__all__ = [
    "FakeBackend",
    "ScriptedBackend",
    "answer_question",
    "estimate_usage",
]
