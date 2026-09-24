"""Unit tests for the ``score`` primitive (M1-T3, PRIM-03 / PRIM-04)."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from jeba.primitives import (
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

_QUESTION = TypeAdapter(Question)


def test_score_question_valid() -> None:
    q = ScoreQuestion(
        instructions="Rate the sentiment.",
        criteria=["very negative", "negative", "neutral", "positive", "very positive"],
    )
    assert q.type == "score"
    assert len(q.criteria) == 5


@pytest.mark.parametrize("n", [MIN_SCORE_LEVELS, MAX_SCORE_LEVELS])
def test_score_accepts_boundary_level_counts(n: int) -> None:
    q = ScoreQuestion(instructions="x", criteria=[f"level {i}" for i in range(n)])
    assert len(q.criteria) == n


@pytest.mark.parametrize("n", [0, 1, MAX_SCORE_LEVELS + 1])
def test_score_rejects_out_of_range_level_counts(n: int) -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="x", criteria=[f"level {i}" for i in range(n)])


def test_score_answer_shape_and_serialization() -> None:
    answer = ScoreAnswer(
        score=1.05,
        legend={0: "Calm", 1: "Frustrated", 2: "Very angry"},
        probabilities={0: 0.0, 1: 0.95, 2: 0.05},
        confidence=0.95,
    )
    dumped = answer.model_dump(mode="json")
    assert dumped["type"] == "score"
    assert dumped["score"] == 1.05
    assert dumped["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert dumped["probabilities"] == {"0": 0.0, "1": 0.95, "2": 0.05}


def test_score_answer_accepts_string_index_keys() -> None:
    answer = ScoreAnswer.model_validate(
        {
            "type": "score",
            "score": 1.0,
            "legend": {"0": "a", "1": "b"},
            "probabilities": {"0": 0.2, "1": 0.8},
            "confidence": 0.8,
        }
    )
    assert answer.legend == {0: "a", 1: "b"}


def test_score_answer_rejects_probability_key_mismatch() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=0.0,
            legend={0: "a", 1: "b"},
            probabilities={0: 1.0},
            confidence=1.0,
        )


def test_score_answer_rejects_non_contiguous_legend() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=1.0,
            legend={0: "a", 2: "c"},
            probabilities={0: 0.5, 2: 0.5},
            confidence=0.5,
        )


def test_score_answer_rejects_score_outside_range() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=2.5,
            legend={0: "a", 1: "b"},
            probabilities={0: 0.5, 1: 0.5},
            confidence=0.5,
        )


def test_question_union_discriminates_score() -> None:
    q = _QUESTION.validate_python(
        {"type": "score", "instructions": "Rate.", "criteria": ["a", "b"]}
    )
    assert isinstance(q, ScoreQuestion)
