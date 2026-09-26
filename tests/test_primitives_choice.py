"""Unit tests for the ``choice`` primitive (M1-T2, PRIM-02 / PRIM-04)."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from tachyone.primitives import (
    MAX_CHOICE_OPTIONS,
    ChoiceAnswer,
    ChoiceQuestion,
    JsonValue,
    Question,
)

_QUESTION = TypeAdapter(Question)


def _options(n: int) -> dict[str, JsonValue | None]:
    return {f"opt{i}": (None if i % 2 else f"option {i}") for i in range(n)}


def test_choice_question_valid() -> None:
    q = ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={"billing": "payments", "technical": None, "sales": "pricing"},
    )
    dumped = q.model_dump(mode="json")
    assert dumped["type"] == "choice"
    assert dumped["criteria"]["technical"] is None


def test_choice_accepts_structured_option_descriptions() -> None:
    q = ChoiceQuestion(
        instructions="Pick one.",
        criteria={"news": {"desc": "journalism"}, "blog": ["casual", "posts"]},
    )
    assert q.criteria["news"] == {"desc": "journalism"}
    assert q.criteria["blog"] == ["casual", "posts"]


def test_choice_accepts_exactly_max_options() -> None:
    q = ChoiceQuestion(instructions="x", criteria=_options(MAX_CHOICE_OPTIONS))
    assert len(q.criteria) == MAX_CHOICE_OPTIONS


def test_choice_rejects_more_than_max_options() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="x", criteria=_options(MAX_CHOICE_OPTIONS + 1))


def test_choice_rejects_empty_options() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="x", criteria={})


def test_choice_answer_shape() -> None:
    answer = ChoiceAnswer(
        choice="technical",
        probabilities={"billing": 0.15, "technical": 0.85, "sales": 0.0},
        confidence=0.85,
    )
    dumped = answer.model_dump(mode="json")
    assert dumped["type"] == "choice"
    assert dumped["choice"] == "technical"
    assert set(dumped["probabilities"]) == {"billing", "technical", "sales"}


def test_choice_answer_rejects_selected_option_absent_from_distribution() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="sales", probabilities={"billing": 1.0}, confidence=1.0)


def test_choice_answer_rejects_empty_distribution() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="billing", probabilities={}, confidence=1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_choice_answer_rejects_out_of_range_probability(bad: float) -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="a", probabilities={"a": bad, "b": 0.0}, confidence=0.5)


def test_question_union_discriminates_choice() -> None:
    q = _QUESTION.validate_python(
        {"type": "choice", "instructions": "Pick.", "criteria": {"a": None, "b": None}}
    )
    assert isinstance(q, ChoiceQuestion)
