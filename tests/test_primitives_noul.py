"""Unit tests for the ``noul`` primitive (M1-T1, PRIM-01)."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from jeba.primitives import NoulAnswer, NoulCriteria, NoulQuestion, Question

_QUESTION = TypeAdapter(Question)


def test_noul_question_minimal() -> None:
    q = NoulQuestion(instructions="Is this text a greeting?")
    assert q.type == "noul"
    assert q.criteria is None
    assert q.model_dump(mode="json") == {
        "type": "noul",
        "instructions": "Is this text a greeting?",
        "criteria": None,
    }


def test_noul_question_with_criteria() -> None:
    q = NoulQuestion(
        instructions="Is the message polite?",
        criteria=NoulCriteria.model_validate({"true": "polite", "false": "impolite"}),
    )
    assert q.criteria is not None
    assert q.criteria.true == "polite"
    assert q.criteria.false == "impolite"


def test_noul_question_structured_instructions_and_criteria() -> None:
    q = NoulQuestion(
        instructions={"text": "hi", "question": "Is `text` a greeting?"},
        criteria=NoulCriteria.model_validate(
            {"true": {"label": "greeting"}, "false": ["not", "a", "greeting"]}
        ),
    )
    dumped = q.model_dump(mode="json")
    assert dumped["instructions"]["text"] == "hi"
    assert dumped["criteria"]["true"] == {"label": "greeting"}


def test_noul_criteria_branches_are_optional() -> None:
    assert NoulCriteria().true is None
    assert NoulCriteria().false is None


def test_noul_answer_shape_and_serialization() -> None:
    answer = NoulAnswer(noul=0.97)
    assert answer.model_dump(mode="json") == {"type": "noul", "noul": 0.97}
    assert "confidence" not in answer.model_dump(mode="json")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2])
def test_noul_answer_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValidationError):
        NoulAnswer(noul=bad)


def test_noul_question_requires_instructions() -> None:
    with pytest.raises(ValidationError):
        NoulQuestion.model_validate({"type": "noul"})


def test_question_union_discriminates_noul() -> None:
    q = _QUESTION.validate_python({"type": "noul", "instructions": "Is it true?"})
    assert isinstance(q, NoulQuestion)


def test_question_union_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        _QUESTION.validate_python({"type": "unknown", "instructions": "x"})
