"""Unit tests for schema-driven decisions (M2-T7, SERVE-08)."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from tachyone.primitives import ChoiceQuestion, NoulQuestion, ScoreQuestion
from tachyone.schemas import decide, schema_to_questions
from tachyone.wire import SystemOneRequest

_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {
            "type": "string",
            "enum": ["billing", "support", "sales"],
            "description": "Which team should handle this?",
        },
        "needs_reply": {"type": "boolean", "description": "Does it need a reply?"},
        "urgency": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
            "description": "How urgent?",
        },
    },
}


def test_schema_yields_valid_questions() -> None:
    questions = decide(_SCHEMA)
    assert isinstance(questions["department"], ChoiceQuestion)
    assert isinstance(questions["needs_reply"], NoulQuestion)
    assert isinstance(questions["urgency"], ScoreQuestion)
    SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": questions})


def test_enum_becomes_choice_with_options() -> None:
    questions = schema_to_questions(_SCHEMA)
    assert set(questions["department"].criteria) == {"billing", "support", "sales"}  # type: ignore[union-attr]


def test_instructions_from_description() -> None:
    questions = schema_to_questions(_SCHEMA)
    assert questions["department"].instructions == "Which team should handle this?"


def test_pydantic_model_is_supported() -> None:
    class Ticket(BaseModel):
        department: Literal["billing", "support"]
        urgent: bool

    questions = decide(Ticket)
    assert set(questions) == {"department", "urgent"}


def test_unbounded_number_fails_clearly() -> None:
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}
    with pytest.raises(ValueError, match="numeric property"):
        schema_to_questions(schema)


def test_free_form_string_fails_clearly() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    with pytest.raises(ValueError, match="unsupported schema"):
        schema_to_questions(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"type": "object", "properties": {}},
        {"type": "object"},
    ],
)
def test_invalid_schema_raises(schema: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schema_to_questions(schema)
