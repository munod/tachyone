"""Turn JSON Schema or pydantic models into decision primitives.

A JSON property maps to a primitive by shape:

- ``enum`` -> ``choice`` (one option per value)
- ``boolean`` -> ``noul``
- ``integer``/``number`` with bounds spanning 2..10 levels -> ``score``

Anything else (free-form strings, arrays, nested objects, unbounded numbers) cannot be a
clean primitives question and raises ``ValueError`` so the failure is explicit (SERVE-08).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from jeba.primitives import (
    MAX_CHOICE_OPTIONS,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)


def _instructions(name: str, prop: Mapping[str, Any]) -> str:
    return str(prop.get("description") or prop.get("title") or f"How should `{name}` be answered?")


def _property_to_question(name: str, prop: Mapping[str, Any]) -> Question:
    if "enum" in prop:
        options = prop["enum"]
        if not isinstance(options, list) or not 1 <= len(options) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"property {name!r} has an invalid enum")
        return ChoiceQuestion(
            instructions=_instructions(name, prop),
            criteria={str(option): None for option in options},
        )
    ptype = prop.get("type")
    if ptype == "boolean":
        return NoulQuestion(instructions=_instructions(name, prop))
    if ptype in {"integer", "number"}:
        low = prop.get("minimum")
        high = prop.get("maximum")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low < high:
            count = int(high - low) + 1
            if 2 <= count <= 10:
                return ScoreQuestion(
                    instructions=_instructions(name, prop),
                    criteria=[str(level) for level in range(int(low), int(high) + 1)],
                )
        raise ValueError(f"numeric property {name!r} needs integer bounds spanning 2..10 levels")
    raise ValueError(f"property {name!r} has an unsupported schema: {dict(prop)!r}")


def schema_to_questions(schema: Mapping[str, Any] | type[BaseModel]) -> dict[str, Question]:
    """Convert a JSON Schema object (or pydantic model) into a question map."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("schema must be an object schema (type: object) or a pydantic model")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError("schema must declare at least one property")
    return {name: _property_to_question(name, prop) for name, prop in properties.items()}


def decide(
    schema: Mapping[str, Any] | type[BaseModel], *, return_details: bool = False
) -> dict[str, Question]:
    """Return the decision primitives that answer ``schema``.

    ``return_details`` is accepted for API stability; it is honored by the prediction path
    once calibrated distributions are exposed (CAL-05, M3).
    """
    del return_details
    return schema_to_questions(schema)


__all__ = ["decide", "schema_to_questions"]
