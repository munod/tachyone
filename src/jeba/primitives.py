"""Canonical pydantic v2 models for the System One primitives and their answers.

Field names and types mirror the TypeSafe Jev ``/v1/systemone`` contract
(``docs/protocol.md``). This module is pure data + validation: no I/O, no model
knowledge, no optional dependencies (NFR-R01).

Answer invariants that need the originating question (for example, probability keys
matching the declared options) are enforced in :mod:`jeba.wire` where both sides are
available.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: JSON-shaped value accepted for ``state``, ``instructions`` and structured criteria.
type JsonValue = str | dict[str, Any] | list[Any]

#: The subject of a request: plain text or structured JSON.
type State = JsonValue

#: Maximum number of options in a ``choice`` question (Jev limit).
MAX_CHOICE_OPTIONS = 255

#: Inclusive bounds on the number of levels in a ``score`` question (Jev limits).
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

#: Probability value, constrained to the closed unit interval.
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Base(BaseModel):
    """Shared config: unknown fields are ignored for forward compatibility."""

    model_config = ConfigDict(extra="ignore")


class NoulCriteria(_Base):
    """Optional clarification of what a yes/no answer means."""

    true: JsonValue | None = None
    false: JsonValue | None = None


class NoulQuestion(_Base):
    """Binary yes/no judgment."""

    type: Literal["noul"] = "noul"
    instructions: JsonValue
    criteria: NoulCriteria | None = None


class NoulAnswer(_Base):
    """Answer to a :class:`NoulQuestion`; carries no separate ``confidence``."""

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


#: Discriminated union of every question type, keyed on ``type``.
Question = Annotated[NoulQuestion, Field(discriminator="type")]

#: Discriminated union of every answer type, keyed on ``type``.
Answer = Annotated[NoulAnswer, Field(discriminator="type")]

__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Answer",
    "JsonValue",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Probability",
    "Question",
    "State",
]
