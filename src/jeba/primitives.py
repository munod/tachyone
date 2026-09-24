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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class ChoiceQuestion(_Base):
    """Pick one option from a labeled set (1..255 options)."""

    type: Literal["choice"] = "choice"
    instructions: JsonValue
    criteria: dict[str, JsonValue | None]

    @field_validator("criteria")
    @classmethod
    def _validate_option_count(
        cls, value: dict[str, JsonValue | None]
    ) -> dict[str, JsonValue | None]:
        if not value:
            raise ValueError("choice requires at least one option")
        if len(value) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"choice accepts at most {MAX_CHOICE_OPTIONS} options, got {len(value)}"
            )
        return value


class ChoiceAnswer(_Base):
    """Answer to a :class:`ChoiceQuestion`: selected option + distribution."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_selected_option(self) -> ChoiceAnswer:
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")
        if self.choice not in self.probabilities:
            raise ValueError("choice must be one of the probabilities keys")
        return self


class ScoreQuestion(_Base):
    """Rate the state on an ordered scale of 2..10 levels."""

    type: Literal["score"] = "score"
    instructions: JsonValue
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _validate_level_count(cls, value: list[str]) -> list[str]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score requires {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(value)}"
            )
        return value


class ScoreAnswer(_Base):
    """Answer to a :class:`ScoreQuestion`: expected level + distribution.

    ``score`` is the probability-weighted position along the levels and may land
    between two levels (matches Jev, which returns a number, not a level index).
    """

    type: Literal["score"] = "score"
    score: float
    legend: dict[int, str]
    probabilities: dict[int, Probability]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_levels(self) -> ScoreAnswer:
        if not self.legend:
            raise ValueError("legend must not be empty")
        expected = set(range(len(self.legend)))
        if set(self.legend) != expected:
            raise ValueError("legend keys must be contiguous indices starting at 0")
        if set(self.probabilities) != expected:
            raise ValueError("probabilities keys must match the legend keys")
        if not 0.0 <= self.score <= float(len(self.legend) - 1):
            raise ValueError("score must fall within the legend range")
        return self


#: Discriminated union of every question type, keyed on ``type``.
Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]

#: Discriminated union of every answer type, keyed on ``type``.
Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]

__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "JsonValue",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Probability",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "State",
]
