"""The frozen ``/v1/systemone`` envelope: request/response models, error mapping, and
backend dispatch.

This module is the compatibility authority in code (``docs/protocol.md``). It knows the
contract and delegates the actual judgment to a :class:`~jeba.backends.base.Backend`;
it never imports a concrete backend.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jeba.backends.base import Backend, PredictionResult
from jeba.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
)

#: Absolute tolerance for ``sum(probabilities) == 1.0`` (NFR-C02).
PROBABILITY_TOLERANCE = 1e-6


class _Base(BaseModel):
    """Shared config: unknown fields are ignored for forward compatibility."""

    model_config = ConfigDict(extra="ignore")


class Usage(_Base):
    """Token accounting reported on every response."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneRequest(_Base):
    """A Jev-compatible ``/v1/systemone`` request body."""

    state: State
    model: str
    questions: dict[str, Question]


class SystemOneResponse(_Base):
    """A Jev-compatible ``/v1/systemone`` response body."""

    model: str
    answers: dict[str, Answer]
    usage: Usage


class WireError(Exception):
    """Base error carrying an HTTP status and a JSON error body."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details

    def to_body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.details is not None:
            error["details"] = self.details
        return {"error": error}


class Unauthorized(WireError):
    """401 — missing or invalid API key."""

    status_code: ClassVar[int] = 401
    code: ClassVar[str] = "unauthorized"


class UnprocessableEntity(WireError):
    """422 — request validation failed."""

    status_code: ClassVar[int] = 422
    code: ClassVar[str] = "unprocessable_entity"


class RateLimited(WireError):
    """429 — client exceeded its rate limit."""

    status_code: ClassVar[int] = 429
    code: ClassVar[str] = "rate_limited"


class Overloaded(WireError):
    """529 — server is temporarily overloaded."""

    status_code: ClassVar[int] = 529
    code: ClassVar[str] = "overloaded"


class BackendError(WireError):
    """500 — the backend failed or returned an inconsistent response."""

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"


def _validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]


def parse_request(payload: object) -> SystemOneRequest:
    """Validate a raw request payload, mapping pydantic errors to a 422 wire error."""
    try:
        return SystemOneRequest.model_validate(payload)
    except ValidationError as exc:
        raise UnprocessableEntity(
            "request body failed validation", details=_validation_details(exc)
        ) from exc


def _check_normalized(probabilities: dict[Any, float], question_id: str) -> None:
    total = sum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise BackendError(
            f"probabilities for question {question_id!r} sum to {total}, expected 1.0"
        )


def _check_answer(question: Question, answer: Answer, question_id: str) -> None:
    if question.type != answer.type:
        raise BackendError(
            f"answer type {answer.type!r} does not match question {question_id!r} "
            f"of type {question.type!r}"
        )
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        expected = set(question.criteria)
        if set(answer.probabilities) != expected:
            raise BackendError(
                f"choice probabilities for {question_id!r} must match the declared options"
            )
        _check_normalized(answer.probabilities, question_id)
    elif isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        expected = set(range(len(question.criteria)))
        if set(answer.legend) != expected or set(answer.probabilities) != expected:
            raise BackendError(
                f"score legend/probabilities for {question_id!r} must match the declared levels"
            )
        _check_normalized(answer.probabilities, question_id)


def _check_answers(questions: dict[str, Question], answers: dict[str, Answer]) -> None:
    if set(answers) != set(questions):
        raise BackendError("backend returned answers for a different set of question ids")
    for question_id, question in questions.items():
        _check_answer(question, answers[question_id], question_id)


async def answer(request: SystemOneRequest, backend: Backend) -> SystemOneResponse:
    """Dispatch ``request`` to ``backend`` and return a validated wire response."""
    result: PredictionResult = await backend.predict(
        request.questions, state=request.state, model=request.model
    )
    _check_answers(request.questions, result.answers)
    return SystemOneResponse(
        model=request.model,
        answers=result.answers,
        usage=result.usage,
    )


__all__ = [
    "PROBABILITY_TOLERANCE",
    "BackendError",
    "Overloaded",
    "RateLimited",
    "SystemOneRequest",
    "SystemOneResponse",
    "Unauthorized",
    "UnprocessableEntity",
    "Usage",
    "WireError",
    "answer",
    "parse_request",
]
