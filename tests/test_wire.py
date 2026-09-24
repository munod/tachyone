"""Unit tests for the ``/v1/systemone`` envelope (M1-T4, WIRE-02 / WIRE-03 / WIRE-07)."""

from __future__ import annotations

from typing import Any

import pytest

from jeba.backends.base import PredictionResult
from jeba.primitives import Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, State
from jeba.wire import (
    BackendError,
    SystemOneRequest,
    UnprocessableEntity,
    Usage,
    answer,
    parse_request,
)

_REQUEST: dict[str, Any] = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jeba-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs"},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated?",
            "criteria": ["calm", "concerned", "angry"],
        },
    },
}


def _answers() -> dict[str, Answer]:
    return {
        "is_urgent": NoulAnswer(noul=0.93),
        "department": ChoiceAnswer(
            choice="technical",
            probabilities={"billing": 0.4, "technical": 0.6},
            confidence=0.6,
        ),
        "frustration": ScoreAnswer(
            score=1.0,
            legend={0: "calm", 1: "concerned", 2: "angry"},
            probabilities={0: 0.2, 1: 0.3, 2: 0.5},
            confidence=0.5,
        ),
    }


class _StubBackend:
    def __init__(self, answers: dict[str, Answer] | None = None) -> None:
        self._answers = answers if answers is not None else _answers()

    async def predict(
        self,
        questions: dict[str, Question],
        *,
        state: State,
        model: str,
        return_details: bool = False,
    ) -> PredictionResult:
        return PredictionResult(
            answers=dict(self._answers),
            usage=Usage(input_tokens=42, output_tokens=7),
        )


def test_parse_request_accepts_valid_body() -> None:
    request = parse_request(_REQUEST)
    assert isinstance(request, SystemOneRequest)
    assert request.model == "jeba-latest"
    assert set(request.questions) == {"is_urgent", "department", "frustration"}


def test_parse_request_accepts_empty_questions() -> None:
    request = parse_request({"state": "x", "model": "m", "questions": {}})
    assert request.questions == {}


@pytest.mark.asyncio
async def test_answer_roundtrip_keys_and_usage() -> None:
    request = parse_request(_REQUEST)
    response = await answer(request, _StubBackend())
    assert response.model == "jeba-latest"
    assert set(response.answers) == set(request.questions)
    assert response.usage == Usage(input_tokens=42, output_tokens=7)
    assert response.answers["is_urgent"].type == "noul"


@pytest.mark.asyncio
async def test_response_serialization_shape() -> None:
    response = await answer(parse_request(_REQUEST), _StubBackend())
    dumped = response.model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}
    assert set(dumped["usage"]) == {"input_tokens", "output_tokens"}
    assert dumped["answers"]["department"]["type"] == "choice"
    assert dumped["answers"]["frustration"]["legend"] == {
        "0": "calm",
        "1": "concerned",
        "2": "angry",
    }


@pytest.mark.asyncio
async def test_answer_rejects_question_id_mismatch() -> None:
    request = parse_request(_REQUEST)
    bad = _answers()
    bad["extra"] = NoulAnswer(noul=0.5)
    with pytest.raises(BackendError):
        await answer(request, _StubBackend(bad))


@pytest.mark.asyncio
async def test_answer_rejects_probability_key_mismatch() -> None:
    request = parse_request(_REQUEST)
    bad = _answers()
    bad["department"] = ChoiceAnswer(
        choice="billing", probabilities={"billing": 1.0}, confidence=1.0
    )
    with pytest.raises(BackendError):
        await answer(request, _StubBackend(bad))


@pytest.mark.asyncio
async def test_answer_rejects_unnormalized_probabilities() -> None:
    request = parse_request(_REQUEST)
    bad = _answers()
    bad["frustration"] = ScoreAnswer(
        score=1.0,
        legend={0: "calm", 1: "concerned", 2: "angry"},
        probabilities={0: 0.1, 1: 0.1, 2: 0.1},
        confidence=0.5,
    )
    with pytest.raises(BackendError):
        await answer(request, _StubBackend(bad))


def test_parse_request_invalid_raises_422() -> None:
    with pytest.raises(UnprocessableEntity) as excinfo:
        parse_request({"model": "m", "questions": {}})
    assert excinfo.value.status_code == 422
    assert excinfo.value.to_body()["error"]["code"] == "unprocessable_entity"
