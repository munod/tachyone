"""Unit tests for the backend seam and its test doubles (M1-T5, BACK-01)."""

from __future__ import annotations

import dataclasses

import pytest

from tachyone.backends.base import Backend, PredictionResult
from tachyone.primitives import ChoiceAnswer, NoulAnswer, ScoreAnswer
from tachyone.wire import Usage, parse_request
from tests.fakes import FakeBackend, ScriptedBackend

_REQUEST = {
    "state": "please refund the duplicate charge",
    "model": "tachyone-latest",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does this request a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "refunds", "technical": "bugs"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "medium", "high"],
        },
    },
}


def test_prediction_result_is_frozen() -> None:
    result = PredictionResult(answers={}, usage=Usage(input_tokens=0, output_tokens=0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answers = {}  # type: ignore[misc]


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(), Backend)
    assert isinstance(ScriptedBackend({}, Usage(input_tokens=0, output_tokens=0)), Backend)


@pytest.mark.asyncio
async def test_fake_backend_is_deterministic_and_keyed() -> None:
    request = parse_request(_REQUEST)
    backend = FakeBackend()
    first = await backend.predict(request.questions, state=request.state, model=request.model)
    second = await backend.predict(request.questions, state=request.state, model=request.model)
    assert first == second
    assert set(first.answers) == set(request.questions)


@pytest.mark.asyncio
async def test_fake_backend_answer_types_and_normalization() -> None:
    request = parse_request(_REQUEST)
    result = await FakeBackend().predict(
        request.questions, state=request.state, model=request.model
    )
    assert isinstance(result.answers["refund"], NoulAnswer)
    team = result.answers["team"]
    assert isinstance(team, ChoiceAnswer)
    assert abs(sum(team.probabilities.values()) - 1.0) < 1e-9
    urgency = result.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert abs(sum(urgency.probabilities.values()) - 1.0) < 1e-9
    assert urgency.legend == {0: "low", 1: "medium", 2: "high"}


@pytest.mark.asyncio
async def test_scripted_backend_returns_only_requested_ids() -> None:
    request = parse_request(_REQUEST)
    scripted = {
        "refund": NoulAnswer(noul=0.9),
        "team": ChoiceAnswer(
            choice="billing", probabilities={"billing": 0.9, "technical": 0.1}, confidence=0.9
        ),
        "urgency": ScoreAnswer(
            score=2.0,
            legend={0: "low", 1: "medium", 2: "high"},
            probabilities={0: 0.0, 1: 0.1, 2: 0.9},
            confidence=0.9,
        ),
    }
    backend = ScriptedBackend(scripted, Usage(input_tokens=1, output_tokens=1))
    result = await backend.predict(request.questions, state=request.state, model=request.model)
    assert result.answers["refund"] == scripted["refund"]
    assert result.usage == Usage(input_tokens=1, output_tokens=1)
