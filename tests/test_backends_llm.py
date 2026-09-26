"""Unit tests for the LLM backend (M2-T2, BACK-02 / BACK-05)."""

from __future__ import annotations

from typing import Any

import pytest

from tachyone.backends.llm import (
    LLMBackend,
    build_answers,
    build_messages,
    extract_json_object,
)
from tachyone.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from tachyone.wire import BackendError, Overloaded, RateLimited

_QUESTIONS = {
    "urgent": NoulQuestion(instructions="Is it urgent?"),
    "team": ChoiceQuestion(
        instructions="Which team?", criteria={"billing": "refunds", "technical": "bugs"}
    ),
    "frustration": ScoreQuestion(
        instructions="How frustrated?", criteria=["calm", "upset", "angry"]
    ),
}


class _Queue:
    """A transport that replays queued responses (or raises queued exceptions)."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def __call__(self, payload: dict[str, Any]) -> str:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return str(item)


def _backend(transport: _Queue) -> LLMBackend:
    return LLMBackend(
        base_url="http://localhost/v1", api_key=None, model="test-model", transport=transport
    )


def test_build_messages_includes_state_and_questions() -> None:
    messages = build_messages({"body": "hello"}, _QUESTIONS)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "urgent" in messages[1]["content"]
    assert "billing" in messages[1]["content"]


@pytest.mark.parametrize(
    "text",
    [
        '{"answers": {}}',
        'Here you go: {"answers": {}} — done',
        '```json\n{"answers": {}}\n```',
    ],
)
def test_extract_json_object_tolerates_prose(text: str) -> None:
    assert extract_json_object(text) == {"answers": {}}


def test_extract_json_object_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


def test_build_answers_normalizes_and_fills_legend() -> None:
    payload = {
        "answers": {
            "urgent": {"noul": 0.8},
            "team": {"choice": "billing", "probabilities": {"billing": 0.4, "technical": 0.1}},
            "frustration": {
                "score": 1.4,
                "probabilities": {"0": 1, "1": 6, "2": 3},
            },
        }
    }
    answers = build_answers(_QUESTIONS, payload)
    assert answers["urgent"].noul == 0.8  # type: ignore[union-attr]
    team = answers["team"]
    assert isinstance(team, ChoiceAnswer)
    assert abs(sum(team.probabilities.values()) - 1.0) < 1e-9
    assert team.choice == "billing"
    frustration = answers["frustration"]
    assert isinstance(frustration, ScoreAnswer)
    assert frustration.legend == {0: "calm", 1: "upset", 2: "angry"}


def test_build_answers_rejects_missing_question() -> None:
    with pytest.raises(ValueError):
        build_answers(_QUESTIONS, {"answers": {"urgent": {"noul": 0.5}}})


@pytest.mark.asyncio
async def test_predict_success() -> None:
    transport = _Queue(['{"answers": {"urgent": {"noul": 0.7}}}'])
    backend = _backend(transport)
    result = await backend.predict({"urgent": _QUESTIONS["urgent"]}, state="x", model="m")
    assert result.answers["urgent"].noul == 0.7  # type: ignore[union-attr]
    assert result.usage.input_tokens >= 1


@pytest.mark.asyncio
async def test_predict_retries_on_invalid_output() -> None:
    transport = _Queue(["not json", '{"answers": {"urgent": {"noul": 0.9}}}'])
    backend = _backend(transport)
    result = await backend.predict({"urgent": _QUESTIONS["urgent"]}, state="x", model="m")
    assert result.answers["urgent"].noul == 0.9  # type: ignore[union-attr]
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_predict_raises_after_exhausting_retries() -> None:
    transport = _Queue(["nope", "still nope", "nope again"])
    backend = LLMBackend(base_url="", api_key=None, model="m", transport=transport, max_retries=2)
    with pytest.raises(BackendError):
        await backend.predict({"urgent": _QUESTIONS["urgent"]}, state="x", model="m")
    assert transport.calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RateLimited("slow"), Overloaded("busy")])
async def test_transport_errors_propagate(error: Exception) -> None:
    with pytest.raises(type(error)):
        await _backend(_Queue([error])).predict(
            {"urgent": _QUESTIONS["urgent"]}, state="x", model="m"
        )


@pytest.mark.asyncio
async def test_empty_questions_short_circuits() -> None:
    transport = _Queue([])
    result = await _backend(transport).predict({}, state="x", model="m")
    assert result.answers == {}
    assert transport.calls == 0
