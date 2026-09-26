"""Golden contract tests: the ``/v1/systemone`` wire is frozen here.

This suite encodes ``docs/protocol.md``. Any change to a public field, type, nesting, or
error status MUST update this file in the same commit (ADR-0001, NFR-M03). Fixtures in
``tests/fixtures/`` capture the documented Jev shapes; parity is compared structurally with
a float tolerance (JSON number formatting is not significant).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from tachyone.primitives import Answer
from tachyone.wire import SystemOneRequest, Usage, answer, parse_request
from tests.fakes import FakeBackend, ScriptedBackend

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).parent / "fixtures"
_ANSWER = TypeAdapter(Answer)
_FLOAT_TOLERANCE = 1e-6


def _load(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _parity(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert abs(actual - expected) <= _FLOAT_TOLERANCE, f"{path}: {actual} != {expected}"
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            _parity(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _parity(a, e, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_request_fixture_parses() -> None:
    request = parse_request(_load("systemone_request_multi.json"))
    assert isinstance(request, SystemOneRequest)
    assert set(request.questions) == {"is_urgent", "department", "frustration"}


@pytest.mark.asyncio
async def test_documented_shape_roundtrip_is_byte_stable() -> None:
    """A scripted backend reproducing the documented answers must serialize identically."""
    request = parse_request(_load("systemone_request_multi.json"))
    expected = _load("systemone_response_multi.json")
    answers = {
        question_id: _ANSWER.validate_python(payload)
        for question_id, payload in expected["answers"].items()
    }
    backend = ScriptedBackend(answers, Usage.model_validate(expected["usage"]))

    response = await answer(request, backend)
    _parity(response.model_dump(mode="json"), expected)


@pytest.mark.asyncio
async def test_multi_question_ids_are_echoed() -> None:
    request = parse_request(_load("systemone_request_multi.json"))
    response = await answer(request, FakeBackend())
    assert set(response.answers) == set(request.questions)
    assert response.model == request.model


@pytest.mark.asyncio
async def test_response_top_level_shape_is_frozen() -> None:
    request = parse_request(_load("systemone_request_multi.json"))
    dumped = (await answer(request, FakeBackend())).model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}
    assert set(dumped["usage"]) == {"input_tokens", "output_tokens"}


@pytest.mark.asyncio
async def test_every_answer_matches_its_question_type() -> None:
    request = parse_request(_load("systemone_request_multi.json"))
    response = await answer(request, FakeBackend())
    for question_id, question in request.questions.items():
        assert response.answers[question_id].type == question.type


@pytest.mark.asyncio
async def test_noul_answer_carries_no_confidence() -> None:
    request = parse_request(_load("systemone_request_multi.json"))
    response = await answer(request, FakeBackend())
    assert "confidence" not in response.answers["is_urgent"].model_dump(mode="json")
