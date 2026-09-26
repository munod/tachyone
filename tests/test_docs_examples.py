"""The JSON examples embedded in ``docs/use-cases.md`` must stay valid wire traffic.

Docs examples are public surface: a reader copies them into a client. Every pair under
``docs/assets/examples/`` is therefore checked against the invariants the wire itself enforces —
question/answer key parity, probabilities over exactly the declared options, ``confidence``
equal to the selected mass, and ``noul`` carrying no ``confidence`` (ADR-0001).

These tests intentionally do **not** touch the frozen golden suite in
``test_contract_wire.py``: they validate committed artifacts, not the protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tachyone.primitives import ChoiceAnswer, NoulAnswer, ScoreAnswer
from tachyone.wire import SystemOneResponse, answer, parse_request
from tests.fakes import FakeBackend

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "docs" / "assets" / "examples"
_DOC = _ROOT / "docs" / "use-cases.md"

#: Mirrors ``wire.PROBABILITY_TOLERANCE``: quantized docs values stay exact here.
_SUM_TOLERANCE = 1e-9


def _slugs(kind: str) -> set[str]:
    suffix = f".{kind}.json"
    return {path.name[: -len(suffix)] for path in _EXAMPLES.glob(f"*{suffix}")}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_examples_exist_and_every_request_has_a_response() -> None:
    requests, responses = _slugs("request"), _slugs("response")
    assert requests, "no docs examples found under docs/assets/examples/"
    assert requests == responses, (
        f"request/response pairs differ: only-request={requests - responses} "
        f"only-response={responses - requests}"
    )


def test_every_example_is_embedded_in_the_use_cases_doc() -> None:
    """A committed example nobody renders is an example that will rot."""
    doc = _DOC.read_text(encoding="utf-8")
    for slug in sorted(_slugs("request")):
        assert f"{slug}.request.json" in doc, f"{slug}.request.json is not embedded in use-cases.md"
        assert f"{slug}.response.json" in doc, (
            f"{slug}.response.json is not embedded in use-cases.md"
        )


def test_requests_parse_as_systemone_requests() -> None:
    for slug in sorted(_slugs("request")):
        request = parse_request(_load(_EXAMPLES / f"{slug}.request.json"))
        assert request.questions, f"{slug}: empty question set"


def test_responses_keep_the_wire_invariants() -> None:
    for slug in sorted(_slugs("request")):
        request = parse_request(_load(_EXAMPLES / f"{slug}.request.json"))
        response = SystemOneResponse.model_validate(_load(_EXAMPLES / f"{slug}.response.json"))

        assert set(response.answers) == set(request.questions), f"{slug}: answer ids differ"
        assert response.model == request.model, f"{slug}: model id is not echoed"

        for question_id, question in request.questions.items():
            payload = response.answers[question_id].model_dump(mode="json")
            assert payload["type"] == question.type, f"{slug}:{question_id}: type mismatch"

            if question.type == "noul":
                assert isinstance(response.answers[question_id], NoulAnswer)
                assert "confidence" not in payload, f"{slug}:{question_id}: noul has no confidence"
                assert 0.0 <= payload["noul"] <= 1.0
                continue

            expected = (
                set(question.criteria)
                if isinstance(response.answers[question_id], ChoiceAnswer)
                else {str(index) for index in range(len(question.criteria))}
            )
            selected = response.answers[question_id]
            assert isinstance(selected, (ChoiceAnswer, ScoreAnswer))
            assert set(payload["probabilities"]) == expected, (
                f"{slug}:{question_id}: probability keys must be the declared options"
            )
            total = sum(payload["probabilities"].values())
            assert abs(total - 1.0) <= _SUM_TOLERANCE, f"{slug}:{question_id}: sum={total}"
            assert max(payload["probabilities"].values()) == pytest.approx(payload["confidence"]), (
                f"{slug}:{question_id}: confidence must be the selected mass"
            )


@pytest.mark.asyncio
async def test_requests_answer_through_the_wire() -> None:
    """The committed bodies are answerable, not just schema-valid."""
    for slug in sorted(_slugs("request")):
        request = parse_request(_load(_EXAMPLES / f"{slug}.request.json"))
        response = await answer(request, FakeBackend())
        assert set(response.answers) == set(request.questions), f"{slug}: ids not echoed"
        assert response.model == request.model, f"{slug}: model not echoed"


def test_langgraph_flow_is_embedded_in_the_doc() -> None:
    doc = (_ROOT / "docs" / "integrations" / "langgraph.md").read_text(encoding="utf-8")
    assert "langgraph_flow.py" in doc, "the executed flow must be the one the docs embed"


def test_langgraph_example_runs() -> None:
    """The documented LangGraph script executes end to end, branch and all.

    With the model-free backend the distribution is flat, so the confidence lands below τ and
    the graph must take the *escalate* edge — the branch cannot silently rot.
    """
    pytest.importorskip("langchain_core")
    pytest.importorskip("langgraph")

    script = _EXAMPLES / "langgraph_flow.py"
    assert script.exists()
    env = {**os.environ, "TACHYONE_BACKEND": "fake"}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "handoff -> human-review" in result.stdout, result.stdout
