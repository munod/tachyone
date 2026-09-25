"""Unit tests for confidence thresholding and System-2 handoff (B-3, CAL-03 / CAL-05 / CAL-06)."""

from __future__ import annotations

import pytest

from jeba.handoff import (
    HandoffReport,
    HandoffSignal,
    Uncertainty,
    assess,
    assess_response,
)
from jeba.primitives import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jeba.wire import SystemOneResponse, Usage


def _response(answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]) -> SystemOneResponse:
    return SystemOneResponse(
        model="m", answers=answers, usage=Usage(input_tokens=0, output_tokens=0)
    )


# --- assess -------------------------------------------------------------------------------


def test_assess_abstains_below_threshold() -> None:
    result = assess({"a": 0.4, "b": 0.6}, threshold=0.8)
    assert isinstance(result, Uncertainty)
    assert result.confidence == pytest.approx(0.6)
    assert result.abstain is True
    assert result.threshold == 0.8


def test_assess_does_not_abstain_at_or_above_threshold() -> None:
    result = assess({"a": 0.8, "b": 0.2}, threshold=0.8)
    assert result.abstain is False


def test_assess_strict_boundary_is_not_abstain() -> None:
    # abstain is `confidence < threshold`, so equality passes.
    assert assess({"a": 0.5, "b": 0.5}, threshold=0.5).abstain is False


def test_assess_reports_entropy_and_margin() -> None:
    result = assess({"a": 0.9, "b": 0.1}, threshold=0.5)
    assert result.entropy == pytest.approx(0.469, abs=1e-3)
    assert result.margin == pytest.approx(0.8)


def test_assess_empty_abstains() -> None:
    result = assess({}, threshold=0.1)
    assert result.confidence == 0.0
    assert result.abstain is True


def test_assess_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError):
        assess({"a": 1.0}, threshold=1.5)
    with pytest.raises(ValueError):
        assess({"a": 1.0}, threshold=-0.1)


# --- assess_response ----------------------------------------------------------------------


def test_assess_response_choice_uses_confidence() -> None:
    response = _response(
        {
            "dept": ChoiceAnswer(
                choice="billing", probabilities={"billing": 0.5, "tech": 0.5}, confidence=0.5
            )
        }
    )
    report = assess_response(response, threshold=0.6)
    assert isinstance(report, HandoffReport)
    assert report.abstain is True
    signal = report.signals["dept"]
    assert isinstance(signal, HandoffSignal)
    assert signal.type == "choice"
    assert signal.confidence == pytest.approx(0.5)
    assert signal.abstain is True


def test_assess_response_score_uses_confidence_and_distribution() -> None:
    response = _response(
        {
            "urgency": ScoreAnswer(
                score=1.0,
                legend={0: "low", 1: "mid", 2: "high"},
                probabilities={0: 0.1, 1: 0.7, 2: 0.2},
                confidence=0.7,
            )
        }
    )
    report = assess_response(response, threshold=0.5)
    assert report.abstain is False
    assert report.signals["urgency"].margin == pytest.approx(0.5)


def test_assess_response_noul_uses_probability_directly() -> None:
    response = _response({"jailbreak": NoulAnswer(noul=0.85)})
    report = assess_response(response, threshold=0.9)
    signal = report.signals["jailbreak"]
    assert signal.type == "noul"
    assert signal.confidence == pytest.approx(0.85)
    assert signal.abstain is True


def test_assess_response_aggregates_any_abstention() -> None:
    response = _response(
        {
            "a": NoulAnswer(noul=0.95),
            "b": NoulAnswer(noul=0.4),
        }
    )
    report = assess_response(response, threshold=0.5)
    assert report.abstain is True
    assert report.signals["a"].abstain is False
    assert report.signals["b"].abstain is True


def test_assess_response_empty_answers_does_not_abstain() -> None:
    report = assess_response(_response({}), threshold=0.5)
    assert report.abstain is False
    assert report.signals == {}


def test_assess_response_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError):
        assess_response(_response({}), threshold=2.0)
