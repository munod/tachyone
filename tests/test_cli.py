"""Unit tests for presets and the CLI (M2-T6, SERVE-03 / SERVE-04 / SERVE-07)."""

from __future__ import annotations

import json

import pytest

from tachyone.cli import main
from tachyone.presets import PRESETS, get_preset
from tachyone.wire import SystemOneRequest


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_expand_to_valid_questions(name: str) -> None:
    questions = get_preset(name)
    assert questions
    SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": questions})
    for question in questions.values():
        assert question.type in {"noul", "choice", "score"}


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        get_preset("nope")


def test_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-presets"]) == 0
    printed = capsys.readouterr().out.split()
    assert set(printed) == set(PRESETS)


def test_prints_questions_without_predicting(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--preset", "triage", "hello"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "department" in payload["questions"]
    assert payload["questions"]["department"]["type"] == "choice"


def test_predict_with_fake_backend(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--predict", "--preset", "triage", "--backend", "fake", "refund please"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"model", "answers", "usage"}
    assert set(result["answers"]) == {"department", "urgency", "frustration", "churn_risk"}


def test_threshold_appends_sibling_handoff(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["--predict", "--preset", "triage", "--backend", "fake", "--threshold", "0.9", "refund"]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    # Canonical keys are untouched; `handoff` is an additive sibling.
    assert set(result) == {"model", "answers", "usage", "handoff"}
    assert isinstance(result["handoff"]["abstain"], bool)
    assert result["handoff"]["threshold"] == pytest.approx(0.9)
    assert set(result["handoff"]["signals"]) == set(result["answers"])
    signal = result["handoff"]["signals"]["department"]
    assert set(signal) == {"type", "confidence", "entropy", "margin", "abstain"}


def test_threshold_default_output_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--predict", "--preset", "triage", "--backend", "fake", "refund please"]
    main(args)
    without = capsys.readouterr().out
    main([*args, "--threshold", "0.9"])
    with_threshold = capsys.readouterr().out
    # Adding --threshold only appends a sibling: every canonical key/value is identical.
    assert "handoff" not in json.loads(without)
    assert json.loads(without) == {
        key: value for key, value in json.loads(with_threshold).items() if key != "handoff"
    }


def test_threshold_requires_predict() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "triage", "--threshold", "0.5", "text"])
    assert excinfo.value.code == 2


def test_threshold_out_of_range_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--predict", "--preset", "triage", "--backend", "fake", "--threshold", "1.5", "x"])
    assert excinfo.value.code == 2


def test_unknown_preset_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "nope", "text"])
    assert excinfo.value.code == 2


def test_predict_requires_text() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--predict", "--preset", "triage"])
    assert excinfo.value.code == 2
