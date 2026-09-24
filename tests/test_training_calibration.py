"""Tests for calibration fitting (M4-T4, CAL-02/CAL-04, TRAIN-04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.fit_calibration import (
    CalibrationExample,
    fit_temperature_report,
    load_examples,
    main,
    save_report,
)


def _overconfident_choice(n: int = 100, correct: int = 60) -> list[CalibrationExample]:
    examples = []
    for index in range(n):
        target = "billing" if index < correct else "technical"
        examples.append(CalibrationExample("choice", {"billing": 0.9, "technical": 0.1}, target))
    return examples


def _overconfident_noul(n: int = 100, correct: int = 60) -> list[CalibrationExample]:
    examples = []
    for index in range(n):
        target = 1 if index < correct else 0
        examples.append(CalibrationExample("noul", {"true": 0.9, "false": 0.1}, target))
    return examples


def test_choice_temperature_reduces_ece() -> None:
    report = fit_temperature_report(_overconfident_choice())
    choice = report["per_primitive"]["choice"]
    assert choice["temperature"] > 1.0
    assert choice["ece_after"] < choice["ece_before"]


def test_noul_temperature_reduces_ece() -> None:
    report = fit_temperature_report(_overconfident_noul())
    noul = report["per_primitive"]["noul"]
    assert noul["temperature"] > 1.0
    assert noul["ece_after"] < noul["ece_before"]


def test_small_split_warns() -> None:
    report = fit_temperature_report(_overconfident_choice(n=10, correct=6), min_samples=30)
    assert any("choice" in warning for warning in report["warnings"])


def test_load_examples_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "noul", "probabilities": 0.8, "target": 1}),
                json.dumps(
                    {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}, "target": "a"}
                ),
            ]
        ),
        encoding="utf-8",
    )
    examples = load_examples(path)
    assert examples[0].type == "noul"
    assert examples[0].probabilities == {"true": 0.8, "false": pytest.approx(0.2)}
    assert examples[1].target == "a"


@pytest.mark.parametrize(
    "line",
    [
        json.dumps({"type": "bogus", "probabilities": {}, "target": 0}),
        json.dumps({"type": "noul", "probabilities": "high", "target": 1}),
        json.dumps({"type": "choice", "probabilities": 0.5, "target": "a"}),
    ],
)
def test_load_examples_rejects_bad_lines(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        load_examples(path)


def test_save_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = fit_temperature_report(_overconfident_choice())
    out = tmp_path / "temperature.json"
    save_report(report, out)
    assert json.loads(out.read_text(encoding="utf-8"))["per_primitive"]["choice"]["temperature"]

    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        "\n".join(
            json.dumps({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}, "target": t})
            for t in ["a"] * 6 + ["b"] * 4
        ),
        encoding="utf-8",
    )
    assert main(["--calibration", str(preds), "--out", str(out)]) == 0
    assert "per_primitive" in capsys.readouterr().out
