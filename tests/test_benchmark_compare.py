"""Unit tests for the head-to-head comparison harness (``benchmarks/compare.py``).

The engines themselves are exercised by the real runs recorded in ``benchmarks/report.md``.
What is tested here is everything that decides whether those numbers are *right*: the metric
definitions, the temperature fit, the row → wire mapping, and the renderer that ends up in the
docs. Nothing here needs pyarrow, torch or a model — the heavy imports stay inside functions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from benchmarks.compare import (
    _percentile,
    _question_payload,
    _record_to_row,
    _softmax,
    _unique_options,
    accuracy,
    brier,
    expected_calibration_error,
    fit_temperature,
    load_tachyone_records,
    render,
)


def test_softmax_is_normalized_and_shift_invariant() -> None:
    probs = _softmax([2.0, 0.0, -1.0])
    assert sum(probs) == pytest.approx(1.0)
    assert _softmax([100.0, 98.0, 97.0]) == pytest.approx(probs)


def test_temperature_flattens_a_distribution() -> None:
    logs = [math.log(value) for value in (0.8, 0.15, 0.05)]
    flat = _softmax([value / 6.0 for value in logs])
    assert max(flat) < 0.8
    assert sum(flat) == pytest.approx(1.0)
    # T=1 is the identity: rescaling the log-probabilities changes nothing.
    assert _softmax([value / 1.0 for value in logs]) == pytest.approx(_softmax(logs))


def test_fit_temperature_recovering_needs_no_further_flattening() -> None:
    """T is fitted on log-probabilities, so an already-calibrated output must come back T ≈ 1.

    The engine reports softmax(logits); the fit finds the T with softmax(log p / T) closest to
    the empirical frequencies — i.e. how much *more* the raw output has to be flattened.
    """
    overconfident = [_softmax([1.0, 0.0])] * 100  # always says class 0 with p = 0.731
    answers = [0] * 60 + [1] * 40  # but class 0 is right only 60% of the time
    assert fit_temperature(overconfident, answers) == pytest.approx(2.45, abs=0.1)

    calibrated = [[0.6, 0.4]] * 100  # confidence matches the empirical frequency
    assert fit_temperature(calibrated, answers) == pytest.approx(1.0, abs=0.05)


def test_accuracy_counts_a_failed_answer_as_wrong() -> None:
    assert accuracy([[0.1, 0.9], None, [0.8, 0.2]], [1, 1, 0]) == pytest.approx(2 / 3)


def test_ece_matches_a_hand_computed_two_bin_case() -> None:
    # 0.9 lands in bin (0.8, 0.9], 0.8 in bin (0.7, 0.8] — both perfectly accurate, so
    # ECE = 0.5*|1 - 0.9| + 0.5*|1 - 0.8| = 0.15
    assert expected_calibration_error([[0.9], [0.8]], [0, 0]) == pytest.approx(0.15)
    assert expected_calibration_error([[0.5], [0.5]], [0, 1]) == pytest.approx(0.0)


def test_brier_is_zero_for_a_certain_correct_answer() -> None:
    assert brier([[1.0, 0.0]], [0]) == pytest.approx(0.0)
    assert brier([[0.5, 0.5]], [0]) == pytest.approx(0.5)


def test_percentile_interpolates() -> None:
    assert math.isnan(_percentile([], 0.5))
    assert _percentile([7.0], 0.95) == 7.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_unique_options_keeps_duplicates_distinct() -> None:
    criteria = _unique_options(["a", "a", "b"])
    assert list(criteria) == ["a", "a (2)", "b"]
    assert set(criteria.values()) == {None}


def test_question_payload_from_a_foreign_option_set() -> None:
    row = {"state": "x", "question": "Which topic?", "options": ["a", "b"], "answer_index": 0}
    payload, keys = _question_payload(row)
    assert payload["type"] == "choice"
    assert list(payload["criteria"]) == ["a", "b"]
    assert keys == ["a", "b"]

    ordered = {**row, "options": ["low", "mid", "high"], "ordered": True}
    payload, keys = _question_payload(ordered)
    assert payload["type"] == "score"
    assert keys == ["0", "1", "2"]


def test_records_map_onto_options_and_answer_index() -> None:
    noul = _record_to_row(
        {
            "type": "noul",
            "state": "s",
            "instructions": "urgent?",
            "target": 1,
            "criteria": {"false": "not urgent", "true": "urgent"},
        }
    )
    assert noul["options"] == ["not urgent", "urgent"]
    assert noul["answer_index"] == 1
    assert noul["task"] == "noul"

    choice = _record_to_row(
        {
            "type": "choice",
            "state": "s",
            "instructions": "team?",
            "target": "billing",
            "criteria": {"billing": None, "tech": None},
        }
    )
    assert choice["options"] == ["billing", "tech"]
    assert choice["answer_index"] == 0

    score = _record_to_row(
        {
            "type": "score",
            "state": "s",
            "instructions": "rate",
            "target": 2,
            "criteria": ["a", "b", "c"],
        }
    )
    assert score["options"] == ["a", "b", "c"]
    assert score["answer_index"] == 2

    # The native question rides along, so Tachyone answers it in its own shape.
    assert _question_payload(noul)[0]["type"] == "noul"
    assert _question_payload(score)[1] == ["0", "1", "2"]


def test_load_tachyone_records_caps_per_task(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    records = [
        {
            "type": "noul",
            "state": f"s{i}",
            "instructions": "q",
            "target": 0,
            "criteria": {"false": "no", "true": "yes"},
        }
        for i in range(5)
    ] + [
        {
            "type": "choice",
            "state": f"c{i}",
            "instructions": "q",
            "target": "a",
            "criteria": {"a": None, "b": None},
        }
        for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    rows = load_tachyone_records(path, per_task=2)
    assert len(rows) == 4  # 2 noul + 2 choice
    assert {row["task"] for row in rows} == {"noul", "choice"}

    with pytest.raises(SystemExit):
        load_tachyone_records(tmp_path / "missing.jsonl", per_task=2)


def _artifact(name: str, accuracy_value: float, p50: float) -> dict[str, Any]:
    return {
        "engine": name,
        "dataset": {"rows": 10, "kind": "probe"},
        "temperature": {"fitted": True, "value": 1.5, "source": "validation split (capped)"},
        "raw": {
            "n": 10,
            "answered": 10,
            "accuracy": accuracy_value,
            "ece": 0.2,
            "brier": 0.4,
            "mean_confidence": 0.7,
        },
        "calibrated": {
            "n": 10,
            "answered": 10,
            "accuracy": accuracy_value,
            "ece": 0.05,
            "brier": 0.3,
            "mean_confidence": 0.6,
        },
        "per_task_raw": {},
        "per_task_calibrated": {
            "choice": {
                "n": 10,
                "answered": 10,
                "accuracy": accuracy_value,
                "ece": 0.05,
                "brier": 0.3,
                "mean_confidence": 0.6,
            }
        },
        "latency_ms": {"n": 10, "p50": p50, "p95": p50 * 2, "mean": p50},
        "throughput": {"items_per_s": 1000.0 / p50, "note": "sequential"},
        "compliance": {"answered": 10, "failed": 0, "rate": 1.0},
        "memory": {"peak_rss_mb": 100.0, "peak_vram_mb": 50.0, "measured_on": "this process"},
        "engine_config": {},
        "environment": {},
    }


def test_render_writes_both_tables_and_the_per_task_breakdown(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(_artifact("tachyone", 0.85, 22.0)), encoding="utf-8")
    second.write_text(json.dumps(_artifact("peer", 0.70, 350.0)), encoding="utf-8")
    out = tmp_path / "table.md"

    render([first, second], out)

    text = out.read_text(encoding="utf-8")
    assert "### Quality" in text and "### Performance" in text
    assert "ECE raw" in text and "JSON ok" in text
    assert "| tachyone | 10 | 10 | 0.850 | 0.200 | 0.050 | 0.300 | 0.600 |" in text
    assert "22.00" in text and "350.00" in text  # engines sorted by name: peer before tachyone
    assert "### Per-task accuracy (calibrated)" in text and "| choice |" in text
