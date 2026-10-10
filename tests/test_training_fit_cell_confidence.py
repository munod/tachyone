"""Tests for ``training.fit_cell_confidence`` — the B-16 per-cell confidence fit."""

from __future__ import annotations

import bisect
import itertools
import json
from pathlib import Path

import pytest

from training.fit_cell_confidence import _quantile_buckets, fit_cells, main

pytestmark = pytest.mark.contract


def test_quantile_edges_reproduce_the_rank_assignment() -> None:
    """The runtime's ``bisect_left(edges, v)`` must equal the fit's rank bucket.

    Ties (strength is rounded to 6 decimals in the predictions) land in the first bucket
    they touch; an empty bucket must be skipped by lookups exactly as the rank assignment
    skips it.
    """
    values = [0.1, 0.2, 0.2, 0.35, 0.5, 0.9, 0.9, 0.9, 0.95, 1.0, -0.3, -0.1, -0.1]
    bins = 4
    buckets, edges = _quantile_buckets(values, bins)
    assert len(edges) == bins - 1
    assert all(right > left for left, right in itertools.pairwise(edges)), edges
    ordered = sorted(values)
    for value, fit_bucket in zip(values, buckets, strict=True):
        runtime_bucket = min(bins - 1, bisect.bisect_left(edges, value))
        rank = bisect.bisect_left(ordered, value)
        rank_bucket = min(bins - 1, rank * bins // len(values))
        assert runtime_bucket == rank_bucket == fit_bucket, (value, fit_bucket, runtime_bucket)


def test_quantile_edges_handle_the_all_tied_case() -> None:
    buckets, edges = _quantile_buckets([1.0] * 6, bins=3)
    assert buckets == [0, 0, 0, 0, 0, 0]
    assert all(right > left for left, right in itertools.pairwise(edges)), edges


def _rows() -> list[dict]:
    rows = []
    for kind in ("choice", "score"):
        for lang in ("pt", "de"):
            for index in range(20):
                peaked = 0.1 + (index % 5) * 0.2
                strength = 0.3 + (index % 4) * 0.15
                rows.append(
                    {
                        "kind": kind,
                        "serve_lang": lang,
                        "peaked": peaked,
                        "strength": strength,
                        "correct": index % 3 != 0,
                    }
                )
    return rows


def test_fit_cells_shape_keys_and_global_fallback() -> None:
    cells = fit_cells(_rows(), bins=4, shrink=4.0)
    assert set(cells) == {"choice:pt", "choice:de", "choice", "score:pt", "score:de", "score"}
    total_choice = sum(1 for row in _rows() if row["kind"] == "choice")
    assert cells["choice"]["n"] == total_choice, "the global cell covers every primitive row"
    assert cells["choice:pt"]["n"] == 20
    for cell in cells.values():
        rows = len(cell["peaked_edges"]) + 1
        columns = len(cell["strength_edges"]) + 1
        assert len(cell["table"]) == rows
        for table_row in cell["table"]:
            assert len(table_row) == columns
            for value in table_row:
                assert value is None or 0.0 <= value <= 1.0
        assert 0.0 <= cell["mean"] <= 1.0


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_main_writes_a_loadable_asset(tmp_path: Path) -> None:
    records = [
        {
            "id": "choice-000000",
            "type": "choice",
            "lang": "pt",
            "state": "Preciso cancelar o pedido de hoje",
            "instructions": "Quero falar com o suporte sobre cobrança",
            "criteria": {"billing": "cobranças", "tech": "bugs"},
            "target": "billing",
        },
        {
            "id": "choice-000001",
            "type": "choice",
            "lang": "pt",
            "state": "Uma cobrança duplicada na fatura",
            "instructions": "Você pode ajudar com esta cobrança?",
            "criteria": {"billing": "cobranças", "tech": "bugs"},
            "target": "tech",
        },
        {
            "id": "score-000000",
            "type": "score",
            "lang": "pt",
            "state": "Hoje preciso de ajuda urgente",
            "instructions": "Quão urgente é este pedido?",
            "criteria": ["low", "medium", "high"],
            "target": 2,
        },
    ]
    predictions = [
        {
            "id": "choice-000000",
            "type": "choice",
            "lang": "pt",
            "probabilities": {"billing": 0.7, "tech": 0.3},
            "target": "billing",
            "strength": 0.9,
        },
        {
            "id": "choice-000001",
            "type": "choice",
            "lang": "pt",
            "probabilities": {"billing": 0.6, "tech": 0.4},
            "target": "tech",
            "strength": 0.4,
        },
        {
            "id": "score-000000",
            "type": "score",
            "lang": "pt",
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            "target": 2,
            "strength": 0.6,
        },
    ]
    records_path = _write_jsonl(tmp_path / "records.jsonl", records)
    predictions_path = _write_jsonl(tmp_path / "predictions.jsonl", predictions)
    temperature = tmp_path / "temperature_calibration.json"
    temperature.write_text(
        json.dumps(
            {
                "per_primitive": {
                    "choice": {"temperature": 1.0, "by_language": {"pt": {"temperature": 2.0}}},
                    "score": {"temperature": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    prototypes = tmp_path / "state_prototypes.json"
    centroids = [[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6]
    prototypes.write_text(
        json.dumps(
            {"version": 1, "metric": "cosine", "k": 2, "dim": 8, "centroids": centroids},
        ),
        encoding="utf-8",
    )
    out = tmp_path / "confidence_calibration.json"

    exit_code = main(
        [
            "--predictions",
            str(predictions_path),
            "--records",
            str(records_path),
            "--temperature",
            str(temperature),
            "--prototypes",
            str(prototypes),
            "--out",
            str(out),
            "--bins",
            "2",
        ]
    )
    assert exit_code == 0
    asset = json.loads(out.read_text(encoding="utf-8"))
    assert asset["version"] == 1
    assert asset["prototypes"]["centroids"] == centroids
    assert set(asset["cells"]) == {"choice", "choice:pt", "score", "score:pt"}
    assert asset["meta"]["serve_key_diagonal"] == 1.0
    assert asset["meta"]["inputs"][str(predictions_path)]

    from tachyone.backends.encoder import ConfidenceCalibration

    def _no_warning(message: str) -> None:
        pytest.fail(f"unexpected warning: {message}")

    loaded = ConfidenceCalibration.from_dict(asset, warn=_no_warning)
    assert loaded is not None
    assert loaded.strength([1.0] + [0.0] * 7) == pytest.approx(1.0)
    # the serve key detected Portuguese, so the per-language cell answers
    value = loaded.cell_confidence("choice", "pt", {"billing": 1.0}, 0.9)
    assert value is not None and 0.0 <= value <= 1.0


def test_main_refuses_mismatched_pairing(tmp_path: Path) -> None:
    records_path = _write_jsonl(tmp_path / "records.jsonl", [{"id": "choice-0", "type": "choice"}])
    predictions_path = _write_jsonl(
        tmp_path / "predictions.jsonl",
        [{"id": "choice-0", "type": "choice", "probabilities": {}, "strength": 0.5}],
    )
    with pytest.raises(SystemExit, match="must pair"):
        main(
            [
                "--predictions",
                str(predictions_path),
                str(predictions_path),
                "--records",
                str(records_path),
                "--temperature",
                str(tmp_path / "t.json"),
                "--prototypes",
                str(tmp_path / "p.json"),
                "--out",
                str(tmp_path / "o.json"),
            ]
        )


def test_fit_cells_weights_shift_the_table_and_options_shape_keys() -> None:
    rows = [
        {
            "kind": "choice",
            "serve_lang": "en",
            "peaked": 0.9,
            "strength": 0.9,
            "correct": True,
            "weight": 1.0,
        },
        {
            "kind": "choice",
            "serve_lang": "en",
            "peaked": 0.9,
            "strength": 0.9,
            "correct": False,
            "weight": 3.0,
        },
    ]
    unweighted = fit_cells(rows, bins=2, shrink=0.0, kinds=("choice",), languages=("en",))
    assert set(unweighted) == {"choice", "choice:en"}, "options shape the cell keys"
    assert unweighted["choice"]["n"] == 2
    # the weight is honored in the mean (1 hit of weight 1 against 1 miss of weight 3)
    heavy = [
        {**rows[0], "weight": 1.0},
        {**rows[1], "weight": 3.0},
    ]
    assert fit_cells(heavy, bins=2, shrink=0.0, kinds=("choice",), languages=("en",))["choice"][
        "mean"
    ] == pytest.approx(0.25)
    assert rows[0]["correct"] is True and rows[1]["correct"] is False


def _record(identifier: str, state: str, target: str) -> dict:
    return {
        "id": identifier,
        "type": "choice",
        "lang": "en",
        "state": state,
        "instructions": "Please handle my request today",
        "criteria": {"billing": "charges", "tech": "bugs"},
        "target": target,
    }


def _prediction(identifier: str, probabilities: dict, strength: float, slice_: str) -> dict:
    return {
        "id": identifier,
        "type": "choice",
        "lang": "en",
        "probabilities": probabilities,
        "target": max(probabilities, key=lambda key: probabilities[key]),
        "strength": strength,
        "slice": slice_,
    }


def test_main_slice_balanced_pooling_and_base_carry_over(tmp_path: Path) -> None:
    records = [
        _record("choice-0", "I need to cancel today's order", "billing"),
        _record("choice-1", "There is a duplicated charge on my invoice", "billing"),
        _record("choice-2", "Please fix the bug in my account", "tech"),
        _record("choice-3", "The app crashes when I open it", "tech"),
    ]
    predictions = [
        _prediction("choice-0", {"billing": 0.9, "tech": 0.1}, 0.95, "training"),
        _prediction("choice-1", {"billing": 0.8, "tech": 0.2}, 0.90, "training"),
        _prediction("choice-2", {"tech": 0.7, "billing": 0.3}, 0.85, "training"),
        _prediction("choice-3", {"tech": 0.6, "billing": 0.4}, 0.80, "training"),
    ]
    holdout_records = [
        _record("choice-4", "Please cancel the order from today", "billing"),
        _record("choice-5", "Help me with the charge on the account", "billing"),
    ]
    holdout_predictions = [
        # deliberately wrong rows: the hard slice the balanced fit must not drown in
        _prediction("choice-4", {"tech": 0.9, "billing": 0.1}, 0.60, "holdout"),
        _prediction("choice-5", {"tech": 0.8, "billing": 0.2}, 0.55, "holdout"),
    ]
    records_path = _write_jsonl(tmp_path / "records.jsonl", records)
    predictions_path = _write_jsonl(tmp_path / "predictions.jsonl", predictions)
    holdout_records_path = _write_jsonl(tmp_path / "holdout_records.jsonl", holdout_records)
    holdout_predictions_path = _write_jsonl(
        tmp_path / "holdout_predictions.jsonl", holdout_predictions
    )
    temperature = tmp_path / "temperature_calibration.json"
    temperature.write_text(
        json.dumps({"per_primitive": {"choice": {"temperature": 1.0}}}), encoding="utf-8"
    )
    prototypes = tmp_path / "state_prototypes.json"
    prototypes.write_text(
        json.dumps(
            {
                "version": 1,
                "metric": "cosine",
                "k": 1,
                "dim": 8,
                "centroids": [[1.0] + [0.0] * 7],
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "confidence_calibration.json"
    base.write_text(
        json.dumps(
            {
                "version": 1,
                "noul": {"knots": [[0.5, 0.5], [1.0, 1.0]]},
                "provenance": {"rows": 1234, "predictions": "data/p3.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "fitted.json"

    exit_code = main(
        [
            "--predictions",
            str(predictions_path),
            str(holdout_predictions_path),
            "--records",
            str(records_path),
            str(holdout_records_path),
            "--temperature",
            str(temperature),
            "--prototypes",
            str(prototypes),
            "--out",
            str(out),
            "--bins",
            "2",
            "--kinds",
            "choice",
            "--languages",
            "en",
            "--base",
            str(base),
            "--balance-slices",
        ]
    )
    assert exit_code == 0
    asset = json.loads(out.read_text(encoding="utf-8"))

    # one factor: only what was asked gets fitted; the base keys ride along untouched
    assert set(asset["cells"]) == {"choice", "choice:en"}
    assert asset["noul"] == {"knots": [[0.5, 0.5], [1.0, 1.0]]}
    assert asset["provenance"] == {"rows": 1234, "predictions": "data/p3.jsonl"}
    assert asset["meta"]["base"]["carried"] == ["noul", "provenance"]
    assert len(asset["meta"]["base"]["sha256"]) == 64
    assert "slice-balanced" in asset["meta"]["recipe"]["basis"]

    # slice balance: pooled accuracy is 4/6, the balanced mean splits the difference (1.0, 0.0)
    assert asset["cells"]["choice"]["mean"] == pytest.approx(0.5)

    diagnostics = asset["meta"]["slice_diagnostics"]
    assert set(diagnostics) == {"all", "training", "holdout"}
    for entry in diagnostics.values():
        assert "choice" in entry
        assert entry["choice"]["map_ece"] >= 0.0
        assert entry["choice"]["temperature_ece"] >= 0.0

    from tachyone.backends.encoder import ConfidenceCalibration

    def _no_warning(message: str) -> None:
        pytest.fail(f"unexpected warning: {message}")

    loaded = ConfidenceCalibration.from_dict(asset, warn=_no_warning)
    assert loaded is not None
    assert loaded.noul_confidence(1.0) == pytest.approx(1.0)
    assert loaded.cell_confidence("choice", "en", {"billing": 1.0}, 0.9) is not None
