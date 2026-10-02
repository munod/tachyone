"""Tests for the P3 confidence fitter (B-13, JB-12).

Everything runs on a synthetic predictions file — no model, no network. The assertions
that matter are the pre-registered protocol: pooled temperature fit, ``score`` pinned with
its reason on file, a monotone ``noul`` map from the evidence signal, and both assets
round-tripping into the *runtime* loaders (a map the engine cannot read is worthless).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from training.fit_confidence import FitError, fit


def _bank(dim: int = 4) -> dict:
    centroids = []
    for axis in range(dim):
        vector = [0.0] * dim
        vector[axis] = 1.0
        centroids.append(vector)
    return {
        "version": 1,
        "metric": "cosine",
        "k": len(centroids),
        "dim": dim,
        "seed": 0,
        "source": {"data": "train_en_domains.jsonl", "records": 21000, "sha256": "0" * 64},
        "model": {"id": "answerdotai/ModernBERT-large", "adapter": "checkpoints/en"},
        "centroids": centroids,
    }


def _row(
    kind: str, *, rid: str, slc: str, strength: float, value, target, lang: str = "en"
) -> dict:
    row = {"id": rid, "type": kind, "lang": lang, "slice": slc, "target": target}
    if kind == "noul":
        row["probabilities"] = float(value)
    else:
        row["probabilities"] = dict(value)
    row["strength"] = strength
    return row


def _fixture(tmp_path: Path, *, with_strength: bool = True) -> Path:
    """Three evidence levels per slice; accuracy rises with strength (a fittable signal)."""
    rng = random.Random(3)
    rows: list[dict] = []
    # training slice: near the bank, the model is right and should be confident
    for index in range(30):
        target = index % 2
        rows.append(
            _row(
                "noul",
                rid=f"cal-trn-noul-{index:05d}",
                slc="training",
                strength=0.99,
                value=0.97 if target else 0.03,
                target=target,
            )
        )
    # remainder: mid strength, half right
    for index in range(30):
        target = index % 2
        correct = index % 2 == 0
        value = 0.8 if (correct and target) else (0.2 if correct else rng.uniform(0.4, 0.6))
        rows.append(
            _row(
                "noul",
                rid=f"cal-rem-boolq-{index:05d}",
                slc="remainder",
                strength=0.72 if index % 2 else 0.73,
                value=value,
                target=target,
            )
        )
    for index in range(40):
        slc = "training" if index < 20 else "remainder"
        strength = 0.99 if slc == "training" else 0.7
        correct = slc == "training" or index % 4 == 0
        rows.append(
            _row(
                "choice",
                rid=f"cal-x-choice-{index:05d}",
                slc=slc,
                strength=strength,
                value={"a": 0.9 if correct else 0.4, "b": 0.1 if correct else 0.6},
                target="a" if correct else "b",
            )
        )
    for index in range(20):
        rows.append(
            _row(
                "score",
                rid=f"cal-trn-score-{index:05d}",
                slc="training",
                strength=0.99,
                value={"0": 0.05, "1": 0.9 if index % 3 else 0.1, "2": 0.05 if index % 3 else 0.9},
                target=1 if index % 3 else 2,
            )
        )
    path = tmp_path / "preds.jsonl"
    path.write_text(
        "".join(
            json.dumps({k: v for k, v in row.items() if with_strength or k != "strength"}) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _write_bank(tmp_path: Path) -> Path:
    path = tmp_path / "state_prototypes.json"
    path.write_text(json.dumps(_bank()), encoding="utf-8")
    return path


def test_fit_reports_the_protocol_and_keeps_score_pinned(tmp_path: Path) -> None:
    report = fit(
        predictions=_fixture(tmp_path),
        prototypes=_write_bank(tmp_path),
        out_dir=tmp_path / "ckpt",
        dry_run=True,
    )
    per_primitive = report["per_primitive"]
    assert set(per_primitive) == {"noul", "choice", "score"}
    assert per_primitive["score"]["temperature"] == 0.1
    assert per_primitive["score"]["pinned"] is True
    assert "expected value" in per_primitive["score"]["reason"]
    assert report["pin"]["score"] == 0.1
    assert isinstance(per_primitive["choice"]["temperature"], float)
    assert "by_language" in per_primitive["choice"]


def test_fit_map_is_monotone_and_covers_both_regimes(tmp_path: Path) -> None:
    report = fit(
        predictions=_fixture(tmp_path),
        prototypes=_write_bank(tmp_path),
        out_dir=tmp_path / "ckpt",
        dry_run=True,
    )
    knots = report["confidence"]["knots"]
    assert len(knots) >= 2
    strengths = [strength for strength, _ in knots]
    levels = [level for _, level in knots]
    assert strengths == sorted(strengths), "knots are ordered by evidence"
    assert levels == sorted(levels), "isotonic: weaker evidence never promises more"
    assert all(0.0 <= level <= 1.0 for level in levels)
    assert knots[-1][0] > 0.9, "the in-domain regime is fitted, not extrapolated"
    assert knots[0][0] < 0.9, "the off-domain regime is fitted too"


def test_fit_diagnostics_show_both_mechanisms_per_slice(tmp_path: Path) -> None:
    report = fit(
        predictions=_fixture(tmp_path),
        prototypes=_write_bank(tmp_path),
        out_dir=tmp_path / "ckpt",
        dry_run=True,
    )
    diagnostics = report["diagnostics"]
    assert set(diagnostics["noul"]["temperature"]) == {"training", "remainder", "all"}
    assert set(diagnostics["noul"]["map"]) == {"training", "remainder", "all"}
    assert diagnostics["noul"]["map"]["training"] <= 0.05, "the map keeps the in-domain fit"
    assert (
        diagnostics["noul"]["map"]["training"] < diagnostics["noul"]["temperature"]["training"]
    ), "one global temperature cannot hold both regimes — that is why P3 fits a map"
    assert "choice" in diagnostics and "score" in diagnostics


def test_fit_writes_assets_the_runtime_can_read(tmp_path: Path) -> None:
    from tachyone.backends.encoder import ConfidenceCalibration, load_temperatures
    from tachyone.router import CheckpointInfo

    out = tmp_path / "ckpt"
    fit(
        predictions=_fixture(tmp_path),
        prototypes=_write_bank(tmp_path),
        out_dir=out,
    )
    assert (out / "temperature_calibration.json").exists()
    assert (out / "confidence_calibration.json").exists()

    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=str(out))
    temperatures = load_temperatures(info, models_dir=str(tmp_path))
    assert temperatures["score"] == 0.1
    assert isinstance(temperatures["choice"], float)

    def _no_warnings(message: str) -> None:
        pytest.fail(f"the runtime refused the fitted asset: {message}")

    payload = json.loads((out / "confidence_calibration.json").read_text(encoding="utf-8"))
    asset = ConfidenceCalibration.from_dict(payload, warn=_no_warnings)
    assert asset is not None
    assert asset.dim == 4
    assert asset.noul_knots
    # the embedded bank is byte-identical to the file it was fitted against
    assert payload["prototypes"] == json.loads(_write_bank(tmp_path).read_text(encoding="utf-8"))
    assert payload["provenance"]["prototypes_sha256"]


def test_fit_records_provenance_of_both_inputs(tmp_path: Path) -> None:
    report = fit(
        predictions=_fixture(tmp_path),
        prototypes=_write_bank(tmp_path),
        out_dir=tmp_path / "ckpt",
        dry_run=True,
    )
    provenance = report["confidence"]["provenance"]
    assert len(provenance["predictions_sha256"]) == 64
    assert len(provenance["prototypes_sha256"]) == 64
    assert provenance["rows"] == report["confidence"]["n"] + 60  # noul rows + choice + score
    assert provenance["noul_rows"] == 60


def test_fit_validates_its_inputs(tmp_path: Path) -> None:
    bank = _write_bank(tmp_path)
    with pytest.raises(FitError, match="missing predictions"):
        fit(predictions=tmp_path / "nope.jsonl", prototypes=bank, out_dir=tmp_path)
    with pytest.raises(FitError, match="missing prototype bank"):
        fit(predictions=_fixture(tmp_path), prototypes=tmp_path / "nope.json", out_dir=tmp_path)

    no_noul = tmp_path / "no_noul.jsonl"
    no_noul.write_text(
        json.dumps(
            {
                "id": "a",
                "type": "choice",
                "lang": "en",
                "slice": "training",
                "target": "a",
                "probabilities": {"a": 1.0},
                "strength": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FitError, match="no noul rows"):
        fit(predictions=no_noul, prototypes=bank, out_dir=tmp_path)

    no_strength = _fixture(tmp_path, with_strength=False)
    with pytest.raises(FitError, match="prototypes"):
        fit(predictions=no_strength, prototypes=bank, out_dir=tmp_path)


def test_fit_refuses_a_flat_strength_distribution(tmp_path: Path) -> None:
    rows = [
        json.dumps(
            {
                "id": f"noul-{index:03d}",
                "type": "noul",
                "lang": "en",
                "slice": "training",
                "target": index % 2,
                "probabilities": 0.9 if index % 2 else 0.1,
                "strength": 0.99,
            }
        )
        for index in range(40)
    ]
    path = tmp_path / "flat.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(FitError, match="collapsed"):
        fit(predictions=path, prototypes=_write_bank(tmp_path), out_dir=tmp_path)
