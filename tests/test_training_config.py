"""Tests for reproducible configs and run metadata (M4-T6, TRAIN-06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.config import (
    CONFIG_DIR,
    CalibrationConfig,
    EvalConfig,
    environment,
    load_data_config,
    load_finetune_config,
    write_artifact,
)

_COMMITTED = sorted(path.name for path in CONFIG_DIR.glob("*.json"))


def test_committed_configs_exist() -> None:
    assert {
        "data.json",
        "data_multi.json",
        "finetune_en.json",
        "finetune_multi.json",
        "calibration.json",
    } <= set(_COMMITTED)


def test_load_committed_configs() -> None:
    data = load_data_config(CONFIG_DIR / "data.json")
    assert data.seed == 42
    assert "en" in data.languages
    english = load_finetune_config(CONFIG_DIR / "finetune_en.json")
    multi = load_finetune_config(CONFIG_DIR / "finetune_multi.json")
    assert english.model_id != multi.model_id
    assert multi.max_len >= english.max_len
    calibration = CalibrationConfig.from_dict(
        json.loads((CONFIG_DIR / "calibration.json").read_text(encoding="utf-8"))
    )
    assert calibration.min_samples == 30
    assert EvalConfig.from_dict({"bins": 5, "limit": 10}).bins == 5


def test_data_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"seed": 1, "bogus": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown data config keys"):
        load_data_config(path)


def test_finetune_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"epochs": 1, "bogus": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_finetune_config(path)


def test_environment_has_core_fields() -> None:
    info = environment()
    assert info["python"]
    assert info["platform"]
    assert info["jeba"]
    assert "git_commit" in info


def test_write_artifact_includes_environment(tmp_path: Path) -> None:
    out = tmp_path / "artifact.json"
    artifact = write_artifact(out, {"metrics": {"accuracy": 0.9}})
    assert artifact["environment"]["jeba"]
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["metrics"]["accuracy"] == 0.9
    assert "environment" in on_disk
