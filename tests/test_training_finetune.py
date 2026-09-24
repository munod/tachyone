"""Tests for the fine-tuning config and dry-run path (M4-T2, TRAIN-02/TRAIN-06)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from training.finetune_rlcd import (
    FinetuneConfig,
    load_config,
    read_records,
    run,
    summarize_dataset,
)
from training.generate_data import DataConfig, generate


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    generate(DataConfig(seed=1, per_type=6, languages=("en", "pt")), path)
    return path


def test_config_roundtrip() -> None:
    config = FinetuneConfig(model_id="x", epochs=2)
    assert FinetuneConfig.from_dict(config.to_dict()) == config


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        FinetuneConfig.from_dict({"epochs": 1, "bogus": 2})


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 5, "lora_rank": 8}), encoding="utf-8")
    config = load_config(path)
    assert config.epochs == 5
    assert config.lora_rank == 8


def test_summarize_dataset(tmp_path: Path) -> None:
    stats = summarize_dataset(_dataset(tmp_path))
    assert stats["total"] == 18
    assert stats["per_type"] == {"noul": 6, "choice": 6, "score": 6}
    assert set(stats["per_lang"]) <= {"en", "pt"}


def test_read_records_limit(tmp_path: Path) -> None:
    assert len(list(read_records(_dataset(tmp_path), limit=4))) == 4


def test_dry_run_returns_report(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    config = FinetuneConfig(data_path=str(data))
    report = run(config, dry_run=True)
    assert report["mode"] == "dry-run"
    assert report["dataset"]["total"] == 18
    assert report["config"]["seed"] == 42


def test_train_without_extra_raises(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the training path is exercised elsewhere")
    config = FinetuneConfig(data_path=str(_dataset(tmp_path)), out_dir=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="train extra"):
        run(config, dry_run=False)
