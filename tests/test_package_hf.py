"""Tests for the Hugging Face packaging helper (release support)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.package_hf import main, package


def _fake_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "checkpoints" / "en"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "answerdotai/ModernBERT-large"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "finetune_config.json").write_text("{}", encoding="utf-8")
    (adapter / "temperature_calibration.json").write_text("{}", encoding="utf-8")
    return adapter


def test_package_assembles_hf_layout(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    out = package(adapter, tmp_path / "hf-en", checkpoint_name="en")
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "temperature_calibration.json").exists()
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---")  # HF front matter preserved
    assert "# tachyone-en (System One decision engine)" in readme


def test_package_requires_adapter_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no adapter files"):
        package(empty, tmp_path / "out", checkpoint_name="x")


def test_package_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter directory not found"):
        package(tmp_path / "nope", tmp_path / "out", checkpoint_name="x")


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    adapter = _fake_adapter(tmp_path)
    out = tmp_path / "hf"
    assert main(["--adapter", str(adapter), "--out", str(out), "--name", "multi"]) == 0
    assert (out / "README.md").exists()
    assert "hf upload" in capsys.readouterr().out
