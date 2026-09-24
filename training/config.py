"""Reproducible configs and run metadata for the training pipeline.

Every stage takes a JSON config committed under ``training/configs/`` (seed + hyperparameters
+ paths). Artifacts record the environment and git commit so a run can be reproduced or
audited (TRAIN-06, NFR-C04). No secrets belong in configs.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from jeba import __version__ as jeba_version
from jeba.calibration import DEFAULT_GRID
from training.generate_data import DataConfig

CONFIG_DIR = Path(__file__).parent / "configs"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reject_unknown(data: Mapping[str, Any], known: set[str], what: str) -> None:
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {what} config keys: {sorted(unknown)}")


def load_data_config(path: str | Path) -> DataConfig:
    """Load a :class:`~training.generate_data.DataConfig` from JSON with strict keys."""
    data = _load_json(path)
    _reject_unknown(data, {field.name for field in fields(DataConfig)}, "data")
    if "languages" in data:
        data["languages"] = tuple(data["languages"])
    return DataConfig(**data)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Settings for temperature fitting."""

    bins: int = 10
    min_samples: int = 30
    grid: tuple[float, ...] = DEFAULT_GRID

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationConfig:
        _reject_unknown(data, {"bins", "min_samples", "grid"}, "calibration")
        payload = dict(data)
        if "grid" in payload:
            payload["grid"] = tuple(float(value) for value in payload["grid"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Settings for the evaluation harness."""

    bins: int = 10
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        _reject_unknown(data, {"bins", "limit"}, "eval")
        return cls(**dict(data))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git available
        return None


def environment() -> dict[str, Any]:
    """Capture the runtime environment for a reproducibility artifact."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jeba": jeba_version,
        "git_commit": _git_commit(),
    }
    for module_name in ("pydantic", "torch", "transformers", "peft"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        info[module_name] = getattr(module, "__version__", "unknown")
    return info


def write_artifact(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``payload`` plus the environment to a JSON artifact and return it."""
    artifact = {"environment": environment(), **payload}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return artifact


def load_finetune_config(path: str | Path):
    """Load a :class:`~training.finetune_rlcd.FinetuneConfig` (strict keys)."""
    from training.finetune_rlcd import FinetuneConfig

    return FinetuneConfig.from_dict(_load_json(path))


__all__ = [
    "CONFIG_DIR",
    "CalibrationConfig",
    "EvalConfig",
    "environment",
    "load_data_config",
    "load_finetune_config",
    "write_artifact",
]
