"""Tests for the frozen-trunk choice-head bank fit (ADR-0016 §5, B5).

The pure parts — config, trunk validation, the two arm dirs and the gate they ship — run
without torch; the fit path itself runs against a stubbed encoder so it exercises the real
optimizer, warm start and writers in milliseconds.
"""

from __future__ import annotations

import importlib.util
import json
import zlib
from pathlib import Path
from typing import Any

import pytest

from tachyone.backends.encoder import CHOICE_HEAD_ASSET, ChoiceHeadBank
from tachyone.primitives import ChoiceQuestion
from training.fit_choice_bank import (
    FIT_RECIPE_ASSET,
    FitBankConfig,
    load_config,
    run,
    trunk_shared_head,
    write_arm,
)
from training.generate_data import DataConfig, generate

_RANK = 2
_HIDDEN = 8


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    generate(DataConfig(seed=1, per_type=3, languages=("en",), domains=("support", "voice")), path)
    return path


def _scorer(value: float = 0.25) -> dict[str, Any]:
    return {
        "rank": _RANK,
        "w1": [[value] * (2 * _HIDDEN)] * _RANK,
        "w2": [[value] * _HIDDEN] * _RANK,
    }


def _trunk(tmp_path: Path, *, rank: int = _RANK) -> Path:
    """A fake frozen trunk: the adapter files plus a legacy single-scorer head."""
    trunk = tmp_path / "trunk"
    trunk.mkdir(exist_ok=True)
    (trunk / "adapter_config.json").write_text(json.dumps({"r": 2}), encoding="utf-8")
    (trunk / "adapter_model.safetensors").write_bytes(b"frozen-weights")
    payload = dict(_scorer(), rank=rank)
    payload["w1"] = [[0.1] * (2 * _HIDDEN)] * rank
    payload["w2"] = [[0.1] * _HIDDEN] * rank
    (trunk / CHOICE_HEAD_ASSET).write_text(json.dumps(payload), encoding="utf-8")
    return trunk


def _config(tmp_path: Path, **overrides: Any) -> FitBankConfig:
    defaults: dict[str, Any] = {
        "data_path": str(_dataset(tmp_path)),
        "trunk_adapter": str(_trunk(tmp_path)),
        "control_dir": str(tmp_path / "ctrl"),
        "bank_dir": str(tmp_path / "bank"),
        "choice_rank": _RANK,
    }
    defaults.update(overrides)
    return FitBankConfig(**defaults)


def _fake_encode(texts: list[str]) -> list[list[float]]:
    """Deterministic 8-dim stand-in for the trunk: same text → same embedding."""
    return [
        [float((zlib.crc32(text.encode("utf-8")) >> (4 * index)) & 15) for index in range(_HIDDEN)]
        for text in texts
    ]


def _thermostat_question() -> ChoiceQuestion:
    return ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Which device is broken?",
            "criteria": {"thermostat": "temperature control", "zzz": "unknown"},
        }
    )


def test_config_roundtrip() -> None:
    config = FitBankConfig(model_id="x", epochs=2)
    assert FitBankConfig.from_dict(config.to_dict()) == config


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        FitBankConfig.from_dict({"epochs": 1, "bogus": 2})


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 5, "choice_rank": 128}), encoding="utf-8")
    config = load_config(path)
    assert config.epochs == 5
    assert config.choice_rank == 128


def test_dry_run_reports_the_plan_without_torch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = run(config, dry_run=True)
    assert report["mode"] == "dry-run"
    assert report["choice_domains"] == ["support", "voice"]
    assert report["languages"] == ["en"]
    assert report["trunk"] == {"adapter": config.trunk_adapter, "choice_head_rank": _RANK}
    assert report["plan"]["control"] == config.control_dir
    assert report["plan"]["bank"] == config.bank_dir
    assert report["dataset"]["per_type"]["choice"] > 0
    assert "arms" not in report  # nothing was fitted


def test_dry_run_rejects_records_without_a_domain(tmp_path: Path) -> None:
    """The isolate exists to fit domain keys; data that has none fails fast (L-011)."""
    data = tmp_path / "legacy.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "choice-0",
                "type": "choice",
                "lang": "en",
                "state": "x",
                "instructions": "Which?",
                "criteria": {"a": None},
                "target": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="domain"):
        run(_config(tmp_path, data_path=str(data)), dry_run=True)


def test_dry_run_fails_fast_on_an_unknown_domain(tmp_path: Path) -> None:
    data = tmp_path / "bad.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "choice-0",
                "type": "choice",
                "lang": "en",
                "domain": "telepathy",
                "state": "x",
                "instructions": "Which?",
                "criteria": {"a": None},
                "target": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="telepathy"):
        run(_config(tmp_path, data_path=str(data)), dry_run=True)


def test_dry_run_requires_a_local_trunk_with_a_head(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="local directory"):
        run(_config(tmp_path, trunk_adapter=str(tmp_path / "nowhere")), dry_run=True)

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match=r"ships no choice_head\.json"):
        run(_config(tmp_path, trunk_adapter=str(bare)), dry_run=True)


def test_dry_run_rejects_a_rank_that_disagrees_with_the_trunk(tmp_path: Path) -> None:
    """The arms warm-start from the trunk, so a mismatch is a bug, not a knob (L-011)."""
    with pytest.raises(SystemExit, match="choice_rank 16 disagrees"):
        run(_config(tmp_path, choice_rank=16), dry_run=True)


def test_trunk_shared_head_accepts_both_asset_shapes(tmp_path: Path) -> None:
    trunk = _trunk(tmp_path)
    legacy = trunk_shared_head(trunk)
    assert legacy["rank"] == _RANK

    keyed = tmp_path / "keyed"
    keyed.mkdir()
    (keyed / CHOICE_HEAD_ASSET).write_text(
        json.dumps({"shared": legacy, "domains": {}}), encoding="utf-8"
    )
    assert trunk_shared_head(keyed) == legacy

    (keyed / CHOICE_HEAD_ASSET).write_text(json.dumps({"domains": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="usable 'shared' head"):
        trunk_shared_head(keyed)


def test_trunk_shared_head_rejects_a_broken_scorer(tmp_path: Path) -> None:
    trunk = _trunk(tmp_path)
    (trunk / CHOICE_HEAD_ASSET).write_text(json.dumps({"rank": 2, "w1": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"w1.*w2"):
        trunk_shared_head(trunk)

    (trunk / CHOICE_HEAD_ASSET).write_text(
        json.dumps({"rank": 2, "w1": [[0.1] * 16] * 2, "w2": [[0.1] * 7] * 2}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="twice w2 rows"):
        trunk_shared_head(trunk)


def test_write_arm_shapes_and_gate(tmp_path: Path) -> None:
    trunk = _trunk(tmp_path)
    control = write_arm(
        tmp_path / "ctrl",
        trunk=trunk,
        shared=_scorer(),
        domains={},
        languages=("en",),
        recipe={"arm": "control"},
    )
    # The control arm ships today's legacy asset: the payload IS the scorer dict.
    control_payload = json.loads((control / CHOICE_HEAD_ASSET).read_text(encoding="utf-8"))
    assert control_payload == _scorer()
    assert "shared" not in control_payload
    assert (control / "adapter_model.safetensors").read_bytes() == b"frozen-weights"

    bank = write_arm(
        tmp_path / "bank",
        trunk=trunk,
        shared=_scorer(),
        domains={"support": _scorer(0.1), "voice": _scorer(0.2)},
        languages=("en",),
        recipe={"arm": "bank"},
    )
    payload = json.loads((bank / CHOICE_HEAD_ASSET).read_text(encoding="utf-8"))
    assert set(payload) == {"shared", "domains"}
    assert set(payload["domains"]) == {"support", "voice"}
    assert payload["domains"]["voice"]["head"] == _scorer(0.2)
    assert payload["domains"]["voice"]["signatures"]

    # The asset answers through the runtime gate: its own vocabulary routes, nothing else
    # falls back to shared (the worst a wrong gate can do).
    asset = ChoiceHeadBank.from_dict(payload)
    assert asset is not None
    assert asset.select_key(_thermostat_question()) == "voice"
    assert asset.select_key(_thermostat_question(), hint="support") == "support"
    assert asset.select_key(_thermostat_question(), hint="nope") == "voice"
    bare = ChoiceQuestion.model_validate(
        {"type": "choice", "instructions": "qqq?", "criteria": {"zzz1": "www2"}}
    )
    assert asset.select_key(bare) is None
    assert asset.select(bare) is asset.shared


def test_write_arm_ships_the_recipe_and_is_deterministic(tmp_path: Path) -> None:
    trunk = _trunk(tmp_path)
    heads: list[bytes] = []
    for name in ("first", "second"):
        arm = write_arm(
            tmp_path / name,
            trunk=trunk,
            shared=_scorer(),
            domains={"voice": _scorer(0.3)},
            languages=("en",),
            recipe={"arm": "bank", "seed": 2},
        )
        recipe = json.loads((arm / FIT_RECIPE_ASSET).read_text(encoding="utf-8"))
        assert recipe["arm"] == "bank"
        assert recipe["seed"] == 2
        assert (arm / "adapter_config.json").read_text(encoding="utf-8") == json.dumps({"r": 2})
        heads.append((arm / CHOICE_HEAD_ASSET).read_bytes())
    assert heads[0] == heads[1]  # same inputs → same bytes


def test_write_arm_needs_the_trunk_adapter_files(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match=r"missing adapter_config\.json"):
        write_arm(
            tmp_path / "out", trunk=bare, shared=_scorer(), domains={}, languages=(), recipe={}
        )


def _run_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> dict[str, Any]:
    import training.fit_choice_bank as fit

    monkeypatch.setattr(fit, "load_encoder", lambda *args, **kwargs: _fake_encode)
    return fit.run(_config(tmp_path, device="cpu", epochs=1, batch_size=4, **overrides))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the train extra")
def test_fit_writes_both_loadable_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _run_fit(tmp_path, monkeypatch)
    assert report["mode"] == "fit"
    assert set(report["arms"]) == {"control", "bank"}
    assert report["encoded_records"] == 6  # 3 records x 2 domains
    for arm in ("control", "bank"):
        metrics = report["arms"][arm]
        assert metrics["train_records"] + metrics["val_records"] == 6
        assert isinstance(metrics["val_loss"], float)
        assert Path(metrics["dir"]).is_dir()

    control_payload = json.loads(
        (tmp_path / "ctrl" / CHOICE_HEAD_ASSET).read_text(encoding="utf-8")
    )
    assert control_payload["rank"] == _RANK
    assert "shared" not in control_payload  # legacy shape: shared-only control

    bank_payload = json.loads((tmp_path / "bank" / CHOICE_HEAD_ASSET).read_text(encoding="utf-8"))
    assert set(bank_payload) == {"shared", "domains"}
    assert set(bank_payload["domains"]) == {"support", "voice"}
    asset = ChoiceHeadBank.from_dict(bank_payload)
    assert asset is not None
    assert asset.select_key(_thermostat_question()) == "voice"

    # Both dirs are loadable adapters: the frozen trunk's files came along, and each arm
    # carries its own recipe (L-011).
    for directory in (tmp_path / "ctrl", tmp_path / "bank"):
        assert (directory / "adapter_model.safetensors").read_bytes() == b"frozen-weights"
        recipe = json.loads((directory / FIT_RECIPE_ASSET).read_text(encoding="utf-8"))
        assert recipe["trunk_adapter"] == str(tmp_path / "trunk")
        assert recipe["config"]["choice_rank"] == _RANK
    bank_recipe = json.loads((tmp_path / "bank" / FIT_RECIPE_ASSET).read_text(encoding="utf-8"))
    assert bank_recipe["arm"] == "bank"
    assert bank_recipe["domains"] == ["support", "voice"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the train extra")
def test_fit_is_deterministic_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _run_fit(tmp_path / "a", monkeypatch)
    second = _run_fit(tmp_path / "b", monkeypatch)
    assert first["arms"]["control"]["val_loss"] == second["arms"]["control"]["val_loss"]
    for name in (CHOICE_HEAD_ASSET, "temperature.json"):  # the learned artifacts, not the paths
        left = (tmp_path / "a" / "bank" / name).read_bytes()
        right = (tmp_path / "b" / "bank" / name).read_bytes()
        assert left == right, name
