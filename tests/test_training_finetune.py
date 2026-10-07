"""Tests for the fine-tuning config and dry-run path (M4-T2, TRAIN-02/TRAIN-06)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from training.finetune_rlcd import (
    FinetuneConfig,
    bucket_by_cell,
    cell_of,
    choice_domains,
    choice_head_payload,
    format_worst_cells,
    load_config,
    read_records,
    require_committed_domains,
    run,
    shuffled,
    split_records,
    summarize_dataset,
    worst_cell,
)
from training.generate_data import DataConfig, generate


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    generate(DataConfig(seed=1, per_type=6, languages=("en", "pt")), path)
    return path


def _scorer_dict(value: float = 0.5) -> dict[str, object]:
    return {"rank": 1, "w1": [[value] * 4], "w2": [[value] * 2]}


def test_choice_domains_reads_choice_records_only() -> None:
    records = [
        {"type": "choice", "domain": "voice"},
        {"type": "choice", "domain": "support"},
        {"type": "noul", "domain": "telepathy"},  # other primitives do not create heads
        {"type": "choice"},  # legacy records carry no domain field
    ]
    assert choice_domains(records) == ("support", "voice")  # sorted, deterministic
    assert choice_domains([{"type": "choice"}]) == ()  # legacy single-domain data


def test_require_committed_domains_fails_fast_with_the_known_list() -> None:
    assert require_committed_domains(["support", "voice"]) == ("support", "voice")
    with pytest.raises(SystemExit, match=r"telepathy.*committed domains"):
        require_committed_domains(["support", "telepathy"])


def test_choice_head_payload_is_legacy_without_domains() -> None:
    """No domains → the payload IS the scorer dict: byte-identical to the old writer."""
    shared = _scorer_dict()
    assert choice_head_payload(shared, {}) == shared


def test_choice_head_payload_ships_head_and_signatures_per_domain() -> None:
    shared = _scorer_dict()
    payload = choice_head_payload(
        shared,
        {"voice": _scorer_dict(0.1), "support": _scorer_dict(0.2)},
        languages=("en",),
    )
    assert payload["shared"] == shared
    assert list(payload["domains"]) == ["support", "voice"]  # sorted, reproducible
    voice = payload["domains"]["voice"]
    assert voice["head"] == _scorer_dict(0.1)
    assert "thermostat" in voice["signatures"]  # committed vocabulary, not learned
    assert len(voice["signatures"]) == len(set(voice["signatures"]))


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
    assert report["choice_domains"] == []  # legacy data: shared head only


def test_dry_run_validates_domain_keys_early(tmp_path: Path) -> None:
    """The cheap validation path also checks the bank keys (L-011: fail before training)."""
    data = tmp_path / "domains.jsonl"
    generate(
        DataConfig(seed=1, per_type=3, languages=("en",), domains=("support", "voice")),
        data,
    )
    report = run(FinetuneConfig(data_path=str(data)), dry_run=True)
    assert report["choice_domains"] == ["support", "voice"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
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
        run(FinetuneConfig(data_path=str(bad)), dry_run=True)


def test_train_without_extra_raises(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the training path is exercised elsewhere")
    config = FinetuneConfig(data_path=str(_dataset(tmp_path)), out_dir=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="train extra"):
        run(config, dry_run=False)


def test_split_records_is_seeded_disjoint_and_valid() -> None:
    records = [{"id": index} for index in range(100)]
    train, val = split_records(records, val_fraction=0.2, seed=7)
    assert (train, val) == split_records(records, val_fraction=0.2, seed=7)
    assert len(train) == 80 and len(val) == 20
    assert {record["id"] for record in train} | {record["id"] for record in val} == set(range(100))
    assert not {record["id"] for record in train} & {record["id"] for record in val}
    with pytest.raises(ValueError, match="val_fraction"):
        split_records(records, val_fraction=1.5, seed=1)
    empty_train, empty_val = split_records([], val_fraction=0.1, seed=1)
    assert empty_train == [] and empty_val == []


def test_split_records_stops_one_domain_owning_the_validation_set(tmp_path: Path) -> None:
    """The old tail split put the *last* domain alone in val (B-5); the shuffle does not."""
    path = tmp_path / "d.jsonl"
    generate(DataConfig(seed=1, per_type=50, languages=("en",), domains=("support", "voice")), path)
    records = list(read_records(path))
    tail = records[int(len(records) * 0.9) :]
    assert {record["domain"] for record in tail} == {"voice"}  # what the old split produced

    train, val = split_records(records, val_fraction=0.1, seed=3)
    assert {record["domain"] for record in train} == {"support", "voice"}
    assert {record["domain"] for record in val} == {"support", "voice"}
    for kind in ("noul", "choice", "score"):
        assert {record["type"] for record in val if record["domain"] == "voice"} >= {kind}


def test_shuffled_is_deterministic_a_permutation_and_pure() -> None:
    items = list(range(50))
    first = shuffled(items, seed="epoch:0:noul")
    assert first == shuffled(items, seed="epoch:0:noul")
    assert first != shuffled(items, seed="epoch:1:noul")
    assert sorted(first) == items
    assert items == list(range(50))  # the input is untouched
    assert shuffled([], seed=1) == []


def test_cell_of_is_primitive_domain_lang_with_a_domainless_fallback() -> None:
    assert cell_of({"type": "noul", "domain": "support", "lang": "it"}) == "noul:support/it"
    # legacy single-domain records carry no domain: one cell per primitive/language, not a crash
    assert cell_of({"type": "score", "lang": "en"}) == "score:-/en"
    assert cell_of({"type": "choice", "domain": "", "lang": "pt"}) == "choice:-/pt"


def test_bucket_by_cell_groups_and_sorts_keys() -> None:
    records = [
        {"type": "noul", "domain": "voice", "lang": "it"},
        {"type": "noul", "domain": "support", "lang": "it"},
        {"type": "noul", "domain": "voice", "lang": "it"},  # same cell as the first row
    ]
    buckets = bucket_by_cell(records)
    assert list(buckets) == ["noul:support/it", "noul:voice/it"]  # sorted, stable output
    assert len(buckets["noul:voice/it"]) == 2
    assert buckets["noul:support/it"] == [records[1]]  # rows keep their identity
    assert bucket_by_cell([]) == {}


def test_worst_cell_picks_the_highest_loss_of_that_primitive_only() -> None:
    cells = {
        "noul:support/it": 0.3612,
        "noul:support/de": 0.0010,
        "choice:support/it": 0.9000,  # another primitive must not hijack the noul pick
        "score:voice/pt": 0.0100,
    }
    assert worst_cell(cells, "noul") == ("noul:support/it", 0.3612)
    assert worst_cell(cells, "choice") == ("choice:support/it", 0.9000)
    assert worst_cell(cells, "score") == ("score:voice/pt", 0.0100)
    assert worst_cell(cells, "missing") is None
    # deterministic tie-break on the cell name, not on dict order
    tie = {"noul:a/x": 0.5, "noul:b/y": 0.5}
    assert worst_cell(tie, "noul") == ("noul:b/y", 0.5)
    assert worst_cell({}, "noul") is None


def test_format_worst_cells_is_one_line_with_the_primitive_prefix_stripped() -> None:
    cells = {
        "noul:support/it": 0.3612,
        "noul:voice/de": 0.0010,
        "choice:ecommerce/fr": 0.0042,
    }
    assert format_worst_cells(cells) == "noul=support/it@0.3612 choice=ecommerce/fr@0.0042"
    assert format_worst_cells({}) == ""  # no rows at all: empty line, never a crash
