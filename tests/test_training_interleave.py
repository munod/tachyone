"""Tests for the interleaved continuation (the B-14 fix): batch order, resume plumbing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training.interleave_continue import (
    PRIMITIVE_ORDER,
    adapter_hyperparameters,
    interleaved_batches,
    resolve_model_id,
    resolve_setting,
    sha256,
)


def _grouped(**kinds: int) -> dict[str, list[dict[str, Any]]]:
    return {
        kind: [{"type": kind, "n": index} for index in range(count)]
        for kind, count in kinds.items()
    }


def test_interleaved_batches_cycles_primitives_in_fixed_order() -> None:
    grouped = _grouped(noul=6, choice=6, score=6)
    order = [kind for kind, _ in interleaved_batches(grouped, batch_size=2, seed="8")]
    assert order[:6] == list(PRIMITIVE_ORDER) * 2  # noul choice score noul choice score ...
    assert len(order) == 9  # two batches per primitive per cycle


def test_interleaved_batches_yields_every_record_exactly_once() -> None:
    grouped = _grouped(noul=7, choice=3, score=5)
    seen: list[dict[str, Any]] = []
    for kind, batch in interleaved_batches(grouped, batch_size=2, seed=8):
        assert all(record["type"] == kind for record in batch)
        seen.extend(batch)
    for kind in PRIMITIVE_ORDER:
        yielded = [record for record in seen if record["type"] == kind]
        # shuffled within the primitive, but a permutation: nothing lost, nothing duplicated
        assert sorted(yielded, key=lambda record: record["n"]) == grouped[kind]


def test_interleaved_batches_skips_exhausted_primitives() -> None:
    grouped = _grouped(noul=4, choice=1, score=4)
    order = [kind for kind, _ in interleaved_batches(grouped, batch_size=1, seed=1)]
    assert len(order) == 9  # nothing is lost when a primitive runs out early
    assert order[:3] == list(PRIMITIVE_ORDER)  # choice's single batch rides the first cycle
    assert order.count("choice") == 1
    # after choice is exhausted the cursor degrades to noul/score alternation, never a stall
    tail = order[3:]
    assert tail == ["noul", "score"] * 3


def test_interleaved_batches_is_seeded_and_total_on_empty_input() -> None:
    grouped = _grouped(noul=10, choice=10, score=10)
    first = list(interleaved_batches(grouped, batch_size=3, seed="ep:8"))
    assert first == list(interleaved_batches(grouped, batch_size=3, seed="ep:8"))
    assert first != list(interleaved_batches(grouped, batch_size=3, seed="ep:9"))
    assert list(interleaved_batches({}, batch_size=4, seed=1)) == []
    assert list(interleaved_batches(_grouped(noul=0), batch_size=4, seed=1)) == []


def test_adapter_hyperparameters_reads_only_the_recipe_keys(tmp_path: Path) -> None:
    adapter = tmp_path / "trunk"
    adapter.mkdir()
    assert adapter_hyperparameters(adapter) == {}  # no finetune_config.json is fine
    (adapter / "finetune_config.json").write_text(
        json.dumps({"model_id": "jhu-clsp/mmBERT-base", "choice_rank": 64, "epochs": 8}),
        encoding="utf-8",
    )
    # `epochs` is not a continuation key: only the declared recipe keys come through
    assert adapter_hyperparameters(adapter) == {
        "model_id": "jhu-clsp/mmBERT-base",
        "choice_rank": 64,
    }


def test_resolve_model_id_precedence_never_infers_from_the_directory(tmp_path: Path) -> None:
    adapter = tmp_path / "tmp_il_orig"  # a name with no "multi": the substring trap
    adapter.mkdir()
    # PEFT's own config is the always-present fallback
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "jhu-clsp/mmBERT-base"}), encoding="utf-8"
    )
    assert resolve_model_id(adapter, None) == "jhu-clsp/mmBERT-base"
    (adapter / "finetune_config.json").write_text(
        json.dumps({"model_id": "trunk/recorded"}), encoding="utf-8"
    )
    assert resolve_model_id(adapter, None) == "trunk/recorded"  # recorded recipe wins
    assert resolve_model_id(adapter, "cli/model") == "cli/model"  # explicit CLI wins
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit, match="--model-id"):
        resolve_model_id(bare, None)  # no source at all: fail, never guess


def test_resolve_setting_is_cli_then_trunk_recipe_then_default() -> None:
    hyper = {"batch_size": 16, "seed": 7}
    assert resolve_setting(4, hyper, "batch_size") == 4  # explicit CLI
    assert resolve_setting(None, hyper, "batch_size") == 16  # trunk's recorded recipe
    assert resolve_setting(None, hyper, "max_len") == 1024  # built-in default


def test_sha256_matches_the_hash_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"tachyone")
    assert sha256(path) == hashlib.sha256(b"tachyone").hexdigest()
    other = tmp_path / "other.bin"
    other.write_bytes(b"differ")
    assert sha256(path) != sha256(other)
