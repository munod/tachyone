"""Tests for deterministic synthetic data generation (M4-T1, TRAIN-01/TRAIN-07)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from jeba.primitives import Question
from training.generate_data import DataConfig, generate, iter_records, main

_QUESTION = TypeAdapter(Question)


def _records(count: int = 12) -> list[dict[str, Any]]:
    return list(iter_records(DataConfig(seed=7, per_type=count, languages=("en", "pt"))))


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    config = DataConfig(seed=123, per_type=30)
    generate(config, first)
    generate(config, second)
    assert first.read_bytes() == second.read_bytes()


def test_different_seeds_differ(tmp_path: Path) -> None:
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    generate(DataConfig(seed=1, per_type=20), one)
    generate(DataConfig(seed=2, per_type=20), two)
    assert one.read_bytes() != two.read_bytes()


def test_counts_and_ids_are_unique() -> None:
    records = _records(5)
    assert len(records) == 15
    ids = [record["id"] for record in records]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "noul-000000"


def test_records_validate_as_primitives() -> None:
    for record in _records(6):
        assert record["type"] in {"noul", "choice", "score"}
        _QUESTION.validate_python(
            {
                "type": record["type"],
                "instructions": record["instructions"],
                "criteria": record["criteria"],
            }
        )
        if record["type"] == "noul":
            assert record["target"] in (0, 1)
        elif record["type"] == "choice":
            assert record["target"] in record["criteria"]
        else:
            assert isinstance(record["target"], int)
            assert 0 <= record["target"] < len(record["criteria"])


def test_languages_are_interleaved() -> None:
    records = _records(4)
    langs = {record["lang"] for record in records}
    assert langs <= {"en", "pt"}


def test_boundary_cases_present() -> None:
    records = _records(40)
    states = [record["state"] for record in records if record["type"] == "noul"]
    assert "" in states  # empty input injected at specific indices
    assert any(len(state) > 500 for state in states)  # very long input


def test_streaming_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "data.jsonl"
    written = generate(DataConfig(seed=3, per_type=5), out)
    assert written == 15
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert json.loads(lines[0])["id"] == "noul-000000"


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.jsonl"
    code = main(["--seed", "9", "--per-type", "3", "--out", str(out)])
    assert code == 0
    assert out.exists()
    assert "wrote 9 records" in capsys.readouterr().out
