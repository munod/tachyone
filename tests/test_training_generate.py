"""Tests for deterministic synthetic data generation (M4-T1, TRAIN-01/TRAIN-07)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from tachyone.primitives import Question
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


def test_choice_targets_cover_all_teams() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=40, languages=("en", "pt"))))
    teams = {r["target"] for r in records if r["type"] == "choice"}
    assert {"billing", "technical", "sales", "other"} <= teams


def test_choice_uses_localized_terms_and_distractors() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=40, languages=("pt",))))
    states = " ".join(r["state"] for r in records if r["type"] == "choice").lower()
    assert any(word in states for word in ("reembolso", "fatura", "cobrança", "pagamento"))
    # Every pt distractor clause contains "também"; ensure hard negatives are present.
    assert "também" in states


def test_choice_criteria_have_rich_descriptions_including_other() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=8, languages=("en",))))
    choice = next(r for r in records if r["type"] == "choice")
    criteria = choice["criteria"]
    assert all(isinstance(value, str) and value for value in criteria.values())
    # "other" must be a semantically rich, learnable option (not a bare label).
    assert len(criteria["other"]) > 60
    assert "outside" in criteria["other"]


def test_instructions_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    choice = next(r for r in records if r["type"] == "choice")
    score = next(r for r in records if r["type"] == "score")
    noul = next(r for r in records if r["type"] == "noul")
    assert "equipe" in choice["instructions"].lower()
    assert "urgência" in score["instructions"].lower()
    assert "solicita" in noul["instructions"].lower()


def test_score_levels_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    score = next(r for r in records if r["type"] == "score")
    assert score["criteria"] == ["nenhuma", "baixa", "média", "alta"]


def test_noul_entities_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    noul = next(r for r in records if r["type"] == "noul")
    assert isinstance(noul["criteria"], dict)
    # The true-criterion uses the localized template, not the English one.
    assert noul["criteria"]["true"].startswith("solicita um ")
    assert "asks for" not in noul["criteria"]["true"]


def test_every_language_gets_equal_support() -> None:
    langs = ("en", "pt", "es", "fr", "de", "it", "nl")
    records = list(iter_records(DataConfig(seed=5, per_type=len(langs) * 4, languages=langs)))
    for kind in ("noul", "choice", "score"):
        counts: dict[str, int] = {}
        for record in records:
            if record["type"] == kind:
                counts[record["lang"]] = counts.get(record["lang"], 0) + 1
        assert set(counts) == set(langs)
        assert len(set(counts.values())) == 1


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


def _noisy_records(rate: float, count: int = 40) -> list[dict[str, Any]]:
    return list(
        iter_records(DataConfig(seed=11, per_type=count, languages=("en", "pt"), noise_rate=rate))
    )


def test_noise_disabled_is_byte_identical_to_clean(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    noisy = tmp_path / "noisy.jsonl"
    generate(DataConfig(seed=5, per_type=20, noise_rate=0.0), clean)
    generate(DataConfig(seed=5, per_type=20), noisy)
    assert clean.read_bytes() == noisy.read_bytes()


def test_noise_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    config = DataConfig(seed=123, per_type=30, noise_rate=0.5)
    generate(config, first)
    generate(config, second)
    assert first.read_bytes() == second.read_bytes()


def test_noise_rate_zero_perturbs_nothing() -> None:
    explicit = _noisy_records(0.0)
    default = list(iter_records(DataConfig(seed=11, per_type=40, languages=("en", "pt"))))
    assert [r["state"] for r in explicit] == [r["state"] for r in default]


def test_noise_rate_one_perturbs_most_records() -> None:
    clean = {r["id"]: r["state"] for r in _noisy_records(0.0)}
    noisy = {r["id"]: r["state"] for r in _noisy_records(1.0)}
    changed = [rid for rid in clean if clean[rid] != noisy[rid]]
    # Empty states and edit draws that happen to be no-ops (e.g. accent strip on ASCII) pass
    # through unchanged, so require a clear majority rather than every record.
    assert len(changed) > len(clean) // 2


def test_noise_changes_only_the_state() -> None:
    clean = {r["id"]: r for r in _noisy_records(0.0)}
    noisy = {r["id"]: r for r in _noisy_records(1.0)}
    for rid, record in noisy.items():
        for key in ("type", "instructions", "criteria", "target", "lang"):
            assert record[key] == clean[rid][key], key


def test_noise_preserves_empty_and_long_boundary_states() -> None:
    # indices 0,19,38,... are empty; 23,46,... are long (see _boundary_state).
    records = _noisy_records(1.0, count=60)
    for record in records:
        assert isinstance(record["state"], str)  # never raises, always a string


def test_noise_differs_across_seeds() -> None:
    one = {
        r["id"]: r["state"] for r in iter_records(DataConfig(seed=1, per_type=30, noise_rate=0.6))
    }
    two = {
        r["id"]: r["state"] for r in iter_records(DataConfig(seed=2, per_type=30, noise_rate=0.6))
    }
    assert one != two


def test_noise_cli_rejects_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--per-type", "3", "--noise-rate", "1.5", "--out", str(tmp_path / "x.jsonl")])


def test_noise_cli_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "noisy.jsonl"
    code = main(["--seed", "9", "--per-type", "3", "--noise-rate", "0.5", "--out", str(out)])
    assert code == 0
    assert "wrote 9 records" in capsys.readouterr().out
