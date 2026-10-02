"""Tests for the B-13 P2 training mixture (JB-5).

The mixture is the treatment arm's single ``data_path``; these tests pin the recipe's
shape, the refusal paths (missing part, malformed record, duplicate ids across parts)
and the exact concatenation order the trainer will see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.build_jev_mixture import (
    DEFAULT_CONFIG,
    BuildError,
    build,
    load_config,
    read_part,
)

PARTS = (
    "data/train_en_domains.jsonl",
    "data/train_jev_sources.jsonl",
    "data/train_jev_families.jsonl",
)


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _record(record_id: str, **overrides) -> dict:
    record = {
        "id": record_id,
        "type": "noul",
        "state": "s",
        "instructions": "i",
        "target": 1,
        "lang": "en",
    }
    record.update(overrides)
    return record


def _config(tmp_path: Path, parts: list[str], out: str) -> Path:
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps({"parts": parts, "out": out}), encoding="utf-8")
    return path


def test_committed_config_names_the_three_parts_in_order() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["parts"] == list(PARTS)
    assert config["out"] == "data/train_jev_en.jsonl"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps({"parts": ["x"], "out": "y", "shuffle": True}), encoding="utf-8")
    with pytest.raises(BuildError, match="unknown config keys"):
        load_config(path)


def test_missing_part_fails_loudly(tmp_path: Path) -> None:
    config = _config(tmp_path, [str(tmp_path / "absent.jsonl")], str(tmp_path / "out.jsonl"))
    with pytest.raises(BuildError, match="missing mixture part"):
        build(config, out_path=tmp_path / "out.jsonl")


def test_malformed_record_is_refused_with_its_line(tmp_path: Path) -> None:
    part = tmp_path / "part.jsonl"
    part.write_text(json.dumps({"id": "a", "type": "noul"}) + "\n", encoding="utf-8")
    with pytest.raises(BuildError, match="misses"):
        read_part(part)


def test_duplicate_ids_across_parts_are_refused(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(first, [_record("same-id")])
    _write(second, [_record("same-id")])
    config = _config(tmp_path, [str(first), str(second)], str(tmp_path / "out.jsonl"))
    with pytest.raises(BuildError, match="duplicate record id"):
        build(config, out_path=tmp_path / "out.jsonl")


def test_ids_may_repeat_inside_one_part_but_not_across_parts(tmp_path: Path) -> None:
    """The incumbent domain datasets use per-(type, domain) id counters — that's legal."""
    part = tmp_path / "domains.jsonl"
    _write(part, [_record("noul-000000"), _record("noul-000000", type="choice", target="x")])
    other = tmp_path / "families.jsonl"
    _write(other, [_record("jev-trap-00000")])
    config = _config(tmp_path, [str(part), str(other)], str(tmp_path / "out.jsonl"))
    report = build(config, out_path=tmp_path / "out.jsonl")
    assert report["total"] == 3
    assert report["unique_ids"] == 2


def test_build_concatenates_parts_in_order(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(first, [_record("a1", source="one"), _record("a2", source="one")])
    _write(second, [_record("b1", source="two", type="choice", target="x", criteria={})])
    config = _config(tmp_path, [str(first), str(second)], str(tmp_path / "out.jsonl"))
    report = build(config, out_path=tmp_path / "out.jsonl")

    assert report["total"] == 3
    assert report["unique_ids"] == 3
    assert report["per_part"] == {str(first): 2, str(second): 1}
    assert report["per_type"] == {"choice": 1, "noul": 2}
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a1", "a2", "b1"]


@pytest.mark.skipif(
    not all((Path.cwd() / part).exists() for part in PARTS),
    reason="mixture parts not built on this machine",
)
def test_real_mixture_covers_every_part_with_unique_ids(tmp_path: Path) -> None:
    out = tmp_path / "train_jev_en.jsonl"
    report = build(DEFAULT_CONFIG, out_path=out)
    expected = sum(len(read_part(Path.cwd() / part)) for part in PARTS)
    assert report["total"] == expected
    assert set(report["per_part"]) == set(PARTS)
    # ids repeat inside the incumbent domain part (per-type counters) but must stay unique
    # across parts — the build succeeding is the cross-part check.
    assert report["unique_ids"] <= report["total"]
    # the treatment arm must carry all three sources: domains + sources + families
    assert {"synthetic", "multi_nli", "long_policy"} <= set(report["per_source"])
