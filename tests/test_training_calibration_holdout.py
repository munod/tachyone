"""Tests for the P3 calibration holdout builder (B-13, JB-9).

The builder's whole job is to enforce two rules — *the fit basis is the real training set*
and *everything else is never trained on and never a JevBench item* — plus reproducibility,
so that is what the tests hold it to. Nothing here generates records (that is the point of
the builder), needs a model, or needs a network; the real-data build runs on the training
box with the pinned sources present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.build_calibration_holdout import (
    DEFAULT_CONFIG,
    BuildError,
    build,
    load_config,
    slice_remainder,
    slice_training,
    trained_states,
)

_MNLI_ROWS = [
    {
        "premise": f"Preamble sentence number {i} of the fixture.",
        "hypothesis": "Hypothesis text.",
        "label": i % 3,
    }
    for i in range(6)
]
_BOOLQ_ROWS = [
    {"question": f"Question {i}?", "passage": f"Passage number {i}.", "answer": bool(i % 2)}
    for i in range(6)
]


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _training_file(directory: Path) -> Path:
    path = directory / "train_en_domains.jsonl"
    rows = [
        {
            "id": f"{kind}-{index:03d}",
            "type": kind,
            "state": f"{kind} state {index}",
            "instructions": f"is it {kind} case {index}?",
        }
        for kind in ("noul", "choice", "score")
        for index in range(10)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(directory: Path) -> Path:
    return _write(
        directory / "holdout.json",
        {
            "_comment": "fixture",
            "training": {
                "path": str(_training_file(directory)),
                "counts": {"noul": 4, "choice": 4, "score": 4},
            },
            "remainder": {"prefix": "rem", "counts": {"multi_nli": 3, "boolq": 3}},
        },
    )


def _sources(directory: Path) -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    pq.write_table(pyarrow.Table.from_pylist(_MNLI_ROWS), sources / "multi_nli.parquet")
    pq.write_table(pyarrow.Table.from_pylist(_BOOLQ_ROWS), sources / "boolq.parquet")
    return sources


def _public_dir(directory: Path) -> Path:
    """An empty public-items directory (the guard still requires a dataset file to exist)."""
    public = directory / "public"
    public.mkdir(parents=True, exist_ok=True)
    (public / "easy.jsonl").write_text("", encoding="utf-8")
    return public


def _trained(directory: Path) -> list[Path]:
    """One training record whose state *is* the first MultiNLI row — it must be dropped."""
    from training.build_jev_sources import mnli_record

    record = mnli_record(_MNLI_ROWS[0]["premise"], _MNLI_ROWS[0]["hypothesis"], 0, 0)
    path = directory / "other_trained.jsonl"
    path.write_text(json.dumps({"state": record["state"]}) + "\n", encoding="utf-8")
    return [path]


# ------------------------------------------------------------------------ the config


def test_committed_config_uses_the_real_training_set_as_basis() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert set(config) - {"_comment"} == {"training", "remainder"}
    assert config["training"]["path"] == "data/train_en_domains.jsonl"
    assert set(config["training"]["counts"]) == {"noul", "choice", "score"}
    assert set(config["remainder"]["counts"]) == {"multi_nli", "boolq"}
    assert "families" not in config and "truncation" not in config, "no generated slices"


def test_config_rejects_unknown_and_missing_sections(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", {"training": {}, "extra": {}})
    with pytest.raises(BuildError, match="unknown calibration holdout"):
        load_config(tmp_path / "a.json")
    _write(tmp_path / "b.json", {"training": {}})
    with pytest.raises(BuildError, match="missing"):
        load_config(tmp_path / "b.json")
    _write(
        tmp_path / "c.json",
        {
            "training": {"path": "x", "counts": {}, "sneaky": 1},
            "remainder": {"counts": {}},
        },
    )
    with pytest.raises(BuildError, match="training"):
        load_config(tmp_path / "c.json")
    _write(
        tmp_path / "d.json", {"training": {"path": "x", "counts": {}}, "remainder": {"counts": {}}}
    )
    with pytest.raises(BuildError, match="counts"):
        load_config(tmp_path / "d.json")


# ------------------------------------------------------------------------ mechanics


def test_trained_states_unions_files_and_skips_missing(tmp_path: Path) -> None:
    present = tmp_path / "one.jsonl"
    present.write_text(
        json.dumps({"state": "State one"}) + "\n" + json.dumps({"state": "state  two"}) + "\n",
        encoding="utf-8",
    )
    states = trained_states([tmp_path / "absent.jsonl", present])
    assert states == {"stateone", "statetwo"}, "normalization is alphanumerics only"


def test_slice_training_strides_over_the_real_file_by_type(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    records = slice_training(config)
    assert len(records) == 12
    by_type = {
        kind: [r for r in records if r["type"] == kind] for kind in ("noul", "choice", "score")
    }
    assert all(len(rows) == 4 for rows in by_type.values())
    assert all(r["slice"] == "training" for r in records)
    assert all(r["id"].startswith("cal-trn-") for r in records)
    assert len({r["id"] for r in records}) == len(records)
    # the sampled rows are rows of the file, byte for byte
    source_states = {
        json.loads(line)["state"]
        for line in _training_file(tmp_path).read_text(encoding="utf-8").splitlines()
    }
    assert {r["state"] for r in records} <= source_states
    # stride, not head: the last row of each pool is reachable
    assert {r["state"] for r in by_type["noul"]} != {f"noul state {i}" for i in range(4)}


def test_slice_training_never_generates_and_validates_counts(tmp_path: Path) -> None:
    config = {
        "training": {"path": str(_training_file(tmp_path)), "counts": {"noul": 99}},
        "remainder": {"counts": {}},
    }
    with pytest.raises(BuildError, match="needs 99 noul rows"):
        slice_training(config)
    missing = {"training": {"path": str(tmp_path / "nope.jsonl"), "counts": {"noul": 1}}}
    with pytest.raises(BuildError, match="missing training file"):
        slice_training(missing)  # type: ignore[arg-type]


def test_slice_remainder_drops_trained_rows_and_rewrites_ids(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from training.build_jev_sources import mnli_record, normalized

    sources = _sources(tmp_path)
    trained = {normalized(json.loads(_trained(tmp_path)[0].read_text(encoding="utf-8"))["state"])}
    records, dropped = slice_remainder(
        {"prefix": "rem", "counts": {"multi_nli": 3, "boolq": 3}},
        data_dir=sources,
        trained=trained,
    )
    assert dropped == 1, "the trained row is counted, not silently re-sampled"
    assert len(records) == 6, "sampling continues past the dropped row to fill the count"
    first = mnli_record(_MNLI_ROWS[0]["premise"], _MNLI_ROWS[0]["hypothesis"], 0, 0)
    assert all(r["state"] != first["state"] for r in records)
    assert all(r["id"].startswith(("cal-rem-mnli", "cal-rem-boolq")) for r in records)
    assert {r["slice"] for r in records} == {"remainder"}


def test_slice_remainder_refuses_to_come_up_short(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    sources = _sources(tmp_path)
    with pytest.raises(BuildError, match="short"):
        slice_remainder(
            {"prefix": "rem", "counts": {"multi_nli": 99}},
            data_dir=sources,
            trained=set(),
        )


def test_slice_remainder_refuses_an_unknown_source(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    with pytest.raises(BuildError, match="unknown sources"):
        slice_remainder(
            {"prefix": "rem", "counts": {"snli": 3}},
            data_dir=_sources(tmp_path),
            trained=set(),
        )


# ----------------------------------------------------------------------- the build


def test_build_writes_both_slices_and_enforces_both_rules(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    config = _config(tmp_path)
    sources = _sources(tmp_path)
    trained = _trained(tmp_path)
    out = tmp_path / "out" / "calibration_holdout.jsonl"
    public = _public_dir(tmp_path)

    report = build(
        config_path=config,
        out_path=out,
        sources_dir=sources,
        trained_paths=trained,
        public_dir=public,
    )
    assert set(report["slices"]) == {"training", "remainder"}
    assert report["basis"] == str(_training_file(tmp_path))
    assert report["total"] == sum(entry["n"] for entry in report["slices"].values())

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == report["total"]
    assert {r["slice"] for r in records} == {"training", "remainder"}
    assert len({r["id"] for r in records}) == len(records), "ids are unique across slices"
    assert all(r["id"].startswith("cal-") for r in records)

    # rule 1: the remainder never trained — the MultiNLI row that is in `trained` is gone
    kept_remainder = [r for r in records if r["slice"] == "remainder"]
    assert report["dropped_seen_in_training"]["remainder"] == 1
    assert len(kept_remainder) == 6

    # rule 2: the training slice is literally rows of the basis file
    source_states = {
        json.loads(line)["state"]
        for line in _training_file(tmp_path).read_text(encoding="utf-8").splitlines()
    }
    assert {r["state"] for r in records if r["slice"] == "training"} <= source_states


def test_build_is_byte_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    config = _config(tmp_path)
    sources = _sources(tmp_path)
    trained = _trained(tmp_path)
    public = _public_dir(tmp_path)
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for out in (first, second):
        build(
            config_path=config,
            out_path=out,
            sources_dir=sources,
            trained_paths=trained,
            public_dir=public,
        )
    assert first.read_bytes() == second.read_bytes()


def test_build_refuses_a_record_that_collides_with_a_public_item(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    config = _config(tmp_path)
    sources = _sources(tmp_path)
    trained = _trained(tmp_path)
    out = tmp_path / "one.jsonl"
    public = _public_dir(tmp_path)
    build(
        config_path=config,
        out_path=out,
        sources_dir=sources,
        trained_paths=trained,
        public_dir=public,
    )
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    (public / "easy.jsonl").write_text(
        json.dumps({"state": record["state"], "question": {"instructions": "q"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BuildError, match="evaluation-only"):
        build(
            config_path=config,
            out_path=tmp_path / "two.jsonl",
            sources_dir=sources,
            trained_paths=trained,
            public_dir=public,
        )


def test_build_needs_training_states(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    with pytest.raises(BuildError, match="no training states"):
        build(
            config_path=_config(tmp_path),
            out_path=tmp_path / "out.jsonl",
            sources_dir=_sources(tmp_path),
            trained_paths=[tmp_path / "absent.jsonl"],
        )
