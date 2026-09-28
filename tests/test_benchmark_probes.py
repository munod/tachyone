"""Unit tests for the public-probe loaders (B-7, ``benchmarks/probes.py``).

One test per probe covers the thing that is easy to get silently wrong — **the label mapping**
(gold label → ``answer_index``) — plus the failure messages that tell the reader how to fetch a
dataset that is not on disk. The datasets themselves are never committed: fixtures are built in
``tmp_path``, and the parquet-based probes need ``pyarrow`` (an opt-in benchmark dependency).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.compare import PROBE_VAL_SPLITS, _probe_split, main
from benchmarks.probes import (
    MASSIVE_INTENTS,
    MASSIVE_LANGUAGES,
    PROBES,
    XNLI_LABELS,
    load_massive,
    load_probe,
    load_typed_decisions,
    load_xnli,
    render,
)


# ------------------------------------------------------------------ typed-decisions
def _typed_decisions_record() -> dict[str, Any]:
    questions = {
        "needs_review": {
            "type": "noul",
            "instructions": "This run requires human review.",
            "criteria": {"false": "No attention needed.", "true": "A human must look."},
        },
        "action": {
            "type": "choice",
            "instructions": "What now?",
            "criteria": {"continue": "proceed", "stop": "halt"},
        },
        "risk": {
            "type": "score",
            "instructions": "How risky?",
            "criteria": ["benign", "low", "high"],
        },
    }
    gold = {
        "needs_review": {"type": "noul", "label": "true", "noul": 0.7},
        "action": {"type": "choice", "label": "stop"},
        "risk": {"type": "score", "label": "2"},
    }
    return {
        "id": "security_incidents_000001",
        "workflow": "security_incidents",
        "split": "test",
        "state": '{"task": "rotate the certificate"}',
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
        "n_questions": 3,
    }


def _write_typed_decisions(path: Path, split: str, records: list[dict[str, Any]]) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    directory = path / "all"
    directory.mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        pyarrow.Table.from_pylist(records), directory / f"{split}-00000-of-00001.parquet"
    )


def test_typed_decisions_maps_gold_labels_onto_answer_index(tmp_path: Path) -> None:
    _write_typed_decisions(tmp_path, "test", [_typed_decisions_record()])

    rows = load_typed_decisions(tmp_path, split="test", per_task=10)

    assert len(rows) == 3
    by_kind = {row["wire"]["type"]: row for row in rows}
    assert by_kind["noul"]["answer_index"] == 1  # gold label "true"
    assert by_kind["noul"]["options"] == ["No attention needed.", "A human must look."]
    assert by_kind["choice"]["answer_index"] == 1  # "stop" is the second criterion
    assert by_kind["score"]["answer_index"] == 2  # gold label "2" is the third level
    # the native question rides along, so Tachyone answers it in its own shape
    assert by_kind["noul"]["wire"]["criteria"]["false"] == "No attention needed."
    assert {row["task"] for row in rows} == {"security_incidents"}


def test_typed_decisions_caps_per_config_and_needs_a_download(tmp_path: Path) -> None:
    _write_typed_decisions(
        tmp_path, "test", [_typed_decisions_record(), {**_typed_decisions_record(), "id": "b"}]
    )
    assert len(load_typed_decisions(tmp_path, split="test", per_task=2)) == 2

    with pytest.raises(SystemExit, match="snapshot_download"):
        load_typed_decisions(tmp_path / "nowhere", split="test", per_task=2)


def test_typed_decisions_rejects_a_label_that_is_not_in_the_options(tmp_path: Path) -> None:
    record = _typed_decisions_record()
    gold = json.loads(record["gold"])
    gold["action"]["label"] = "invented"
    record["gold"] = json.dumps(gold)
    _write_typed_decisions(tmp_path, "test", [record])

    with pytest.raises(SystemExit, match="not in its options"):
        load_typed_decisions(tmp_path, split="test", per_task=10)


def test_typed_decisions_noul_without_criteria_uses_bare_true_false(tmp_path: Path) -> None:
    """Two dataset questions ship without `criteria`; they are plain true/false, not an error."""
    record = _typed_decisions_record()
    questions = json.loads(record["questions"])
    questions["needs_review"] = {"type": "noul", "instructions": "This invoice is a duplicate."}
    gold = json.loads(record["gold"])
    gold["needs_review"] = {"type": "noul", "label": "false", "noul": 0.04}
    record["questions"] = json.dumps(questions)
    record["gold"] = json.dumps(gold)
    _write_typed_decisions(tmp_path, "test", [record])

    rows = load_typed_decisions(tmp_path, split="test", per_task=10)
    noul = next(row for row in rows if row["wire"]["type"] == "noul")

    assert noul["options"] == ["false", "true"]
    assert noul["answer_index"] == 0
    assert noul["wire"]["criteria"] is None  # still a valid `/v1/systemone` question


# ------------------------------------------------------------------------- massive
def _write_massive(path: Path, locale: str, records: list[dict[str, Any]]) -> None:
    directory = path / "data"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{locale}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def _massive_record(intent: str, partition: str, utt: str) -> dict[str, Any]:
    return {"id": "0", "locale": "en-US", "partition": partition, "intent": intent, "utt": utt}


def test_massive_maps_intent_onto_the_fixed_60_label_option_set(tmp_path: Path) -> None:
    _write_massive(
        tmp_path,
        "en-US",
        [
            _massive_record("alarm_set", "test", "wake me up at five"),
            _massive_record("weather_query", "test", "will it rain tomorrow"),
            _massive_record("alarm_set", "dev", "ignored dev row"),
        ],
    )

    rows = load_massive(tmp_path, split="test", per_task=10, languages=("en",))

    assert len(rows) == 2  # the dev row is not part of the reported split
    assert len(rows[0]["options"]) == len(MASSIVE_INTENTS) == 60
    assert rows[0]["options"][rows[0]["answer_index"]] == "alarm_set"
    assert rows[1]["options"][rows[1]["answer_index"]] == "weather_query"
    assert rows[0]["state"] == "wake me up at five"
    assert rows[0]["task"] == "en"


def test_massive_caps_per_language_and_refuses_a_partial_run(tmp_path: Path) -> None:
    _write_massive(
        tmp_path, "en-US", [_massive_record("alarm_set", "test", f"u{i}") for i in range(3)]
    )
    # es is not extracted yet: the loader refuses instead of silently evaluating one language
    with pytest.raises(SystemExit, match="not extracted"):
        load_massive(tmp_path, split="test", per_task=2, languages=("en", "es"))

    _write_massive(tmp_path, "es-ES", [_massive_record("alarm_set", "test", "despiértame")])
    rows = load_massive(tmp_path, split="test", per_task=2, languages=("en", "es"))
    assert [row["task"] for row in rows] == ["en", "en", "es"]
    assert {row["answer_index"] for row in rows} == {MASSIVE_INTENTS.index("alarm_set")}


def test_massive_names_the_command_when_the_tarball_is_not_extracted(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="tar -xzf"):
        load_massive(tmp_path, split="test", per_task=1, languages=("pt",))

    with pytest.raises(SystemExit, match="no locale for"):
        load_massive(tmp_path, split="test", per_task=1, languages=("ja",))


def test_massive_default_languages_are_the_trained_seven() -> None:
    assert MASSIVE_LANGUAGES == ("en", "pt", "es", "fr", "de", "it", "nl")
    assert len(MASSIVE_INTENTS) == 60  # the dataset's fixed vocabulary


# --------------------------------------------------------------------------- xnli
def _write_xnli(path: Path, language: str, split: str, records: list[dict[str, Any]]) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    directory = path / language
    directory.mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        pyarrow.Table.from_pylist(records), directory / f"{split}-00000-of-00001.parquet"
    )


def test_xnli_maps_the_label_onto_the_documented_option_order(tmp_path: Path) -> None:
    _write_xnli(
        tmp_path,
        "en",
        "test",
        [
            {"premise": "A man plays a guitar.", "hypothesis": "Someone plays music.", "label": 0},
            {"premise": "It is raining.", "hypothesis": "The street is dry.", "label": 2},
        ],
    )

    rows = load_xnli(tmp_path, split="test", per_task=10)

    assert len(rows) == 2
    assert list(XNLI_LABELS) == ["entailment", "neutral", "contradiction"]
    assert rows[0]["answer_index"] == 0 and rows[1]["answer_index"] == 2
    assert rows[0]["state"] == "Premise: A man plays a guitar.\nHypothesis: Someone plays music."
    assert rows[0]["question"] == "Does the hypothesis follow from the premise?"
    assert rows[0]["task"] == "en"


def test_xnli_needs_a_download_for_a_missing_language(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="snapshot_download"):
        load_xnli(tmp_path, split="test", per_task=10, languages=("en",))


# ---------------------------------------------------------------------- dispatcher
def test_load_probe_rejects_an_unknown_probe() -> None:
    with pytest.raises(SystemExit, match="typed-decisions, massive, xnli"):
        load_probe("made-up", Path("."), split="test", per_task=1)


def test_cli_rejects_an_unknown_probe_and_defaults_are_per_dataset() -> None:
    assert set(PROBE_VAL_SPLITS) == {"system-one-decisions", *PROBES}
    with pytest.raises(SystemExit):
        main(["run", "--engine", "tachyone", "--probe", "made-up", "--out", "x.json"])

    args = argparse.Namespace(split=None, val_split=None, probe="massive", languages=None)
    assert _probe_split(args, validation=False) == "test"
    assert _probe_split(args, validation=True) == "dev"

    args.probe = "system-one-decisions"
    assert _probe_split(args, validation=True) == "val"

    args.split, args.val_split = "holdout", "tune"
    assert _probe_split(args, validation=False) == "holdout"
    assert _probe_split(args, validation=True) == "tune"


# -------------------------------------------------------------------------- render
def _artifact(engine: str, probe: str, rows: int) -> dict[str, Any]:
    summary = {
        "n": rows,
        "answered": rows,
        "accuracy": 0.5,
        "ece": 0.1,
        "brier": 0.4,
        "mean_confidence": 0.6,
    }
    return {
        "engine": engine,
        "dataset": {
            "kind": "probe",
            "probe": probe,
            "repo": None,
            "path": "data/x",
            "split": "test",
            "val_split": "dev",
            "per_task": 64,
            "limit": 0,
            "languages": None,
            "rows": rows,
            "warmup_excluded": 3,
        },
        "temperature": {"fitted": True, "value": 1.5, "source": "validation split (capped)"},
        "raw": summary,
        "calibrated": summary,
        "per_task_raw": {},
        "per_task_calibrated": {"en": summary},
        "latency_ms": {"n": rows, "p50": 22.0, "p95": 40.0, "mean": 25.0},
        "throughput": {"items_per_s": 45.0, "note": "sequential"},
        "compliance": {"answered": rows, "failed": 0, "rate": 1.0},
        "memory": {"peak_rss_mb": 100.0, "peak_vram_mb": 50.0, "measured_on": "this process"},
        "engine_config": {},
        "environment": {"gpu": "NVIDIA GeForce RTX 3060", "python": "3.12.13"},
    }


def test_render_composes_probes_md_with_licences_and_reproduce_commands(tmp_path: Path) -> None:
    artifact = tmp_path / "probe_xnli.json"
    artifact.write_text(json.dumps(_artifact("tachyone (encoder)", "xnli", 5010)), encoding="utf-8")
    synthetic = tmp_path / "en.json"
    # Deliberately not the published figure: the prose must track the artifact (B-11 republished
    # every number as a set, so a literal would outlive the data behind it).
    synthetic.write_text(
        json.dumps({"overall": {"accuracy": 0.912, "ece": 0.019, "n": 1500}}), encoding="utf-8"
    )
    out = tmp_path / "probes.md"

    render([artifact], out, synthetic)

    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Public probes")
    assert "## `xnli` — facebook/xnli (CC BY-NC 4.0)" in text
    assert "facebookresearch/XNLI" in text  # where the licence was actually verified
    assert "| in-sample synthetic eval" in text and "| 1500 | 0.912 | 0.019 |" in text
    assert "against the synthetic 0.912" in text  # prose reads the artifact too
    assert "benchmarks.compare run --engine tachyone" in text
    assert "#### Per task: accuracy and calibration" in text
    assert "**Chance level:** 0.333 (3 labels)" in text


def test_render_refuses_an_artifact_that_names_no_probe(tmp_path: Path) -> None:
    artifact = _artifact("tachyone (encoder)", "probe", 10)
    artifact["dataset"]["probe"] = None
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SystemExit, match="re-run it with --probe"):
        render([path], tmp_path / "probes.md", None)
