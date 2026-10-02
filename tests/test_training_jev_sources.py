"""Tests for the pinned JevBench-family source fetch (B-13 P2, JB-1) and the record
builder that converts those sources into Tachyone records (JB-2).

Everything runs offline: the lock is validated structurally, downloads are exercised
against ``file://`` URLs, the fetch/check/record flow runs on fake local files, and the
builder's pure functions are tested on in-code rows. The end-to-end build runs only when
the pinned parquet files are present (skipped in CI, exercised on the training box).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from benchmarks.probes import XNLI_INSTRUCTION, XNLI_LABELS
from tachyone.primitives import ChoiceQuestion, NoulQuestion
from training.build_jev_sources import (
    BOOLQ_CRITERIA,
    DEFAULT_DATA_DIR,
    DEFAULT_DESCRIPTIONS,
    BuildError,
    assert_no_public_overlap,
    boolq_record,
    build,
    intent_group,
    load_config,
    load_descriptions,
    mnli_pairs,
    mnli_record,
    normalized,
    reservoir,
    sample_distractors,
)
from training.build_jev_sources import (
    DEFAULT_CONFIG as BUILDER_CONFIG,
)
from training.fetch_jev_sources import (
    DEFAULT_LOCK,
    LockError,
    download,
    fetch,
    load_lock,
    main,
    resolve_url,
    sha256_of,
)

PAYLOAD = b"pretend-parquet-bytes" * 64
_PINNED_PARQUET = DEFAULT_DATA_DIR / "multi_nli.parquet"
needs_sources = pytest.mark.skipif(
    not _PINNED_PARQUET.exists(), reason="pinned sources not fetched on this machine"
)


def _real_lock() -> dict[str, Any]:
    return json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))


def _fake_source(**overrides: Any) -> dict[str, Any]:
    source = {
        "repo": "acme/data",
        "revision": "0" * 40,
        "file": "data/train.parquet",
        "url": "https://huggingface.co/datasets/acme/data",
        "license": ["cc-by-4.0"],
        "sha256": "",
        "purpose": "testing",
    }
    source.update(overrides)
    return source


def test_committed_lock_names_the_three_reviewed_sources() -> None:
    lock = load_lock(DEFAULT_LOCK)
    assert set(lock["sources"]) == {"multi_nli", "boolq", "banking77"}
    for name, source in lock["sources"].items():
        assert len(source["revision"]) == 40, name
        assert source["url"] == f"https://huggingface.co/datasets/{source['repo']}", name
        assert source["purpose"], name
        assert source["license"], name


def test_excluded_sources_record_the_exact_licence_words() -> None:
    """AG News and SST-5 stay out until reviewed — tev1's DATA_SOURCES.md says the same."""
    lock = load_lock(DEFAULT_LOCK)
    assert lock["excluded"]["ag_news"]["license"] == "unknown"
    assert lock["excluded"]["sst5"]["license"] == "unspecified"
    for entry in lock["excluded"].values():
        assert entry["reason"]


def test_resolve_url_is_the_pinned_resolve_layout() -> None:
    source = load_lock(DEFAULT_LOCK)["sources"]["boolq"]
    assert resolve_url(source) == (
        "https://huggingface.co/datasets/google/boolq"
        f"/resolve/{source['revision']}/{source['file']}"
    )


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("no sources", lambda lock: lock.pop("sources")),
        ("short revision", lambda lock: lock["sources"]["boolq"].update(revision="abc")),
        ("bad sha", lambda lock: lock["sources"]["boolq"].update(sha256="XYZ")),
        ("no purpose", lambda lock: lock["sources"]["boolq"].pop("purpose")),
        ("no licence", lambda lock: lock["sources"]["boolq"].update(license=[])),
        ("not parquet", lambda lock: lock["sources"]["boolq"].update(file="data/x.csv")),
        ("excluded w/o reason", lambda lock: lock["excluded"]["ag_news"].pop("reason")),
        ("no excluded", lambda lock: lock.pop("excluded")),
    ],
)
def test_load_lock_rejects_broken_locks(tmp_path: Path, name: str, mutation: Any) -> None:
    lock = _real_lock()
    mutation(lock)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(LockError):
        load_lock(path)


def test_download_streams_to_the_destination(tmp_path: Path) -> None:
    source = tmp_path / "remote.bin"
    source.write_bytes(PAYLOAD)
    dest = tmp_path / "cache" / "one.parquet"
    download(source.as_uri(), dest)
    assert dest.read_bytes() == PAYLOAD
    assert sha256_of(dest) == hashlib.sha256(PAYLOAD).hexdigest()
    assert not dest.with_suffix(".parquet.part").exists()


def test_download_failure_leaves_no_part_file(tmp_path: Path) -> None:
    dest = tmp_path / "one.parquet"
    with pytest.raises(SystemExit, match="download failed"):
        download((tmp_path / "absent.bin").as_uri(), dest)
    assert not dest.exists()
    assert not dest.with_suffix(".parquet.part").exists()


def test_check_reports_missing_without_touching_the_network(tmp_path: Path) -> None:
    lock = load_lock(DEFAULT_LOCK)
    results, changed = fetch(lock, data_dir=tmp_path, check=True)
    assert results == dict.fromkeys(lock["sources"], "")
    assert changed is False


def test_record_then_verify_then_refuse_a_tampered_file(tmp_path: Path) -> None:
    lock = {"sources": {"one": _fake_source()}, "excluded": {"x": {"license": "?", "reason": "r"}}}
    (tmp_path / "one.parquet").write_bytes(PAYLOAD)

    results, changed = fetch(lock, data_dir=tmp_path, record=True)
    assert changed is True
    assert results["one"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert lock["sources"]["one"]["sha256"] == results["one"]

    _, changed = fetch(lock, data_dir=tmp_path)  # verify path
    assert changed is False

    (tmp_path / "one.parquet").write_bytes(b"different bytes")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch(lock, data_dir=tmp_path)


def test_main_check_exits_1_when_files_are_absent(tmp_path: Path) -> None:
    code = main(["--check", "--data-dir", str(tmp_path)])
    assert code == 1


def test_main_rejects_record_and_check_together(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(["--record", "--check", "--data-dir", str(tmp_path)])


# ---------------------------------------------------------------- JB-2: the builder


def test_config_is_strict_and_pins_every_sampled_source() -> None:
    config = load_config(BUILDER_CONFIG)
    assert isinstance(config["seed"], int)
    assert config["distractors_per_choice"] == 4
    lock = load_lock(DEFAULT_LOCK)
    assert set(config["counts"]) == set(lock["sources"]) == {"multi_nli", "boolq", "banking77"}
    for count in config["counts"].values():
        assert count > 0


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"seed": 1, "counts": {}, "distractors_per_choice": 1, "oops": 2}))
    with pytest.raises(BuildError, match="unknown config keys"):
        load_config(path)


def test_descriptions_cover_all_77_intents_in_label_order() -> None:
    """The key order **is** the label-index mapping of the pinned dataset."""
    names, descriptions = load_descriptions(DEFAULT_DESCRIPTIONS)
    assert len(names) == 77
    assert names[0] == "activate_my_card"
    assert names[11] == "card_arrival"  # the card the pinned README documents
    assert names[-1] == "wrong_exchange_rate_for_cash_withdrawal"
    # the two oddly-spelled class names must stay verbatim (traceability to the label index)
    assert "Refund_not_showing_up" in names
    assert "reverted_card_payment?" in names
    for name in names:
        assert descriptions[name].strip().endswith(".")


def test_reservoir_is_deterministic_capped_and_complete() -> None:
    items = list(range(1000))
    first = reservoir(items, 50, random.Random("s"))
    assert first == reservoir(items, 50, random.Random("s"))
    assert first != reservoir(items, 50, random.Random("other"))
    assert len(first) == 50 and len(set(first)) == 50
    assert sorted(reservoir(items, 2000, random.Random("s"))) == items  # k >= n keeps all


def test_mnli_pairs_unpacks_one_pair_per_row_and_tolerates_list_layouts() -> None:
    scalar = {"premise": "P one.", "hypothesis": "H one.", "label": 0}
    assert list(mnli_pairs(scalar)) == [("P one.", "H one.", 0)]
    listed = {
        "premise": ["P one.", "P two."],
        "hypothesis": ["H one.", "H two."],
        "label": [0, 2],
    }
    assert list(mnli_pairs(listed)) == [("P one.", "H one.", 0), ("P two.", "H two.", 2)]
    with pytest.raises(BuildError, match="ragged"):
        list(mnli_pairs({**listed, "label": [0]}))


def test_mnli_record_mirrors_the_xnli_probe_shape() -> None:
    """Train and probe must ask the same question — that is the transfer this buys."""
    record = mnli_record("A premise.", "A hypothesis.", 1, 7)
    assert record["state"] == "Premise: A premise.\nHypothesis: A hypothesis."
    assert record["instructions"] == XNLI_INSTRUCTION
    assert record["target"] == "neutral"
    assert list(record["criteria"]) == list(XNLI_LABELS)
    assert record["type"] == "choice" and record["lang"] == "en"
    ChoiceQuestion(instructions=record["instructions"], criteria=record["criteria"])  # wire-valid


def test_mnli_record_rejects_an_out_of_range_label() -> None:
    with pytest.raises(BuildError, match="outside"):
        mnli_record("p", "h", -1, 0)


def test_boolq_record_takes_the_target_from_the_answer() -> None:
    yes = boolq_record("is it so?", "The passage says so.", True, 3)
    no = boolq_record("is it so?", "The passage does not say.", False, 4)
    assert yes["target"] == 1 and no["target"] == 0
    assert yes["state"].startswith("Passage: ") and "\nQuestion: " in yes["state"]
    assert set(yes["criteria"]) == set(BOOLQ_CRITERIA)
    NoulQuestion(instructions=yes["instructions"], criteria=yes["criteria"])  # wire-valid
    with pytest.raises(BuildError, match="must be a bool"):
        boolq_record("q", "p", "true", 0)


def test_sample_distractors_prefers_siblings_and_never_the_target() -> None:
    names, _ = load_descriptions(DEFAULT_DESCRIPTIONS)
    rng = random.Random("distractors")
    for target in ("card_arrival", "transfer_fee_charged", "verify_my_identity"):
        picks = sample_distractors(target, names, 4, rng)
        assert len(picks) == len(set(picks)) == 4
        assert target not in picks
        siblings = sum(1 for pick in picks if intent_group(pick) == intent_group(target))
        assert siblings >= 2, (target, picks)


def test_normalized_overlap_check_refuses_collisions(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "original.jsonl").write_text(
        json.dumps(
            {
                "state": "Where is my package?",
                "question": {"instructions": "Which intent?"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clean = [
        {"id": "a", "state": "A completely different sentence.", "instructions": "Something else."},
        {
            "id": "b",
            "state": "Where is my parcel?",
            "instructions": "Which intent?",
        },  # instruction!
    ]
    with pytest.raises(BuildError, match="evaluation-only"):
        assert_no_public_overlap(clean, public)
    assert_no_public_overlap(clean[:1], public)  # neither state nor instruction collides


@needs_sources
def test_build_end_to_end_produces_wire_valid_records(tmp_path: Path) -> None:
    out = tmp_path / "records.jsonl"
    report = build(out_path=out)
    config = load_config(BUILDER_CONFIG)
    assert report["total"] == sum(config["counts"].values())
    assert report["per_type"] == {"choice": 8000, "noul": 3000}
    assert report["checked_against_public"] is False

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == report["total"]
    revisions = {source["revision"][:12] for source in load_lock(DEFAULT_LOCK)["sources"].values()}
    seen_ids: set[str] = set()
    for line in lines:
        record = json.loads(line)
        assert record["id"] not in seen_ids
        seen_ids.add(record["id"])
        assert record["lang"] == "en"
        assert record["source"].split("@")[1] in revisions
        if record["type"] == "choice":
            question = ChoiceQuestion(
                instructions=record["instructions"], criteria=record["criteria"]
            )
            assert record["target"] in question.criteria
        else:
            question = NoulQuestion(
                instructions=record["instructions"], criteria=record["criteria"]
            )
            assert record["target"] in (0, 1)
        assert record["state"]
        assert normalized(record["state"]) == normalized(record["state"])
