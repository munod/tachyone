"""Tests for the pinned JevBench-family source fetch (B-13 P2, JB-1).

Everything runs offline: the lock is validated structurally, downloads are exercised
against ``file://`` URLs, and the fetch/check/record flow runs on fake local files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

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
