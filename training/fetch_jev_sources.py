"""Fetch the pinned public sources for the JevBench-family records (B-13 P2).

The exact revisions and their sha256 live in ``training/data/jev_sources.lock.json``;
that lock (not the data) is the committed recipe — downloads land in ``data/sources/``
(gitignored). Files are streamed from the pinned resolve URL, verified, and never
overwritten: a hash mismatch is a hard error, not a re-download.

    uv run python -m training.fetch_jev_sources            # fetch missing + verify
    uv run python -m training.fetch_jev_sources --check    # verify only (no network)
    uv run python -m training.fetch_jev_sources --record   # first run: write sha256 into
                                                           # the lock, then commit it

Only stdlib is imported (urllib/hashlib/json), so the module is importable without the
``train`` extra; the parquet reading happens later in ``training/build_jev_sources.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = _ROOT / "training" / "data" / "jev_sources.lock.json"
DEFAULT_DATA_DIR = _ROOT / "data" / "sources"

_HEX40 = 40
_HEX64 = 64
_CHUNK = 1 << 20
_USER_AGENT = "tachyone-fetch/1 (+https://github.com/munod/tachyone)"


class LockError(ValueError):
    """The lock file is structurally wrong — fail before touching the network."""


def load_lock(path: str | Path = DEFAULT_LOCK) -> dict[str, Any]:
    """Load and strictly validate the sources lock."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("sources"), dict) or not data["sources"]:
        raise LockError("lock needs a non-empty 'sources' object")
    if not isinstance(data.get("excluded"), dict):
        raise LockError("lock needs an 'excluded' object (record why each source is out)")
    for name, source in data["sources"].items():
        for key in ("repo", "revision", "file", "url", "license", "sha256", "purpose"):
            if key not in source:
                raise LockError(f"source {name!r} is missing {key!r}")
        if len(source["revision"]) != _HEX40:
            raise LockError(f"source {name!r}: revision must be a 40-hex git sha")
        sha = source["sha256"]
        if sha and (len(sha) != _HEX64 or any(c not in "0123456789abcdef" for c in sha)):
            raise LockError(f"source {name!r}: sha256 must be 64 lowercase hex chars")
        if not source["file"].endswith(".parquet"):
            raise LockError(f"source {name!r}: expected a pinned parquet file")
        if not source["license"]:
            raise LockError(f"source {name!r}: licence metadata must be recorded")
    for name, excluded in data["excluded"].items():
        if "license" not in excluded or "reason" not in excluded:
            raise LockError(f"excluded source {name!r} needs 'license' and 'reason'")
    return data


def resolve_url(source: Mapping[str, Any]) -> str:
    """The pinned resolve URL for one source."""
    return (
        f"https://huggingface.co/datasets/{source['repo']}"
        f"/resolve/{source['revision']}/{source['file']}"
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` sibling, then rename (no partial files)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
    except OSError as exc:  # urllib wraps network failures in OSError subclasses
        part.unlink(missing_ok=True)
        raise SystemExit(f"download failed for {url}: {exc}") from exc
    part.replace(dest)


def fetch(
    lock: dict[str, Any],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    record: bool = False,
    check: bool = False,
) -> tuple[dict[str, str], bool]:
    """Fetch and verify every source; return ``(name -> sha256, lock_changed)``.

    Existing files are verified, never re-downloaded. ``check`` reports without network.
    ``record`` fills empty ``sha256`` fields from the downloaded bytes — the caller is
    responsible for persisting the lock.
    """
    results: dict[str, str] = {}
    changed = False
    for name, source in lock["sources"].items():
        path = data_dir / f"{name}{Path(source['file']).suffix}"
        expected = source["sha256"]
        if not path.exists():
            if check:
                print(f"[missing] {name}: {path}")
                results[name] = ""
                continue
            print(f"[fetch]   {name}: {resolve_url(source)}")
            download(resolve_url(source), path)
        actual = sha256_of(path)
        if expected and actual != expected:
            raise SystemExit(
                f"sha256 mismatch for {name}: lock={expected} file={actual} "
                f"({path}) — delete the file to re-download, or investigate the pin"
            )
        if record and not expected:
            source["sha256"] = actual
            changed = True
            print(f"[record]  {name}: {actual}")
        results[name] = actual
    if record and not changed:
        print("[record]  nothing to do: every lock entry already has a sha256")
    if changed:
        print("[record]  the lock changed — review and commit training/data/jev_sources.lock.json")
    return results, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-fetch-jev-sources", description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK), help="path to the sources lock")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="download directory")
    parser.add_argument("--check", action="store_true", help="verify only; never touch the network")
    parser.add_argument(
        "--record", action="store_true", help="write missing sha256 values into the lock"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    lock = load_lock(args.lock)
    if args.record and args.check:
        raise SystemExit("--record and --check are mutually exclusive")
    results, changed = fetch(
        lock, data_dir=Path(args.data_dir), record=args.record, check=args.check
    )
    if changed:
        Path(args.lock).write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.check and any(not digest for digest in results.values()):
        return 1  # missing files, reported above
    for name, digest in results.items():
        state = "ok" if digest else "MISSING"
        print(f"{name:12s} {state} {digest[:12]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
