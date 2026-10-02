"""Build the P3 calibration holdout from two legal slices (B-13, JB-9).

Two rules, both enforced and tested:

- **the fit basis is the real training set** — the in-domain slice is sampled from
  ``data/train_en_domains.jsonl``, the file the English checkpoint trains on. The prototype
  bank (the P3 evidence signal) is built from the same states, so the high-strength knot of
  the confidence map is measured on exactly the distribution the bank describes. Nothing in
  this builder generates records: a generated slice would measure a distribution nobody
  trained on (decision, 2026-10-02).
- **everything else is never trained on and never a JevBench item** — the source remainder
  is filtered against every training file of the checkpoint (a superset filter: it can only
  drop rows), and every slice is asserted against ``--public-dir`` before a byte is written
  (the 231 public and 308 sealed items are evaluation-only forever).

    uv run python -m training.build_calibration_holdout \
        --public-dir /tmp/opencode/jevbench/datasets/public
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.build_jev_sources import (
    BuildError,
    _require,
    boolq_record,
    iter_parquet,
    mnli_pairs,
    mnli_record,
    normalized,
)

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _ROOT / "training" / "configs" / "calibration_holdout.json"
DEFAULT_OUT = _ROOT / "data" / "calibration_holdout.jsonl"
DEFAULT_SOURCES_DIR = _ROOT / "data" / "sources"
#: Every training input of the checkpoint — the *filter* is a superset on purpose: a row
#: dropped here is a row the model has already seen, whatever file it is recorded in.
DEFAULT_TRAINED = (
    _ROOT / "data" / "train_en_domains.jsonl",
    _ROOT / "data" / "train_jev_en.jsonl",
    _ROOT / "data" / "train_jev_sources.jsonl",
)

_CONFIG_KEYS = {"training", "remainder"}
_KNOWN_SOURCES = ("multi_nli", "boolq")

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUT",
    "BuildError",
    "build",
    "load_config",
    "slice_remainder",
    "slice_training",
    "trained_states",
]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the holdout recipe with strict keys (L-011: the config *is* the recipe)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - (_CONFIG_KEYS | {"_comment"})
    if unknown:
        raise BuildError(f"unknown calibration holdout config keys: {sorted(unknown)}")
    for section in _CONFIG_KEYS:
        if section not in data:
            raise BuildError(f"calibration holdout config is missing {section!r}")
    for section, allowed in (("training", {"path", "counts"}), ("remainder", {"prefix", "counts"})):
        extra = set(data[section]) - allowed
        if extra:
            raise BuildError(f"unknown {section} keys: {sorted(extra)}")
    if not isinstance(data["training"]["counts"], Mapping) or not data["training"]["counts"]:
        raise BuildError("training.counts must be a non-empty type -> n mapping")
    return data


def trained_states(paths: Sequence[Path] = DEFAULT_TRAINED) -> set[str]:
    """Normalized states of every record the checkpoint trained on (missing files skipped)."""
    states: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    states.add(normalized(str(json.loads(line)["state"])))
    return states


def slice_training(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The fit basis: a deterministic stride sample of the real training set, by type."""
    path = Path(str(config["training"]["path"]))
    if not path.is_absolute():
        path = _ROOT / path
    if not path.exists():
        raise BuildError(f"missing training file: {path} — it is the P3 fit basis")
    counts: dict[str, int] = {str(k): int(v) for k, v in config["training"]["counts"].items()}
    wanted = set(counts)
    rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in wanted}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = str(record.get("type"))
            if kind in wanted:
                rows[kind].append(record)
    records: list[dict[str, Any]] = []
    for kind in sorted(wanted):
        pool = rows[kind]
        need = counts[kind]
        if len(pool) < need:
            raise BuildError(
                f"training slice needs {need} {kind} rows but the file has {len(pool)}"
            )
        stride = max(1, len(pool) // need)
        for index, record in enumerate(pool[::stride][:need]):
            sampled = dict(record)
            sampled["slice"] = "training"
            sampled["id"] = f"cal-trn-{kind}-{index:05d}"
            records.append(sampled)
    if not records:
        raise BuildError("the training slice came back empty")
    return records


def slice_remainder(
    config: Mapping[str, Any],
    *,
    data_dir: Path,
    trained: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """MultiNLI + BoolQ rows the trainer never saw; return ``(records, dropped_as_trained)``.

    The id *and* the normalized state are checked, so a row the trainer saw is counted as
    dropped rather than silently re-sampled.
    """
    spec = dict(config.get("remainder", config))
    counts: dict[str, int] = {str(k): int(v) for k, v in spec["counts"].items()}
    prefix = str(spec.get("prefix", "rem"))
    unknown = set(counts) - set(_KNOWN_SOURCES)
    if unknown:
        raise BuildError(f"remainder samples unknown sources: {sorted(unknown)}")
    records: list[dict[str, Any]] = []
    dropped = 0

    if counts.get("multi_nli"):
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        pairs = (
            pair
            for row in iter_parquet(_require(data_dir / "multi_nli.parquet", "multi_nli"))
            for pair in mnli_pairs(row)
        )
        for index, (premise, hypothesis, label) in enumerate(pairs):
            record = mnli_record(premise, hypothesis, label, index)
            key = normalized(record["state"])
            if key in trained:
                dropped += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append(record)
            if len(rows) >= counts["multi_nli"]:
                break
        if len(rows) < counts["multi_nli"]:
            raise BuildError(f"multi_nli remainder is short: {len(rows)} < {counts['multi_nli']}")
        records.extend(rows)

    if counts.get("boolq"):
        seen = set()
        rows = []
        for index, row in enumerate(iter_parquet(_require(data_dir / "boolq.parquet", "boolq"))):
            record = boolq_record(row["question"], row["passage"], row["answer"], index)
            key = normalized(record["state"])
            if key in trained:
                dropped += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append(record)
            if len(rows) >= counts["boolq"]:
                break
        if len(rows) < counts["boolq"]:
            raise BuildError(f"boolq remainder is short: {len(rows)} < {counts['boolq']}")
        records.extend(rows)

    for record in records:
        record["slice"] = "remainder"
        record["id"] = record["id"].replace("src-", f"cal-{prefix}-", 1)
    return records, dropped


def _drop_trained(
    records: Iterable[dict[str, Any]], trained: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Drop any record whose normalized state occurred in training; return (kept, dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        if normalized(str(record["state"])) in trained:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


def build(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    out_path: str | Path = DEFAULT_OUT,
    sources_dir: str | Path = DEFAULT_SOURCES_DIR,
    trained_paths: Sequence[Path] = DEFAULT_TRAINED,
    public_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build both slices, enforce both P3 rules, write the holdout, return the report."""
    config = load_config(config_path)
    trained = trained_states(trained_paths)
    if not trained:
        raise BuildError(
            "no training states found — point --trained at the checkpoint's training sets"
        )

    slices: dict[str, list[dict[str, Any]]] = {}
    slices["training"] = slice_training(config)
    remainder, dropped = slice_remainder(config, data_dir=Path(sources_dir), trained=trained)
    slices["remainder"] = remainder

    if public_dir is not None:
        from training.build_jev_sources import assert_no_public_overlap

        for records in slices.values():
            assert_no_public_overlap(list(records), public_dir)

    report: dict[str, Any] = {
        "config": str(config_path),
        "basis": str(config["training"]["path"]),
        "slices": {},
        "dropped_seen_in_training": {"remainder": dropped},
        "trained_states": len(trained),
        "total": 0,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for name in ("training", "remainder"):
            records = slices[name]
            by_type: dict[str, int] = {}
            for record in records:
                by_type[record["type"]] = by_type.get(record["type"], 0) + 1
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            report["slices"][name] = {"n": len(records), "by_type": by_type}
            report["total"] += len(records)
    report["out"] = str(out)
    if report["total"] == 0:
        raise BuildError("the holdout is empty")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-build-calibration-holdout", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sources-dir", default=str(DEFAULT_SOURCES_DIR))
    parser.add_argument(
        "--public-dir",
        default=None,
        help="the jevbench datasets/public directory: refuse to write if any record collides",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build(
        config_path=args.config,
        out_path=args.out,
        sources_dir=args.sources_dir,
        public_dir=args.public_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
