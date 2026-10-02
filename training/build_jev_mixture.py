"""Concatenate the B-13 P2 training mixture from its ordered parts (JB-5).

The fine-tune reads a single ``data_path``, so the mixture is a recipe, not a shell
command: ``training/configs/jev_mixture.json`` names the ordered parts and this builder
validates them (present, shape-complete, **globally unique ids**) before writing the
concatenation. Order is preserved; the trainer shuffles per epoch anyway (B-5's
``shuffled()``), and ``split_records`` seeds its own shuffle.

    uv run python -m training.build_jev_mixture
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from training.build_jev_sources import BuildError

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _ROOT / "training" / "configs" / "jev_mixture.json"

_CONFIG_KEYS = {"parts", "out"}
_REQUIRED_RECORD_KEYS = ("id", "type", "state", "instructions", "target", "lang")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Strict recipe loader (unknown keys are a mistake, L-011)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("_comment", None)
    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise BuildError(f"unknown config keys: {sorted(unknown)}")
    if _CONFIG_KEYS - set(data):
        raise BuildError(f"missing config keys: {sorted(_CONFIG_KEYS - set(data))}")
    if not isinstance(data["parts"], list) or not data["parts"]:
        raise BuildError("parts must be a non-empty list")
    return data


def read_part(path: Path) -> list[dict[str, Any]]:
    """One part: present, line-complete, and shaped like a record."""
    if not path.exists():
        raise BuildError(f"missing mixture part: {path} — build it first")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [key for key in _REQUIRED_RECORD_KEYS if key not in record]
        if missing:
            raise BuildError(f"{path}:{number}: record {record.get('id')!r} misses {missing}")
        records.append(record)
    if not records:
        raise BuildError(f"mixture part is empty: {path}")
    return records


def build(
    config_path: str | Path = DEFAULT_CONFIG, *, out_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate every part, refuse cross-part id collisions, write the concatenation."""
    config = load_config(config_path)
    records: list[dict[str, Any]] = []
    per_part: dict[str, int] = {}
    # Ids must be unique across parts (provenance), but the incumbent domain datasets use
    # per-(type, domain) counters, so ids legitimately repeat *inside* one part.
    seen: dict[str, str] = {}
    for part in config["parts"]:
        # Absolute parts (tests) pass through; repo-relative parts resolve from the root,
        # independent of where the config file itself lives.
        path = Path(part) if Path(part).is_absolute() else _ROOT / str(part)
        part_records = read_part(path)
        for record in part_records:
            record_id = str(record["id"])
            previous = seen.get(record_id)
            if previous is not None and previous != path.name:
                raise BuildError(f"duplicate record id {record_id!r}: {previous} and {path.name}")
            seen[record_id] = path.name
        records.extend(part_records)
        per_part[str(part)] = len(part_records)

    out = (
        Path(out_path)
        if out_path is not None
        else (
            _ROOT / str(config["out"])
            if not Path(str(config["out"])).is_absolute()
            else Path(str(config["out"]))
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    per_type: dict[str, int] = {}
    per_source: dict[str, int] = {}
    for record in records:
        per_type[str(record["type"])] = per_type.get(str(record["type"]), 0) + 1
        source = str(record.get("source", "unknown"))
        if source.startswith("synthetic:jev:"):
            family = source.rsplit(":", 1)[-1]  # long_policy, trap, ...
        elif "@" in source:
            family = source.split("@", 1)[0]  # multi_nli@<rev>, boolq@<rev>, ...
        else:
            family = source  # "synthetic" (the domain datasets)
        per_source[family] = per_source.get(family, 0) + 1
    return {
        "out": str(out),
        "total": len(records),
        "per_part": per_part,
        "per_type": dict(sorted(per_type.items())),
        "per_source": dict(sorted(per_source.items())),
        "unique_ids": len(seen),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-build-jev-mixture", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=None, help="override the config's output path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args.config, out_path=args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
