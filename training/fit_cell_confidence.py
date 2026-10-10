"""Fit the per-cell 2D confidence map for one adapter (B-16).

Reads the per-row strength predictions for the pooled fit basis (in-template train slice +
holdout slice — B-15's accepted calibration basis), the matching eval records (the serve
key comes from :func:`tachyone.backends.encoder.calibration_language`, the same function the
runtime applies, so fit and serve cannot drift), the fitted temperatures (both signals are
computed on the temperature-scaled distribution) and the prototype bank; writes
``confidence_calibration.json`` with the bank embedded so the pair is atomic (P3/JB-10).

Recipe — pre-registered in BACKLOG B-16 before this script existed:

- signal pair: ``peakedness`` (``1 - normalized_entropy`` of the served distribution) x
  ``strength`` (max cosine to the bank);
- quantile buckets ``bins x bins`` (ties land in the first bucket they touch), Laplace
  ``shrink`` toward the cell mean, unfitted buckets emitted as ``null``;
- cells keyed ``kind:serve-lang`` (the six trained languages) plus a per-kind global cell
  over **every** row of the primitive — the runtime's ``kind:lang`` → ``kind`` fallback.

The predictions rows and their records are joined **positionally within each file pair** —
never by ``id``: an eval id is not unique even inside one file (it restarts per domain,
L-019), so the pairing asserts ``id`` and ``type`` equality at every position and refuses
to continue on any drift, with equal lengths checked before the first row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tachyone.backends.encoder import CONFIDENCE_VERSION, CellConfidence, calibration_language
from tachyone.calibration import (
    apply_temperature,
    confidence,
    expected_calibration_error,
    normalized_entropy,
    parse_temperature_report,
)

LANGUAGES: tuple[str, ...] = ("de", "es", "fr", "it", "nl", "pt")
KINDS: tuple[str, ...] = ("choice", "score")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile_buckets(values: list[float], bins: int) -> tuple[list[int], list[float]]:
    """Rank buckets (ties in the first bucket they touch) and the per-bucket maxima.

    The stored edges are the bucket maxima of all but the last bucket; the runtime's
    ``bisect_left(edges, value)`` reproduces the training assignment exactly, including
    ties (the first edge equal to the value selects that bucket) and *empty* buckets
    (an empty bucket inherits the previous maximum, bumped by one ULP, so lookups skip
    over the region no basis row occupies — as the rank assignment would).
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    buckets = [0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        bucket = min(bins - 1, start * bins // len(order))
        for position in range(start, end + 1):
            buckets[order[position]] = bucket
        start = end + 1
    maxima: list[float | None] = [None] * max(0, bins - 1)
    for value, bucket in zip(values, buckets, strict=True):
        if bucket >= len(maxima):
            continue
        current = maxima[bucket]
        if current is None or value > current:
            maxima[bucket] = value
    edges: list[float] = []
    previous = -math.inf
    for maximum in maxima:
        candidate = previous if maximum is None else maximum
        if candidate <= previous:
            candidate = math.nextafter(previous, math.inf)
        edges.append(candidate)
        previous = candidate
    return buckets, edges


def fit_cells(
    rows: list[dict[str, Any]],
    *,
    bins: int,
    shrink: float,
    kinds: Sequence[str] = KINDS,
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, dict[str, Any]]:
    """Build every ``cells`` entry from pooled rows carrying (kind, serve_lang, signals).

    Each row may carry a ``weight`` (default 1.0); the cell accuracies, the shrink target
    and the per-bucket counts are weight-aware, so a caller can pool slices with balanced
    mass (B-18's slice-balanced pooling) instead of by row count. The quantile buckets
    stay rank-based over rows — the bucket *edges* describe where the evidence sits, not
    how loudly each slice voted.
    """
    cells: dict[str, dict[str, Any]] = {}
    keys: list[tuple[str, str | None]] = []
    for kind in kinds:
        keys.append((kind, None))  # global fallback over every row of the primitive
        keys.extend((kind, lang) for lang in languages)
    for key in keys:
        kind, lang = key
        subset = [
            row
            for row in rows
            if row["kind"] == kind and (lang is None or row["serve_lang"] == lang)
        ]
        if not subset:
            continue
        peaked = [row["peaked"] for row in subset]
        strength = [row["strength"] for row in subset]
        weights = [float(row.get("weight", 1.0)) for row in subset]
        total_weight = sum(weights)
        mean = (
            sum(w for row, w in zip(subset, weights, strict=True) if row["correct"]) / total_weight
        )
        peaked_buckets, peaked_edges = _quantile_buckets(peaked, bins)
        strength_buckets, strength_edges = _quantile_buckets(strength, bins)
        counts: dict[tuple[int, int], list[float]] = {}
        for index, (pb, sb) in enumerate(zip(peaked_buckets, strength_buckets, strict=True)):
            counts.setdefault((pb, sb), [0.0, 0.0])
            counts[(pb, sb)][0] += weights[index] if subset[index]["correct"] else 0.0
            counts[(pb, sb)][1] += weights[index]
        table: list[list[float | None]] = [
            [None] * (len(strength_edges) + 1) for _ in range(len(peaked_edges) + 1)
        ]
        for (pb, sb), (hits, count) in counts.items():
            table[pb][sb] = (hits + shrink * mean) / (count + shrink)
        cells[f"{kind}:{lang}" if lang else kind] = {
            "peaked_edges": peaked_edges,
            "strength_edges": strength_edges,
            "table": table,
            "mean": mean,
            "n": len(subset),
        }
    return cells


def _slice_diagnostics(
    pooled: list[dict[str, Any]], cells: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per-slice ECE of the fitted map against the temperature-only mechanism (B-18, T4).

    Evaluated through :class:`CellConfidence` itself — the same lookup the runtime
    performs (``kind:serve-lang`` then the per-primitive global cell), so the reported
    numbers are what the served path would print, not a re-implementation of it.
    """
    fitted = {
        name: CellConfidence(
            tuple(cell["peaked_edges"]),
            tuple(cell["strength_edges"]),
            tuple(tuple(row) for row in cell["table"]),
            float(cell["mean"]),
            int(cell["n"]),
        )
        for name, cell in cells.items()
    }
    out: dict[str, Any] = {}
    for label in ["all", *sorted({row["slice"] for row in pooled})]:
        subset = pooled if label == "all" else [row for row in pooled if row["slice"] == label]
        if not subset:
            continue
        entry: dict[str, Any] = {"n": len(subset)}
        for kind in sorted({row["kind"] for row in subset}):
            rows = [row for row in subset if row["kind"] == kind]
            correct = [bool(row["correct"]) for row in rows]
            mapped: list[float] = []
            for row in rows:
                cell = fitted.get(f"{row['kind']}:{row['serve_lang']}")
                if cell is None:
                    cell = fitted.get(row["kind"])
                if cell is None:
                    mapped.append(float(row["t_confidence"]))
                else:
                    mapped.append(cell.confidence(float(row["peaked"]), float(row["strength"])))
            entry[kind] = {
                "map_ece": round(expected_calibration_error(mapped, correct, bins=10), 6),
                "temperature_ece": round(
                    expected_calibration_error(
                        [float(row["t_confidence"]) for row in rows], correct, bins=10
                    ),
                    6,
                ),
            }
        out[label] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="strength-prediction JSONLs (one per records file, same order)",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="the eval JSONL each predictions file was produced from (pairwise, same order)",
    )
    parser.add_argument("--temperature", required=True, help="the fitted temperature report")
    parser.add_argument("--prototypes", required=True, help="state_prototypes.json to embed")
    parser.add_argument("--out", required=True, help="confidence_calibration.json to write")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--shrink", type=float, default=4.0)
    parser.add_argument(
        "--kinds",
        default=",".join(KINDS),
        help="question types to fit cells for (default: choice,score)",
    )
    parser.add_argument(
        "--languages",
        default=",".join(LANGUAGES),
        help="serve keys that get their own cell; anything else feeds the global one",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="existing asset whose unrefit keys are carried over (B-18: noul/provenance)",
    )
    parser.add_argument(
        "--balance-slices",
        action="store_true",
        help="pool the predictions files with equal total weight (B-18 slice-balanced pooling)",
    )
    args = parser.parse_args(argv)

    if len(args.predictions) != len(args.records):
        raise SystemExit("--predictions and --records must pair one to one")

    from training.evaluate import record_to_example  # local import keeps module load light

    temperatures = parse_temperature_report(json.loads(Path(args.temperature).read_text()))
    kinds = tuple(item.strip() for item in args.kinds.split(",") if item.strip())
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    pooled: list[dict[str, Any]] = []
    seen = 0
    matched = 0
    diagonal = 0
    detectable = 0
    for file_index, (predictions_path, records_path) in enumerate(
        zip(args.predictions, args.records, strict=True)
    ):
        predictions = _load_jsonl(Path(predictions_path))
        records = _load_jsonl(Path(records_path))
        if len(predictions) != len(records):
            raise SystemExit(
                f"{predictions_path} has {len(predictions)} rows but {records_path} "
                f"has {len(records)} — refusing to pair rows (L-019: an eval id is not "
                f"a row key)"
            )
        slices_seen = sorted({str(row["slice"]) for row in predictions if row.get("slice")})
        slice_label = "+".join(slices_seen) if slices_seen else Path(predictions_path).name
        for record, row in zip(records, predictions, strict=True):
            if row.get("id") != record["id"] or row.get("type") != record["type"]:
                raise SystemExit(
                    f"misaligned pairing at {record['id']!r}: predictions carry "
                    f"{row.get('id')!r}/{row.get('type')!r} — refusing to fit (L-019)"
                )
            seen += 1
            example = record_to_example(record)
            serve = calibration_language(example.state, example.question)
            serve_lang = serve if serve in languages else None
            kind = record["type"]
            if kind not in kinds:
                continue
            matched += 1
            if serve_lang is not None:
                detectable += 1
                diagonal += int(serve_lang == record["lang"])
            temperature = temperatures.get(
                f"{kind}:{serve_lang or ''}", temperatures.get(kind, 1.0)
            )
            probabilities = row["probabilities"]
            scaled = apply_temperature(probabilities, temperature)
            best = max(scaled, key=scaled.__getitem__)
            target = str(record["target"]) if kind == "score" else record["target"]
            pooled.append(
                {
                    "kind": kind,
                    "serve_lang": serve_lang,
                    "peaked": 1.0 - normalized_entropy(scaled),
                    "strength": float(row["strength"]),
                    "correct": best == target,
                    "t_confidence": confidence(scaled),
                    "slice": slice_label,
                    "file_index": file_index,
                }
            )

    if not pooled:
        raise SystemExit(f"no {'/'.join(kinds)} rows to fit")
    if args.balance_slices:
        file_counts: dict[int, int] = {}
        for row in pooled:
            file_counts[row["file_index"]] = file_counts.get(row["file_index"], 0) + 1
        per_slice = len(pooled) / len(file_counts)
        for row in pooled:
            row["weight"] = per_slice / file_counts[row["file_index"]]
    cells = fit_cells(pooled, bins=args.bins, shrink=args.shrink, kinds=kinds, languages=languages)
    if not cells:
        raise SystemExit("fit produced no cells")

    prototypes = json.loads(Path(args.prototypes).read_text())
    asset = {
        "version": CONFIDENCE_VERSION,
        "prototypes": prototypes,
        "cells": cells,
        "meta": {
            "recipe": {
                "signals": "peakedness x strength",
                "bins": args.bins,
                "shrink": args.shrink,
                "basis": (
                    "pooled in-template train + holdout, slice-balanced (B-18)"
                    if args.balance_slices
                    else "pooled in-template train + holdout (B-15 accepted)"
                ),
                "key": "kind:calibration_language (state + localized question, B-16)",
                "kinds": list(kinds),
                "languages": list(languages),
            },
            "fit_basis_rows": len(pooled),
            "serve_key_diagonal": round(diagonal / detectable, 4) if detectable else None,
            "slice_diagnostics": _slice_diagnostics(pooled, cells),
            "inputs": {
                path: _sha256(Path(path))
                for path in [
                    *args.predictions,
                    *args.records,
                    args.temperature,
                    args.prototypes,
                ]
            },
        },
    }
    if args.base:
        base_payload = json.loads(Path(args.base).read_text(encoding="utf-8"))
        if not isinstance(base_payload, dict):
            raise SystemExit(f"{args.base} is not a JSON object — refusing to carry it over")
        carried: list[str] = []
        for key in ("noul", "provenance"):
            if key in base_payload:
                asset[key] = base_payload[key]
                carried.append(key)
        asset["meta"]["base"] = {
            "path": str(args.base),
            "sha256": _sha256(Path(args.base)),
            "carried": carried,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "cells": len(cells),
                "rows": len(pooled),
                "serve_key_diagonal": round(diagonal / detectable, 4) if detectable else None,
                "shas": asset["meta"]["inputs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
