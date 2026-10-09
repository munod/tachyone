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
from pathlib import Path
from typing import Any

from tachyone.backends.encoder import CONFIDENCE_VERSION, calibration_language
from tachyone.calibration import (
    apply_temperature,
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
) -> dict[str, dict[str, Any]]:
    """Build every ``cells`` entry from pooled rows carrying (kind, serve_lang, signals)."""
    cells: dict[str, dict[str, Any]] = {}
    keys: list[tuple[str, str | None]] = []
    for kind in KINDS:
        keys.append((kind, None))  # global fallback over every row of the primitive
        keys.extend((kind, lang) for lang in LANGUAGES)
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
        correct = [row["correct"] for row in subset]
        mean = sum(1.0 for ok in correct if ok) / len(correct)
        peaked_buckets, peaked_edges = _quantile_buckets(peaked, bins)
        strength_buckets, strength_edges = _quantile_buckets(strength, bins)
        counts: dict[tuple[int, int], list[int]] = {}
        for index, (pb, sb) in enumerate(zip(peaked_buckets, strength_buckets, strict=True)):
            counts.setdefault((pb, sb), [0, 0])
            counts[(pb, sb)][0] += int(correct[index])
            counts[(pb, sb)][1] += 1
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
    args = parser.parse_args(argv)

    if len(args.predictions) != len(args.records):
        raise SystemExit("--predictions and --records must pair one to one")

    from training.evaluate import record_to_example  # local import keeps module load light

    temperatures = parse_temperature_report(json.loads(Path(args.temperature).read_text()))
    pooled: list[dict[str, Any]] = []
    seen = 0
    matched = 0
    diagonal = 0
    detectable = 0
    for predictions_path, records_path in zip(args.predictions, args.records, strict=True):
        predictions = _load_jsonl(Path(predictions_path))
        records = _load_jsonl(Path(records_path))
        if len(predictions) != len(records):
            raise SystemExit(
                f"{predictions_path} has {len(predictions)} rows but {records_path} "
                f"has {len(records)} — refusing to pair rows (L-019: an eval id is not "
                f"a row key)"
            )
        for record, row in zip(records, predictions, strict=True):
            if row.get("id") != record["id"] or row.get("type") != record["type"]:
                raise SystemExit(
                    f"misaligned pairing at {record['id']!r}: predictions carry "
                    f"{row.get('id')!r}/{row.get('type')!r} — refusing to fit (L-019)"
                )
            seen += 1
            example = record_to_example(record)
            serve = calibration_language(example.state, example.question)
            serve_lang = serve if serve in LANGUAGES else None
            kind = record["type"]
            if kind not in KINDS:
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
                }
            )

    if not pooled:
        raise SystemExit("no choice/score rows to fit")
    cells = fit_cells(pooled, bins=args.bins, shrink=args.shrink)
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
                "basis": "pooled in-template train + holdout (B-15 accepted)",
                "key": "kind:calibration_language (state + localized question, B-16)",
            },
            "fit_basis_rows": len(pooled),
            "serve_key_diagonal": round(diagonal / detectable, 4) if detectable else None,
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
