"""Fit P3 confidence on the legal calibration holdout (B-13, JB-12).

One command turns one predictions file into both assets the runtime reads:

- ``temperature_calibration.json`` — the existing schema, one temperature per primitive
  from the pre-registered grid, **except ``score``, which is pinned**: its answer *is* the
  distribution's expected value, so a temperature change would move the answer, and this
  phase must not move Intelligence (spec, *P3 design*).
- ``confidence_calibration.json`` — the prototype bank embedded together with the fitted
  ``noul`` map, so a map can never be paired with a bank it was not fitted against.

``choice`` keeps a global temperature because its own confidence already tracks its
accuracy in every regime; ``noul`` gets the map because it does not (measured AUC 0.477 on
the public items). Fit input is only what ``training.predict`` emitted for the holdout:

    uv run python -m training.predict --data data/calibration_holdout.jsonl \
        --adapter checkpoints/en --prototypes checkpoints/en/state_prototypes.json \
        --out-predictions data/calibration_holdout_preds.jsonl
    uv run python -m training.fit_confidence --predictions data/calibration_holdout_preds.jsonl \
        --prototypes checkpoints/en/state_prototypes.json --out-dir checkpoints/en
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tachyone.calibration import (
    expected_calibration_error,
    fit_confidence_map,
    interpolate_confidence,
)
from training.fit_calibration import _correct, _ece, fit_temperature_report, load_examples

KINDS = ("noul", "choice", "score")
#: ``score`` ships the temperature it was published with; the fit reports, it does not move it.
PINNED_SCORE_TEMPERATURE = 0.1
PIN_REASON = (
    "score's answer is the distribution's expected value: a temperature change moves the "
    "answer (rounded-EV accuracy 0.53 at T=1 vs 0.9996 shipped), so this phase pins it — "
    "P3 must not move Intelligence"
)


class FitError(RuntimeError):
    """A holdout that cannot support the pre-registered fit protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grouped(examples: Sequence[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for example in examples:
        groups.setdefault(example.slice or "all", []).append(example)
    return groups


def _map_confidence(example: Any, knots: Sequence[Sequence[float]]) -> float:
    """The confidence the map reports for this row, clamped to the binary floor (P3)."""
    if example.strength is None:
        raise FitError("a row has no 'strength' — run training.predict with --prototypes")
    level = interpolate_confidence(example.strength, knots)
    return min(1.0, max(0.5, level))


def _ece_rows(confidences: Sequence[float], correct: Sequence[bool], bins: int) -> float:
    return expected_calibration_error(list(confidences), list(correct), bins=bins)


def fit(
    *,
    predictions: str | Path,
    prototypes: str | Path,
    out_dir: str | Path,
    bins: int = 10,
    min_samples: int = 30,
    pin_score: float = PINNED_SCORE_TEMPERATURE,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fit both assets from one predictions file; write them unless ``dry_run``."""
    preds_path = Path(predictions)
    prototypes_path = Path(prototypes)
    if not preds_path.exists():
        raise FitError(f"missing predictions: {preds_path}")
    if not prototypes_path.exists():
        raise FitError(f"missing prototype bank: {prototypes_path}")
    examples = load_examples(preds_path)
    if not examples:
        raise FitError(f"no rows in {preds_path}")

    # --- temperatures: pooled fit, score pinned -----------------------------------------
    report = fit_temperature_report(examples, bins=bins, min_samples=min_samples)
    score_rows = [example for example in examples if example.type == "score"]
    if score_rows:
        pinned = fit_temperature_report(
            score_rows, bins=bins, min_samples=min_samples, grid=(pin_score,)
        )
        entry = dict(pinned["per_primitive"]["score"])
        entry["temperature"] = pin_score
        entry["pinned"] = True
        entry["reason"] = PIN_REASON
        report["per_primitive"]["score"] = entry
    report["pin"] = {"score": pin_score, "reason": PIN_REASON}

    # --- the noul evidence map ------------------------------------------------------------
    noul_rows = [example for example in examples if example.type == "noul"]
    if not noul_rows:
        raise FitError("the holdout has no noul rows to fit a map on")
    strengths: list[float] = []
    for example in noul_rows:
        if example.strength is None:
            raise FitError(
                "noul rows carry no 'strength' — regenerate the predictions with "
                "training.predict --prototypes"
            )
        strengths.append(float(example.strength))
    knots = fit_confidence_map(
        strengths,
        [_correct(example) for example in noul_rows],
        bins=bins,
    )
    if len(knots) < 2:
        raise FitError(
            f"the map collapsed to {len(knots)} knot(s) — the strengths do not span a range"
        )

    # --- diagnostics: every slice, both mechanisms ---------------------------------------
    diagnostics: dict[str, Any] = {"noul": {"temperature": {}, "map": {}}}
    for kind in KINDS:
        if kind == "noul":
            continue
        rows = [example for example in examples if example.type == kind]
        if not rows:
            continue
        temperature = float(report["per_primitive"][kind]["temperature"])
        diagnostics[kind] = {
            f"{label}": round(_ece(group, temperature, bins), 6)
            for label, group in sorted(_grouped(rows).items())
        }
        diagnostics[kind]["all"] = round(_ece(rows, temperature, bins), 6)
    temperature = float(report["per_primitive"]["noul"]["temperature"])
    for label, group in sorted(_grouped(noul_rows).items()):
        diagnostics["noul"]["temperature"][label] = round(_ece(group, temperature, bins), 6)
        diagnostics["noul"]["map"][label] = round(
            _ece_rows(
                [_map_confidence(example, knots) for example in group],
                [_correct(example) for example in group],
                bins,
            ),
            6,
        )
    diagnostics["noul"]["temperature"]["all"] = round(_ece(noul_rows, temperature, bins), 6)
    diagnostics["noul"]["map"]["all"] = round(
        _ece_rows(
            [_map_confidence(example, knots) for example in noul_rows],
            [_correct(example) for example in noul_rows],
            bins,
        ),
        6,
    )

    # --- the two assets --------------------------------------------------------------------
    bank = json.loads(prototypes_path.read_text(encoding="utf-8"))
    provenance = {
        "predictions": str(preds_path),
        "predictions_sha256": _sha256(preds_path),
        "prototypes": str(prototypes_path),
        "prototypes_sha256": _sha256(prototypes_path),
        "rows": len(examples),
        "noul_rows": len(noul_rows),
        "bins": bins,
    }
    confidence_asset = {
        "version": 1,
        "prototypes": bank,
        "noul": {"knots": [[float(x), float(y)] for x, y in knots]},
        "provenance": provenance,
    }
    report["confidence"] = {
        "knots": [[round(float(x), 6), round(float(y), 6)] for x, y in knots],
        "n": len(noul_rows),
        "provenance": provenance,
    }
    report["diagnostics"] = diagnostics
    # The temperature report *is* the file's content; only the confidence asset is extra.
    report["_assets"] = {"confidence_calibration.json": confidence_asset}

    if not dry_run:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        temperature_report = {key: value for key, value in report.items() if key != "_assets"}
        (out / "temperature_calibration.json").write_text(
            json.dumps(temperature_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (out / "confidence_calibration.json").write_text(
            json.dumps(confidence_asset, indent=1, sort_keys=True), encoding="utf-8"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-fit-confidence", description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL from training.predict")
    parser.add_argument("--prototypes", required=True, help="state_prototypes.json (JB-10)")
    parser.add_argument("--out-dir", required=True, help="the checkpoint directory")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--pin-score", type=float, default=PINNED_SCORE_TEMPERATURE)
    parser.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = fit(
        predictions=args.predictions,
        prototypes=args.prototypes,
        out_dir=args.out_dir,
        bins=args.bins,
        min_samples=args.min_samples,
        pin_score=args.pin_score,
        dry_run=args.dry_run,
    )
    printable = {key: value for key, value in report.items() if key != "_assets"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
