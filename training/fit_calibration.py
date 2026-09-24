"""Fit per-primitive temperature on a calibration split to minimize ECE.

The script consumes a JSONL of *predictions* (distribution + target), never the test split,
and writes the fitted temperatures beside the checkpoint (CAL-02, TRAIN-04). The math is pure
Python (``jeba.calibration``), so it runs without torch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jeba.calibration import (
    DEFAULT_GRID,
    apply_temperature,
    confidence,
    expected_calibration_error,
    fit_temperature,
)

KINDS = ("noul", "choice", "score")


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One prediction to calibrate: a distribution (or noul probability) and its target."""

    type: str
    probabilities: dict[str, float]
    target: str | int


def load_examples(path: str | Path) -> list[CalibrationExample]:
    """Load prediction examples from JSONL."""
    examples: list[CalibrationExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            kind = payload.get("type")
            if kind not in KINDS:
                raise ValueError(f"line {line_number}: unknown type {kind!r}")
            probabilities = payload.get("probabilities")
            if kind == "noul":
                if not isinstance(probabilities, (int, float)):
                    raise ValueError(f"line {line_number}: noul needs a numeric probability")
                value = float(probabilities)
                distribution = {"true": value, "false": 1.0 - value}
                target: str | int = 1 if float(payload["target"]) >= 0.5 else 0
            else:
                if not isinstance(probabilities, Mapping):
                    raise ValueError(f"line {line_number}: {kind} needs a probabilities map")
                distribution = {str(key): float(value) for key, value in probabilities.items()}
                target = payload["target"]
            examples.append(CalibrationExample(kind, distribution, target))
    return examples


def _correct(example: CalibrationExample) -> bool:
    if example.type == "noul":
        predicted_true = example.probabilities.get("true", 0.0) >= 0.5
        return predicted_true == bool(example.target)
    predicted = max(example.probabilities, key=lambda key: example.probabilities[key])
    return str(predicted) == str(example.target)


def _ece(examples: Sequence[CalibrationExample], temperature: float, bins: int) -> float:
    confidences = [
        confidence(apply_temperature(example.probabilities, temperature)) for example in examples
    ]
    return expected_calibration_error(confidences, [_correct(example) for example in examples])


def fit_temperature_report(
    examples: Sequence[CalibrationExample],
    *,
    bins: int = 10,
    min_samples: int = 30,
    grid: Sequence[float] = DEFAULT_GRID,
) -> dict[str, Any]:
    """Fit one temperature per primitive and report ECE before/after."""
    report: dict[str, Any] = {"per_primitive": {}, "bins": bins, "warnings": []}
    for kind in KINDS:
        subset = [example for example in examples if example.type == kind]
        if not subset:
            continue
        if len(subset) < min_samples:
            report["warnings"].append(
                f"{kind}: only {len(subset)} examples (< {min_samples}); fit may be noisy"
            )
        before = _ece(subset, 1.0, bins)
        fitted = fit_temperature(
            [example.probabilities for example in subset],
            [_correct(example) for example in subset],
            grid=grid,
        )
        after = _ece(subset, fitted.value, bins)
        report["per_primitive"][kind] = {
            "n": len(subset),
            "temperature": fitted.value,
            "ece_before": round(before, 6),
            "ece_after": round(after, 6),
        }
    return report


def save_report(report: dict[str, Any], out_path: str | Path) -> None:
    """Persist the fitted temperatures (and the full report) as JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-fit-calibration", description=__doc__)
    parser.add_argument(
        "--calibration", required=True, help="predictions JSONL (calibration split)"
    )
    parser.add_argument("--out", required=True, help="output temperature report JSON")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=30)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = fit_temperature_report(
        load_examples(args.calibration), bins=args.bins, min_samples=args.min_samples
    )
    save_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
