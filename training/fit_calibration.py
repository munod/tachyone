"""Fit per-(primitive, language) temperature on a calibration split to minimize ECE.

The script consumes a JSONL of *predictions* (distribution + target + language), never the test
split, and writes the fitted temperatures beside the checkpoint (CAL-02, TRAIN-04). Each
primitive keeps a global fit and gains a per-language fit; a language with too few examples is
warned about and falls back to the per-primitive value at runtime. The math is pure Python
(``tachyone.calibration``), so it runs without torch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tachyone.calibration import (
    DEFAULT_GRID,
    apply_temperature,
    confidence,
    expected_calibration_error,
    fit_temperature,
)

KINDS = ("noul", "choice", "score")


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One prediction to calibrate: a distribution (or noul probability), target, and language."""

    type: str
    probabilities: dict[str, float]
    target: str | int
    lang: str = "und"


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
            lang = str(payload.get("lang") or "und")
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
            examples.append(CalibrationExample(kind, distribution, target, lang))
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


def _fit_subset(
    examples: Sequence[CalibrationExample],
    *,
    bins: int,
    min_samples: int,
    grid: Sequence[float],
    label: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Fit one temperature for ``examples`` and report ECE before/after."""
    if len(examples) < min_samples:
        warnings.append(
            f"{label}: only {len(examples)} examples (< {min_samples}); fit may be noisy"
        )
    before = _ece(examples, 1.0, bins)
    fitted = fit_temperature(
        [example.probabilities for example in examples],
        [_correct(example) for example in examples],
        grid=grid,
    )
    after = _ece(examples, fitted.value, bins)
    return {
        "n": len(examples),
        "temperature": fitted.value,
        "ece_before": round(before, 6),
        "ece_after": round(after, 6),
    }


def fit_temperature_report(
    examples: Sequence[CalibrationExample],
    *,
    bins: int = 10,
    min_samples: int = 30,
    grid: Sequence[float] = DEFAULT_GRID,
) -> dict[str, Any]:
    """Fit one temperature per primitive and, additionally, per language.

    ``per_primitive[kind]`` keeps its global ``temperature``/``ece_*`` keys (stable shape) and
    gains ``by_language[lang]`` with the same keys. A language below ``min_samples`` is warned
    about; the runtime falls back to the per-primitive fit for it.
    """
    report: dict[str, Any] = {"per_primitive": {}, "bins": bins, "warnings": []}
    warnings: list[str] = report["warnings"]
    for kind in KINDS:
        subset = [example for example in examples if example.type == kind]
        if not subset:
            continue
        entry = _fit_subset(
            subset, bins=bins, min_samples=min_samples, grid=grid, label=kind, warnings=warnings
        )
        by_language: dict[str, Any] = {}
        for lang in sorted({example.lang for example in subset}):
            language_subset = [example for example in subset if example.lang == lang]
            by_language[lang] = _fit_subset(
                language_subset,
                bins=bins,
                min_samples=min_samples,
                grid=grid,
                label=f"{kind}:{lang}",
                warnings=warnings,
            )
        entry["by_language"] = by_language
        report["per_primitive"][kind] = entry
    return report


def save_report(report: dict[str, Any], out_path: str | Path) -> None:
    """Persist the fitted temperatures (and the full report) as JSON."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-fit-calibration", description=__doc__)
    parser.add_argument(
        "--calibration", required=True, help="predictions JSONL (calibration split)"
    )
    parser.add_argument("--out", required=True, help="output temperature report JSON")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument(
        "--grid",
        default=",".join(str(value) for value in DEFAULT_GRID),
        help="comma-separated temperature grid",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    grid = tuple(float(item) for item in args.grid.split(",") if item.strip()) or DEFAULT_GRID
    report = fit_temperature_report(
        load_examples(args.calibration),
        bins=args.bins,
        min_samples=args.min_samples,
        grid=grid,
    )
    save_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
