"""Confidence derivation and temperature calibration.

Confidence is derived from the answer's probability distribution (CAL-03): the mass on the
selected option, so a peaked distribution is confident and a near-uniform one is not. Full
distributions are always part of the canonical answer, so ``return_details`` is a no-op
extension flag rather than a separate payload (CAL-05).

Temperature operates on probabilities via power scaling (``p**(1/T)``): ``T > 1`` softens a
distribution, ``T < 1`` sharpens it. ``fit_temperature`` picks the value minimizing expected
calibration error (ECE) on a held-out set; M4 reuses it for real calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Default temperature grid searched by :func:`fit_temperature`. Wide enough that a weakly
#: served language's sharpening/softening optimum is not truncated at an endpoint (B-1).
DEFAULT_GRID: tuple[float, ...] = (
    0.05,
    0.1,
    0.15,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
)


@dataclass(frozen=True, slots=True)
class Temperature:
    """A fitted temperature and the ECE it achieved."""

    value: float
    ece: float


def confidence(probabilities: Mapping[Any, float]) -> float:
    """Return the confidence (selected mass) of a distribution, in ``[0, 1]``.

    The input need not be normalized; negative weights are treated as zero. An empty or
    all-zero distribution yields ``0.0``.
    """
    values = [max(0.0, float(value)) for value in probabilities.values()]
    total = sum(values)
    if total <= 0.0 or not values:
        return 0.0
    return min(1.0, max(values) / total)


def _normalized_values(probabilities: Mapping[Any, float]) -> list[float]:
    """Return non-negative weights renormalized to sum to 1, or an empty list if degenerate."""
    values = [max(0.0, float(value)) for value in probabilities.values()]
    total = sum(values)
    if total <= 0.0 or not values:
        return []
    return [value / total for value in values]


def normalized_entropy(probabilities: Mapping[Any, float]) -> float:
    """Return Shannon entropy normalized to ``[0, 1]`` (``0`` certain, ``1`` uniform).

    The input need not be normalized; negative weights are treated as zero. An empty,
    all-zero, or single-key distribution yields ``0.0`` (no uncertainty is expressible).
    """
    values = _normalized_values(probabilities)
    if len(values) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in values if value > 0.0)
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def margin(probabilities: Mapping[Any, float]) -> float:
    """Return the gap between the top two probabilities, in ``[0, 1]``.

    A larger value means a more dominant top option. The input need not be normalized;
    negative weights are treated as zero. An empty, all-zero, or single-key distribution
    yields ``0.0``.
    """
    values = sorted(_normalized_values(probabilities), reverse=True)
    if len(values) < 2:
        return 0.0
    return min(1.0, max(0.0, values[0] - values[1]))


def apply_temperature(probabilities: Mapping[Any, float], temperature: float) -> dict[Any, float]:
    """Rescale a distribution by ``temperature`` and renormalize it."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    exponent = 1.0 / temperature
    powered = {key: max(0.0, float(value)) ** exponent for key, value in probabilities.items()}
    total = sum(powered.values())
    if total <= 0.0:
        uniform = 1.0 / len(powered) if powered else 0.0
        return dict.fromkeys(powered, uniform)
    return {key: value / total for key, value in powered.items()}


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> float:
    """Top-label ECE: the gap between confidence and accuracy across confidence bins."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for value, is_correct in zip(confidences, correct, strict=True):
        index = min(bins - 1, max(0, int(value * bins)))
        buckets[index].append((value, 1.0 if is_correct else 0.0))
    total = len(confidences)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return ece


def fit_temperature(
    samples: Sequence[Mapping[Any, float]],
    correct: Sequence[bool],
    *,
    grid: Sequence[float] = DEFAULT_GRID,
) -> Temperature:
    """Search ``grid`` for the temperature minimizing ECE on ``samples``."""
    if len(samples) != len(correct):
        raise ValueError("samples and correct must have the same length")
    best = Temperature(value=1.0, ece=float("inf"))
    for temperature in grid:
        confidences = [confidence(apply_temperature(sample, temperature)) for sample in samples]
        ece = expected_calibration_error(confidences, correct)
        if ece < best.ece:
            best = Temperature(value=temperature, ece=ece)
    return best


def parse_temperature_report(report: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a fitted-temperature report into ``{kind, kind:lang}`` keys.

    Accepts both the legacy per-primitive shape (``per_primitive[kind]["temperature"]``) and
    the per-language extension (``per_primitive[kind]["by_language"][lang]["temperature"]``),
    so old and new calibration files both load (B-1).
    """
    temperatures: dict[str, float] = {}
    if not isinstance(report, Mapping):
        return temperatures
    per_primitive = report.get("per_primitive", {})
    if not isinstance(per_primitive, Mapping):
        return temperatures
    for kind, entry in per_primitive.items():
        if not isinstance(entry, Mapping):
            continue
        if "temperature" in entry:
            temperatures[str(kind)] = float(entry["temperature"])
        by_language = entry.get("by_language")
        if isinstance(by_language, Mapping):
            for lang, lang_entry in by_language.items():
                if isinstance(lang_entry, Mapping) and "temperature" in lang_entry:
                    temperatures[f"{kind}:{lang}"] = float(lang_entry["temperature"])
    return temperatures


__all__ = [
    "DEFAULT_GRID",
    "Temperature",
    "apply_temperature",
    "confidence",
    "expected_calibration_error",
    "fit_temperature",
    "margin",
    "normalized_entropy",
    "parse_temperature_report",
]
