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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Default temperature grid searched by :func:`fit_temperature`.
DEFAULT_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


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


__all__ = [
    "DEFAULT_GRID",
    "Temperature",
    "apply_temperature",
    "confidence",
    "expected_calibration_error",
    "fit_temperature",
]
