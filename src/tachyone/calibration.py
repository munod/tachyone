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
from itertools import pairwise
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


def interpolate_confidence(strength: float, knots: Sequence[Sequence[float]]) -> float:
    """Piecewise-linear ``confidence = f(strength)`` over ``(strength, confidence)`` knots.

    The knots are sorted by strength; outside the fitted range the answer is the nearest
    knot's confidence (an input *more* in-domain than anything seen keeps the in-domain
    confidence, an input further out keeps the weakest one fitted — never an extrapolation
    into confidence 0 or 1). The caller decides what the value means for its primitive; for
    ``noul`` the runtime clamps it to ``[0.5, 1]``, because a binary confidence below 0.5
    would invert the answer rather than express doubt (P3).
    """
    if not knots:
        raise ValueError("knots must not be empty")
    if strength <= knots[0][0]:
        return float(knots[0][1])
    if strength >= knots[-1][0]:
        return float(knots[-1][1])
    for (left_x, left_y), (right_x, right_y) in pairwise(knots):
        if left_x <= strength <= right_x:
            if right_x == left_x:
                return float(right_y)
            ratio = (strength - left_x) / (right_x - left_x)
            return float(left_y + ratio * (right_y - left_y))
    return float(knots[-1][1])  # pragma: no cover - unreachable given the two guards above


def _isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Pool-adjacent-violators: the weighted, non-decreasing fit closest to ``values``."""
    blocks: list[list[float]] = []  # [sum(w*y), sum(w), count]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([value * weight, weight, 1.0])
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous[0] / previous[1] <= current[0] / current[1]:
                break
            blocks.pop()
            blocks.pop()
            blocks.append(
                [previous[0] + current[0], previous[1] + current[1], previous[2] + current[2]]
            )
    fitted: list[float] = []
    for total, weight, count in blocks:
        fitted.extend([total / weight] * int(count))
    return fitted


def fit_confidence_map(
    strengths: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> list[tuple[float, float]]:
    """Fit ``confidence = f(strength)`` as empirical accuracy per equal-frequency bin.

    This is the ``noul`` calibration of P3: the primitive's own confidence carries no
    difficulty information (measured AUC 0.477 on the public items), so confidence comes
    from the *evidence* — how similar the state is to the training distribution — and the
    bin accuracies are what the model actually achieves at each level of it.

    Equal-frequency bins (not equal-width) because strengths cluster near the top of the
    range and equal-width bins would leave the sparse region unfitted. The bin accuracies
    are then made non-decreasing with pool-adjacent-violators: less in-domain evidence must
    never promise *more* confidence than more in-domain evidence, which is what lets the
    map be fitted on one surface and hold on another.
    """
    if len(strengths) != len(correct):
        raise ValueError("strengths and correct must have the same length")
    if not strengths:
        raise ValueError("need at least one sample")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    order = sorted(range(len(strengths)), key=lambda index: strengths[index])
    groups: list[list[int]] = [[] for _ in range(bins)]
    for position, index in enumerate(order):
        groups[min(bins - 1, position * bins // len(order))].append(index)
    centers: list[float] = []
    accuracies: list[float] = []
    counts: list[float] = []
    for group in groups:
        if not group:
            continue
        values = [strengths[index] for index in group]
        centers.append(sum(values) / len(values))
        accuracies.append(sum(1.0 for index in group if correct[index]) / len(group))
        counts.append(float(len(group)))
    if not centers:  # pragma: no cover - a non-empty input always fills a group
        raise ValueError("no samples to fit")
    adjusted = _isotonic(accuracies, counts)
    # Equal-frequency bins over a clustered signal can put two groups at the same strength;
    # merge them so the emitted map is strictly increasing (the runtime refuses knots that
    # are not), keeping the pooled accuracy of the merged group.
    knots: list[list[float]] = []
    for center, value, count in zip(centers, adjusted, counts, strict=True):
        if knots and knots[-1][0] == center:
            pooled = knots[-1]
            total = pooled[2] + count
            pooled[1] = (pooled[1] * pooled[2] + value * count) / total
            pooled[2] = total
            continue
        knots.append([center, value, count])
    return [(center, value) for center, value, _count in knots]


__all__ = [
    "DEFAULT_GRID",
    "Temperature",
    "apply_temperature",
    "confidence",
    "expected_calibration_error",
    "fit_confidence_map",
    "fit_temperature",
    "interpolate_confidence",
    "margin",
    "normalized_entropy",
    "parse_temperature_report",
]
