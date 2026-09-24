"""RLCD: training against strictly proper scoring rules.

A strictly proper scoring rule is minimized in expectation by reporting the true
probability, which is what makes ``confidence`` meaningful (CAL-01). This module holds the
pure-Python math — Brier and logarithmic (log) losses and the RLCD reward — testable without
torch. The training loop in ``finetune_rlcd.py`` reuses the same rules on logits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

PROPER_RULES: tuple[str, ...] = ("brier", "log")

_EPSILON = 1e-12


def _normalize(probabilities: Mapping[Any, float]) -> dict[Any, float]:
    values = {key: max(0.0, float(value)) for key, value in probabilities.items()}
    total = sum(values.values())
    if total <= 0.0:
        uniform = 1.0 / len(values) if values else 0.0
        return dict.fromkeys(values, uniform)
    return {key: value / total for key, value in values.items()}


def _one_hot(target: Any, keys: list[Any]) -> dict[Any, float]:
    if target not in keys:
        raise ValueError(f"target {target!r} is not one of {keys!r}")
    return {key: (1.0 if key == target else 0.0) for key in keys}


def categorical_loss(
    probabilities: Mapping[Any, float],
    target: Any,
    *,
    rule: str = "brier",
) -> float:
    """Proper scoring loss for a distribution over discrete options/levels."""
    if rule not in PROPER_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {PROPER_RULES}")
    distribution = _normalize(probabilities)
    keys = list(distribution)
    truth = _one_hot(target, keys)
    if rule == "brier":
        return sum((distribution[key] - truth[key]) ** 2 for key in keys)
    return -sum(truth[key] * math.log(max(distribution[key], _EPSILON)) for key in keys)


def noul_loss(probability: float, target: float, *, rule: str = "brier") -> float:
    """Proper scoring loss for a binary probability ``p`` against a soft label ``y``."""
    if rule not in PROPER_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {PROPER_RULES}")
    p = min(1.0, max(0.0, float(probability)))
    y = min(1.0, max(0.0, float(target)))
    if rule == "brier":
        return (p - y) ** 2
    p = min(1.0 - _EPSILON, max(_EPSILON, p))
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def rlcd_loss(
    kind: str,
    probabilities: float | Mapping[Any, float],
    target: float | Any,
    *,
    rule: str = "brier",
) -> float:
    """Dispatch to the right proper-scoring loss for the primitive ``kind``."""
    if kind == "noul":
        if isinstance(probabilities, Mapping):
            raise TypeError("noul expects a single probability, not a distribution")
        return noul_loss(probabilities, float(target), rule=rule)
    if kind in {"choice", "score"}:
        if not isinstance(probabilities, Mapping):
            raise TypeError(f"{kind} expects a distribution")
        return categorical_loss(probabilities, target, rule=rule)
    raise ValueError(f"unknown primitive kind: {kind!r}")


def rlcd_reward(
    kind: str,
    probabilities: float | Mapping[Any, float],
    target: float | Any,
    *,
    rule: str = "brier",
) -> float:
    """The RL signal: the negative proper-scoring loss (higher is better)."""
    return -rlcd_loss(kind, probabilities, target, rule=rule)


__all__ = [
    "PROPER_RULES",
    "categorical_loss",
    "noul_loss",
    "rlcd_loss",
    "rlcd_reward",
]
