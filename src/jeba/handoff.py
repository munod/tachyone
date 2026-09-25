"""Client-side confidence thresholding and System-2 handoff.

jeba already returns ``confidence`` for ``choice``/``score`` and a probability for ``noul``, but
callers who want to *abstain* or *hand off to a slower System-2 LLM* when the engine is unsure
have to reimplement the threshold. This module provides the missing typed signal (B-3).

Everything here is **additive and client-side**: it reads a canonical
:class:`~jeba.wire.SystemOneResponse` and never changes the ``/v1/systemone`` shape (ADR-0001,
STATE L-004). The threshold is always required — there is no universal default because the right
``τ`` depends on the task (see ``docs/cookbook-handoff.md``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jeba.calibration import confidence, margin, normalized_entropy
from jeba.primitives import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from jeba.wire import SystemOneResponse


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Uncertainty metrics for one probability distribution and the resulting decision."""

    confidence: float
    entropy: float
    margin: float
    threshold: float
    abstain: bool


@dataclass(frozen=True, slots=True)
class HandoffSignal:
    """The handoff decision for a single answered question."""

    question_id: str
    type: str
    confidence: float
    entropy: float
    margin: float
    threshold: float
    abstain: bool


@dataclass(frozen=True, slots=True)
class HandoffReport:
    """Aggregate handoff decision across every answer in a response."""

    abstain: bool
    threshold: float
    signals: dict[str, HandoffSignal]


def _check_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    return value


def assess(probabilities: Mapping[Any, float], *, threshold: float) -> Uncertainty:
    """Assess a distribution against ``threshold`` and decide whether to hand off.

    ``abstain`` is true when the selected mass is strictly below ``threshold``. Empty or
    degenerate distributions score ``0.0`` and therefore abstain for any positive ``τ``.
    """
    tau = _check_threshold(threshold)
    selected = confidence(probabilities)
    return Uncertainty(
        confidence=selected,
        entropy=normalized_entropy(probabilities),
        margin=margin(probabilities),
        threshold=tau,
        abstain=selected < tau,
    )


def _answer_probabilities(answer: Answer) -> Mapping[Any, float]:
    if isinstance(answer, NoulAnswer):
        return {"noul": answer.noul, "not-noul": 1.0 - answer.noul}
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return answer.probabilities
    # The Answer union is closed; this is defensive only.
    raise TypeError(f"unsupported answer type {answer.type!r}")  # pragma: no cover


def _answer_confidence(answer: Answer, probabilities: Mapping[Any, float]) -> float:
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        return min(1.0, max(0.0, answer.confidence))
    # `noul` has no separate confidence: for a binary outcome, certainty is the mass on the
    # more likely side, `max(p, 1-p)`. Using `p` directly would flag a confident "no" (p≈0) as
    # uncertain (review #1). Entropy/margin already reflect this same distribution.
    return confidence(probabilities)


def assess_response(response: SystemOneResponse, *, threshold: float) -> HandoffReport:
    """Assess every answer in ``response`` and return an aggregate :class:`HandoffReport`.

    ``choice``/``score`` use their answer ``confidence``; ``noul`` has no separate confidence,
    so its certainty is ``max(p, 1-p)``. The aggregate ``abstain`` is true when any question
    abstains.
    """
    tau = _check_threshold(threshold)
    signals: dict[str, HandoffSignal] = {}
    for question_id, answer in response.answers.items():
        probabilities = _answer_probabilities(answer)
        selected = _answer_confidence(answer, probabilities)
        signals[question_id] = HandoffSignal(
            question_id=question_id,
            type=answer.type,
            confidence=selected,
            entropy=normalized_entropy(probabilities),
            margin=margin(probabilities),
            threshold=tau,
            abstain=selected < tau,
        )
    return HandoffReport(
        abstain=any(signal.abstain for signal in signals.values()),
        threshold=tau,
        signals=signals,
    )


__all__ = [
    "HandoffReport",
    "HandoffSignal",
    "Uncertainty",
    "assess",
    "assess_response",
]
