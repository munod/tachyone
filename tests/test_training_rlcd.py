"""Tests for RLCD proper-scoring losses (M4-T3, CAL-01 / TRAIN-03)."""

from __future__ import annotations

import pytest
from training.rlcd import (
    PROPER_RULES,
    categorical_loss,
    noul_loss,
    rlcd_loss,
    rlcd_reward,
)


@pytest.mark.parametrize("rule", PROPER_RULES)
def test_loss_is_non_negative(rule: str) -> None:
    assert noul_loss(0.3, 1.0, rule=rule) >= 0.0
    assert categorical_loss({"a": 0.3, "b": 0.7}, "b", rule=rule) >= 0.0


@pytest.mark.parametrize("rule", PROPER_RULES)
def test_noul_loss_is_proper(rule: str) -> None:
    """Expected loss over the realized outcome is minimized at the true probability."""
    truth = 0.7
    grid = [index / 100 for index in range(1, 100)]
    expected = {
        p: truth * noul_loss(p, 1.0, rule=rule) + (1 - truth) * noul_loss(p, 0.0, rule=rule)
        for p in grid
    }
    best = min(expected, key=lambda p: expected[p])
    assert abs(best - truth) <= 0.02


@pytest.mark.parametrize("rule", PROPER_RULES)
def test_categorical_loss_is_proper(rule: str) -> None:
    truth = {"a": 0.2, "b": 0.3, "c": 0.5}
    candidates = [
        {"a": 0.2, "b": 0.3, "c": 0.5},  # correct
        {"a": 0.5, "b": 0.25, "c": 0.25},  # wrong
        {"a": 0.0, "b": 0.5, "c": 0.5},  # wrong
    ]

    def expected(distribution: dict[str, float]) -> float:
        return sum(truth[key] * categorical_loss(distribution, key, rule=rule) for key in truth)

    losses = [expected(candidate) for candidate in candidates]
    assert losses[0] < losses[1]
    assert losses[0] < losses[2]


@pytest.mark.parametrize("rule", PROPER_RULES)
def test_perfect_prediction_is_zero(rule: str) -> None:
    assert noul_loss(1.0, 1.0, rule=rule) == pytest.approx(0.0, abs=1e-9)
    assert categorical_loss({"a": 1.0, "b": 0.0}, "a", rule=rule) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_is_numerically_stable() -> None:
    assert noul_loss(0.0, 0.0) == 0.0  # brier
    assert noul_loss(0.0, 0.0, rule="log") >= 0.0
    assert noul_loss(0.0, 1.0, rule="log") < float("inf")
    assert categorical_loss({"a": 0.0, "b": 1.0}, "b", rule="log") < float("inf")


def test_unnormalized_probabilities_are_normalized() -> None:
    assert categorical_loss({"a": 1.0, "b": 3.0}, "b") == pytest.approx(0.125, abs=1e-9)


def test_rlcd_dispatch_and_reward() -> None:
    loss = rlcd_loss("choice", {"a": 0.2, "b": 0.8}, "b")
    assert rlcd_reward("choice", {"a": 0.2, "b": 0.8}, "b") == -loss
    assert rlcd_loss("noul", 0.6, 1.0) == pytest.approx(0.16)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        categorical_loss({"a": 0.5, "b": 0.5}, "missing")
    with pytest.raises(ValueError):
        categorical_loss({"a": 1.0}, "a", rule="nope")
    with pytest.raises(ValueError):
        noul_loss(0.5, 1.0, rule="nope")
    with pytest.raises(ValueError):
        rlcd_loss("unknown", 0.5, 1.0)
    with pytest.raises(TypeError):
        rlcd_loss("noul", {"a": 1.0}, 1.0)
    with pytest.raises(TypeError):
        rlcd_loss("choice", 0.5, "a")
