"""Unit tests for confidence and temperature calibration (M3-T3, CAL-03 / CAL-05)."""

from __future__ import annotations

import pytest

from tachyone.calibration import (
    apply_temperature,
    confidence,
    expected_calibration_error,
    fit_temperature,
    margin,
    normalized_entropy,
    parse_temperature_report,
)


def test_confidence_is_selected_mass() -> None:
    assert confidence({"a": 0.8, "b": 0.2}) == pytest.approx(0.8)


def test_confidence_uniform_is_one_over_n() -> None:
    assert confidence({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(0.25)


def test_confidence_single_option_is_one() -> None:
    assert confidence({"only": 1.0}) == 1.0


def test_confidence_normalizes_unnormalized_input() -> None:
    assert confidence({"a": 3.0, "b": 1.0}) == pytest.approx(0.75)


def test_confidence_empty_or_zero() -> None:
    assert confidence({}) == 0.0
    assert confidence({"a": 0.0, "b": 0.0}) == 0.0


def test_confidence_monotone_with_concentration() -> None:
    flat = confidence({"a": 0.5, "b": 0.5})
    sharp = confidence({"a": 0.9, "b": 0.1})
    assert flat < sharp <= 1.0


def test_apply_temperature_identity() -> None:
    original = {"a": 0.7, "b": 0.3}
    assert apply_temperature(original, 1.0) == pytest.approx(original)


def test_apply_temperature_softens_and_sharpens() -> None:
    softened = apply_temperature({"a": 0.9, "b": 0.1}, 2.0)
    sharpened = apply_temperature({"a": 0.9, "b": 0.1}, 0.5)
    assert softened["a"] < 0.9 < sharpened["a"]
    assert sum(softened.values()) == pytest.approx(1.0)
    assert sum(sharpened.values()) == pytest.approx(1.0)


def test_apply_temperature_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        apply_temperature({"a": 1.0}, 0.0)


def test_ece_zero_for_perfect_calibration() -> None:
    confidences = [0.5] * 10
    correct = [True, False] * 5
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_overconfidence() -> None:
    confidences = [0.9] * 10
    correct = [True] * 6 + [False] * 4
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.3, abs=1e-9)


def test_ece_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])


def test_fit_temperature_reduces_ece_on_overconfident_set() -> None:
    samples = [{"a": 0.9, "b": 0.1}] * 100
    correct = [True] * 60 + [False] * 40
    baseline = expected_calibration_error([0.9] * 100, correct)
    fitted = fit_temperature(samples, correct)
    assert fitted.value != 1.0
    assert fitted.ece < baseline


def test_normalized_entropy_uniform_is_one() -> None:
    assert normalized_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)


def test_normalized_entropy_one_hot_is_zero() -> None:
    assert normalized_entropy({"a": 1.0, "b": 0.0, "c": 0.0}) == pytest.approx(0.0)


def test_normalized_entropy_empty_or_single() -> None:
    assert normalized_entropy({}) == 0.0
    assert normalized_entropy({"only": 1.0}) == 0.0
    assert normalized_entropy({"a": 0.0, "b": 0.0}) == 0.0


def test_normalized_entropy_normalizes_unnormalized_input() -> None:
    assert normalized_entropy({"a": 3.0, "b": 1.0}) == pytest.approx(
        normalized_entropy({"a": 0.75, "b": 0.25})
    )


def test_normalized_entropy_between_zero_and_one() -> None:
    value = normalized_entropy({"a": 0.7, "b": 0.2, "c": 0.1})
    assert 0.0 < value < 1.0


def test_margin_is_top_two_gap() -> None:
    assert margin({"a": 0.7, "b": 0.2, "c": 0.1}) == pytest.approx(0.5)


def test_margin_tie_is_zero() -> None:
    assert margin({"a": 0.5, "b": 0.5}) == 0.0


def test_margin_dominant_is_high() -> None:
    assert margin({"a": 0.95, "b": 0.05}) == pytest.approx(0.9)


def test_margin_empty_or_single() -> None:
    assert margin({}) == 0.0
    assert margin({"only": 1.0}) == 0.0
    assert margin({"a": 0.0, "b": 0.0}) == 0.0


def test_margin_normalizes_unnormalized_input() -> None:
    assert margin({"a": 3.0, "b": 1.0}) == pytest.approx(0.5)


def test_uncertainty_helpers_agree_on_monotonicity() -> None:
    flat = {"a": 0.5, "b": 0.5}
    sharp = {"a": 0.9, "b": 0.1}
    assert normalized_entropy(flat) > normalized_entropy(sharp)
    assert margin(flat) < margin(sharp)


def test_parse_temperature_report_flattens_legacy_and_per_language() -> None:
    report = {
        "per_primitive": {
            "choice": {"temperature": 2.0, "by_language": {"pt": {"temperature": 3.0}}},
            "score": {"temperature": 0.5},
        }
    }
    parsed = parse_temperature_report(report)
    assert parsed["choice"] == 2.0
    assert parsed["choice:pt"] == 3.0
    assert parsed["score"] == 0.5
    assert "score:pt" not in parsed


def test_parse_temperature_report_tolerates_bad_shapes() -> None:
    assert parse_temperature_report({"per_primitive": []}) == {}
    assert parse_temperature_report(["nope"]) == {}  # type: ignore[arg-type]
    assert parse_temperature_report({"per_primitive": {"choice": []}}) == {}
