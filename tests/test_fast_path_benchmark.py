"""Tests for the fast-path micro-benchmark helpers (B-2, NFR-P01)."""

from __future__ import annotations

import pytest

from benchmarks.fast_path import _parity, _percentile, _throughput


def test_percentile_handles_bounds_and_empty() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    assert _percentile(values, 0.5) == 3.0
    assert _percentile([], 0.5) == 0.0


def test_parity_reports_max_abs_difference() -> None:
    left = [[1.0, 2.0], [3.0, 4.0]]
    right = [[1.0, 2.5], [3.0, 4.0]]
    assert _parity(left, right) == pytest.approx(0.5)
    assert _parity(left, left) == 0.0


def test_parity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        _parity([[1.0]], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        _parity([[1.0, 2.0]], [[1.0]])


def test_throughput_counts_items_per_second() -> None:
    def encode(texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    assert _throughput(encode, ["a", "b", "c", "d"], batch_size=2) > 0.0
