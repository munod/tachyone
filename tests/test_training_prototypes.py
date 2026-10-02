"""Tests for the prototype bank builder (B-13 P3, JB-10).

Pure Python: the k-means, the asset schema and the provenance are exercised on synthetic
vectors and a fake encoder. The end-to-end build (real trunk + adapter, GPU) is skipped
unless the machine has torch and a CUDA device — it is run on the training box, and its
*separation* property is asserted there by the smoke test at the bottom.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from training.build_prototypes import (
    DEFAULT_DATA,
    PrototypeError,
    build_asset,
    kmeans,
    read_states,
    save_asset,
    validate_asset,
)


def _clustered() -> tuple[list[list[float]], list[int]]:
    """Three tight unit clusters in 8 dimensions with their labels."""
    centers = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    vectors: list[list[float]] = []
    labels: list[int] = []
    for label, center in enumerate(centers):
        for offset in range(10):
            vector = list(center)
            vector[label] += 0.01 * offset
            vector[(label + 1) % 8] = 0.02 * offset
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
            labels.append(label)
    return vectors, labels


def _asset(dim: int = 4) -> dict[str, Any]:
    return {
        "version": 1,
        "metric": "cosine",
        "k": 2,
        "dim": dim,
        "seed": 0,
        "source": {"data": "x", "records": 10, "sha256": "0" * 64},
        "model": {"id": "m", "adapter": None},
        "centroids": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    }


# ------------------------------------------------------------------------- k-means


def test_kmeans_is_deterministic_and_unit_length() -> None:
    vectors, _ = _clustered()
    first = kmeans(vectors, 3, seed=0)
    second = kmeans(vectors, 3, seed=0)
    assert first == second
    assert len(first) == 3
    for centroid in first:
        assert abs(sum(value * value for value in centroid) - 1.0) < 1e-9


def test_kmeans_lands_one_centroid_per_cluster() -> None:
    vectors, labels = _clustered()
    centroids = kmeans(vectors, 3, seed=0)
    winners: set[int] = set()
    for centroid in centroids:
        nearest = max(
            range(len(vectors)),
            key=lambda i: sum(a * b for a, b in zip(centroid, vectors[i], strict=True)),
        )
        winners.add(labels[nearest])
    assert winners == {0, 1, 2}, f"centroids collapsed onto {winners}"


def test_kmeans_seeds_are_independent_but_always_valid() -> None:
    vectors, _ = _clustered()
    for seed in (0, 1, 7):
        centroids = kmeans(vectors, 3, seed=seed)
        assert len(centroids) == 3
        for centroid in centroids:
            assert abs(sum(value * value for value in centroid) - 1.0) < 1e-9
        # whatever the seed, no two centroids collapse onto the same point
        for i, left in enumerate(centroids):
            for right in centroids[i + 1 :]:
                dot = sum(a * b for a, b in zip(left, right, strict=True))
                assert dot < 0.999, f"identical centroids under seed {seed}"


def test_kmeans_keeps_empty_clusters_alive() -> None:
    """Duplicate rows leave a cluster empty: the centroid must survive, not become NaN."""
    vectors = [[1.0, 0.0]] * 6 + [[0.0, 1.0]] * 6
    centroids = kmeans(vectors, 8, seed=3, iterations=5)
    assert len(centroids) == 8
    for centroid in centroids:
        assert all(math.isfinite(value) for value in centroid)
        assert abs(sum(value * value for value in centroid) - 1.0) < 1e-9


def test_kmeans_validates_k() -> None:
    with pytest.raises(PrototypeError, match="k must be"):
        kmeans([[1.0, 0.0]], 0)
    with pytest.raises(PrototypeError, match="needs at least"):
        kmeans([[1.0, 0.0], [0.0, 1.0]], 5)


# ------------------------------------------------------------------------- the asset


def test_validate_asset_accepts_a_well_formed_bank() -> None:
    validate_asset(_asset())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda a: a.update(version=99), "version"),
        (lambda a: a.update(metric="dot"), "metric"),
        (lambda a: a.update(k=3), "does not match"),
        (lambda a: a.update(dim=5), "not a 5-vector"),
        (lambda a: a["centroids"][0].append(0.0), "not a 4-vector"),
        (lambda a: a["centroids"].clear(), "non-empty"),
        (lambda a: a.update(centroids="nope"), "non-empty"),
        (lambda a: a.update(dim=0), "dim must be"),
        (lambda a: a["centroids"][1].__setitem__(0, "x"), "non-numeric"),
        (lambda a: a["centroids"][1].__setitem__(0, 7.0), "unit-length"),
    ],
)
def test_validate_asset_rejects_what_the_runtime_cannot_use(mutate: Any, match: str) -> None:
    asset = _asset()
    mutate(asset)
    with pytest.raises(PrototypeError, match=match):
        validate_asset(asset)
    with pytest.raises(PrototypeError, match="JSON object"):
        validate_asset([1, 2, 3])


def test_save_asset_round_trips(tmp_path: Path) -> None:
    asset = _asset()
    path = tmp_path / "nested" / "state_prototypes.json"
    save_asset(asset, path)
    assert json.loads(path.read_text(encoding="utf-8")) == asset
    validate_asset(json.loads(path.read_text(encoding="utf-8")))


def test_build_asset_uses_the_fake_encoder_and_records_provenance(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(json.dumps({"state": f"state {i}"}) + "\n" for i in range(40)),
        encoding="utf-8",
    )

    def encode(texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text) % 5), float(i), 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
            for i, text in enumerate(texts)
        ]

    asset = build_asset(
        read_states(data),
        encode=encode,
        k=4,
        seed=2,
        data_path=data,
        model_id="fake",
        adapter=None,
        limit=32,
    )
    validate_asset(asset)
    assert asset["k"] == 4
    assert asset["dim"] == 8
    assert asset["source"]["records"] == 32, "limit is part of the provenance"
    assert len(asset["source"]["sha256"]) == 64
    assert asset["source"]["data"].endswith("train.jsonl")
    assert asset["model"]["id"] == "fake"
    assert all(value == round(value, 5) for c in asset["centroids"] for value in c)


def test_build_asset_rejects_ragged_embeddings(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"state": "one"}) + "\n", encoding="utf-8")
    with pytest.raises(PrototypeError, match="ragged"):
        build_asset(
            read_states(data),
            encode=lambda texts: [[1.0, 0.0], [1.0, 0.0, 0.0]],
            k=1,
            seed=0,
            data_path=data,
            model_id="fake",
            adapter=None,
        )


def test_read_states_flattens_structured_states(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"state": "plain"}) + "\n" + json.dumps({"state": {"a": "nested"}}) + "\n",
        encoding="utf-8",
    )
    assert read_states(path) == ["plain", "nested"]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(PrototypeError, match="no records"):
        read_states(empty)


def test_committed_basis_is_the_training_set() -> None:
    assert DEFAULT_DATA.name == "train_en_domains.jsonl", "P3 builds on the real training set"


def test_real_build_needs_a_model(tmp_path: Path) -> None:
    """End-to-end on the training box: torch + CUDA + the committed basis."""
    pytest.importorskip("torch")
    if not DEFAULT_DATA.exists():
        pytest.skip("training data not present")
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("no CUDA device")
    from training.build_prototypes import build

    asset = build(
        out_path=tmp_path / "state_prototypes.json",
        k=8,
        seed=0,
        adapter="checkpoints/en",
        limit=512,
    )
    validate_asset(asset)
    assert asset["dim"] > 100, "the real ModernBERT-large embedding is wide"
    assert asset["k"] == 8
