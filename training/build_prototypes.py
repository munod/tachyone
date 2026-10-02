"""Build the training-state prototype bank — P3's evidence signal (B-13, JB-10).

``strength = max_k cos(state_embedding, centroid_k)`` over k-means centroids of the
checkpoint's own training states is the signal that separates "this input looks like what
the model was trained on" from "it does not" (spec, *P3 design*: AUC 1.000 measured). It
feeds the instance-dependent ``noul`` confidence map, so the bank and the map must be built
from the same basis — the fit embeds this asset into ``confidence_calibration.json`` and
records its sha256, which is what makes the pair atomic at runtime.

Spherical k-means (centroids renormalized every iteration) because the runtime compares
with a cosine, and the states are encoded through the same loader the runtime uses — trunk
plus adapter, 512-token window — so the centroids live in the *inference* space.

    uv run python -m training.build_prototypes \
        --data data/train_en_domains.jsonl --out checkpoints/en/state_prototypes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tachyone.router import state_text

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = _ROOT / "data" / "train_en_domains.jsonl"
DEFAULT_OUT = _ROOT / "checkpoints" / "en" / "state_prototypes.json"

#: Asset schema version; the runtime refuses a version it does not know.
PROTOTYPE_VERSION = 1


class PrototypeError(RuntimeError):
    """A bank that cannot be built or would not be usable by the runtime."""


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0.0:
        raise PrototypeError("cannot normalize a zero vector")
    return [value / norm for value in vector]


def kmeans(
    vectors: Sequence[Sequence[float]],
    k: int,
    *,
    seed: int = 0,
    iterations: int = 25,
) -> list[list[float]]:
    """Spherical k-means: assign by dot product, renormalize each centroid every round.

    Deterministic for a given input, ``k`` and ``seed`` (L-003 discipline: one seeded RNG,
    no draw that can correlate with anything else). Raises when ``k`` exceeds the number of
    distinct rows, so a mis-sized bank fails at build time instead of shipping duplicates.
    """
    if k < 1:
        raise PrototypeError(f"k must be >= 1, got {k}")
    if len(vectors) < k:
        raise PrototypeError(f"k={k} needs at least {len(vectors)} vectors")
    points = [_normalize(vector) for vector in vectors]
    # k-means++ seeding: sample the next start with probability proportional to how badly
    # the already-chosen centroids miss it (1 - best cosine). Picking k rows uniformly can
    # land two starts in one cluster and leave another empty — measured, seed 0.
    rng = random.Random(seed)
    first = rng.randrange(len(points))
    chosen = [first]
    while len(chosen) < k:
        misses = [
            max(0.0, 1.0 - max(_dot(points[index], points[other]) for other in chosen))
            for index in range(len(points))
        ]
        total = sum(misses)
        if total <= 0.0:
            chosen.extend(
                index for index in rng.sample(range(len(points)), k) if index not in chosen
            )
            break
        pick = rng.random() * total
        running = 0.0
        selected = len(points) - 1
        for index, miss in enumerate(misses):
            running += miss
            if running >= pick:
                selected = index
                break
        chosen.append(selected)
    centroids = [list(points[index]) for index in chosen[:k]]
    assignment = [-1] * len(points)
    for _ in range(iterations):
        changed = False
        labels = [max(range(k), key=lambda c: _dot(point, centroids[c])) for point in points]
        for index, label in enumerate(labels):
            if label != assignment[index]:
                assignment[index] = label
                changed = True
        for centroid_index in range(k):
            members = [
                points[index] for index, label in enumerate(labels) if label == centroid_index
            ]
            if not members:
                continue  # an empty cluster keeps its previous centroid (never NaN)
            mean = [sum(values) for values in zip(*members, strict=True)]
            centroids[centroid_index] = _normalize(mean)
        if not changed:
            break
    return centroids


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_states(path: Path) -> list[str]:
    """The state text of every record in ``path`` (the bank's basis)."""
    states: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                states.append(state_text(json.loads(line)["state"]))
    if not states:
        raise PrototypeError(f"no records in {path}")
    return states


def build_asset(
    states: Sequence[str],
    *,
    encode: Any,
    k: int,
    seed: int,
    data_path: Path,
    model_id: str,
    adapter: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Encode ``states`` with the runtime's own encoder and reduce them to a bank."""
    chosen = list(states[:limit] if limit else states)
    embeddings: list[list[float]] = []
    for index in range(0, len(chosen), 32):
        embeddings.extend(encode(chosen[index : index + 32]))
    dim = len(embeddings[0])
    if any(len(vector) != dim for vector in embeddings):
        raise PrototypeError("the encoder returned ragged embeddings")
    centroids = kmeans(embeddings, k, seed=seed)
    asset: dict[str, Any] = {
        "version": PROTOTYPE_VERSION,
        "metric": "cosine",
        "k": len(centroids),
        "dim": dim,
        "seed": seed,
        "source": {
            "data": str(data_path),
            "records": len(chosen),
            "sha256": sha256_of(data_path),
        },
        "model": {"id": model_id, "adapter": adapter},
        "centroids": [[round(value, 5) for value in centroid] for centroid in centroids],
    }
    validate_asset(asset)
    return asset


def validate_asset(payload: Any) -> None:
    """Reject anything the runtime could not use (it loads this asset without a guard rail)."""
    if not isinstance(payload, dict):
        raise PrototypeError("prototype asset must be a JSON object")
    if payload.get("version") != PROTOTYPE_VERSION:
        raise PrototypeError(f"unsupported prototype asset version: {payload.get('version')!r}")
    if payload.get("metric") != "cosine":
        raise PrototypeError(f"unsupported metric: {payload.get('metric')!r}")
    centroids = payload.get("centroids")
    dim = payload.get("dim")
    k = payload.get("k")
    if not isinstance(centroids, list) or not centroids:
        raise PrototypeError("centroids must be a non-empty list")
    if k != len(centroids):
        raise PrototypeError(f"k={k!r} does not match {len(centroids)} centroids")
    if not isinstance(dim, int) or dim < 1:
        raise PrototypeError(f"dim must be a positive int, got {dim!r}")
    for index, centroid in enumerate(centroids):
        if not isinstance(centroid, list) or len(centroid) != dim:
            raise PrototypeError(f"centroid {index} is not a {dim}-vector")
        if not all(isinstance(value, (int, float)) for value in centroid):
            raise PrototypeError(f"centroid {index} contains a non-numeric value")
        norm = sum(float(value) ** 2 for value in centroid) ** 0.5
        if not 0.99 <= norm <= 1.01:
            raise PrototypeError(f"centroid {index} is not unit-length (norm {norm:.4f})")


def save_asset(asset: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset, indent=1, sort_keys=True), encoding="utf-8")


def build(
    *,
    data_path: str | Path = DEFAULT_DATA,
    out_path: str | Path = DEFAULT_OUT,
    k: int = 32,
    seed: int = 0,
    model_id: str | None = None,
    adapter: str | None = None,
    device: str = "auto",
    max_len: int = 512,
    limit: int | None = None,
) -> dict[str, Any]:
    """Encode the training states and write the bank; return the asset."""
    from training.predict import _build_model, _default_model_id

    data = Path(data_path)
    if not data.exists():
        raise PrototypeError(f"missing basis file: {data}")
    adapter_dir = adapter
    resolved_model = model_id or _default_model_id(adapter_dir or "")
    model = _build_model(resolved_model, adapter_dir, device=device, max_len=max_len)
    asset = build_asset(
        read_states(data),
        encode=model._encode,
        k=k,
        seed=seed,
        data_path=data,
        model_id=resolved_model,
        adapter=adapter_dir,
        limit=limit,
    )
    save_asset(asset, out_path)
    return asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-build-prototypes", description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter", default=None, help="the checkpoint dir (trunk + adapter)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="encode only the first N states")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    asset = build(
        data_path=args.data,
        out_path=args.out,
        k=args.k,
        seed=args.seed,
        model_id=args.model_id,
        adapter=args.adapter,
        device=args.device,
        max_len=args.max_len,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "k": asset["k"],
                "dim": asset["dim"],
                "records": asset["source"]["records"],
                "source_sha256": asset["source"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
