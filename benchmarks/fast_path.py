"""Micro-benchmark: stock encoder forward vs the CUDA-graph fast path (B-2, NFR-P01).

Loads the same checkpoint twice (stock and ``fast=True``) and reports per-question latency
(p50/p95), throughput across batch sizes, and embedding parity between the two paths. Requires
the ``train`` extra and a CUDA device; without one it exits with a clear message. Pure helpers
are unit-tested without a GPU. Record the output in ``benchmarks/report.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from jeba.backends.encoder import EncoderModel, load_encoder
from jeba.fast import cuda_available
from jeba.primitives import NoulAnswer
from jeba.router import CheckpointInfo, state_text
from training.evaluate import EvalExample, load_examples
from training.predict import _DEFAULT_MODELS_DIR

#: A batch of texts encoded to dense vectors.
type Encode = Callable[[list[str]], list[list[float]]]

#: Batch sizes swept for the throughput measurement.
_BATCH_SIZES: tuple[int, ...] = (1, 4, 16)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _parity(left: list[list[float]], right: list[list[float]]) -> float:
    """Maximum absolute element difference between two embedding batches."""
    if len(left) != len(right):
        raise ValueError("embedding batches differ in length")
    max_diff = 0.0
    for first, second in zip(left, right, strict=True):
        if len(first) != len(second):
            raise ValueError("embedding vectors differ in dimension")
        for a, b in zip(first, second, strict=True):
            max_diff = max(max_diff, abs(a - b))
    return max_diff


def _time(encode: Encode, texts: Sequence[str], *, repeats: int) -> dict[str, float | int]:
    latencies: list[float] = []
    for _ in range(max(1, repeats)):
        for text in texts:
            start = time.perf_counter()
            encode([text])
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        "n": len(latencies),
        "p50": round(_percentile(latencies, 0.50), 4),
        "p95": round(_percentile(latencies, 0.95), 4),
    }


def _throughput(encode: Encode, texts: Sequence[str], *, batch_size: int) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        began = time.perf_counter()
        encode(chunk)
        total += time.perf_counter() - began
        count += len(chunk)
    return round(count / total, 3) if total > 0 else 0.0


def _warmup(encode: Encode, texts: Sequence[str], batch_sizes: Sequence[int]) -> None:
    """Touch every shape first so CUDA-graph capture is not charged to the timed loops."""
    for size in batch_sizes:
        for start in range(0, len(texts), size):
            encode(list(texts[start : start + size]))


def _answer_distribution(encode: Encode, example: EvalExample) -> list[float]:
    answer = EncoderModel(encode).answer_state(example.state, {"q": example.question})["q"]
    if isinstance(answer, NoulAnswer):
        return [answer.noul, 1.0 - answer.noul]
    return [float(value) for value in answer.probabilities.values()]


def _answer_parity(
    examples: Sequence[EvalExample], left: Encode, right: Encode
) -> tuple[float, int]:
    """Max abs probability difference and number of changed top labels between two paths."""
    max_diff = 0.0
    flips = 0
    for example in examples:
        first = _answer_distribution(left, example)
        second = _answer_distribution(right, example)
        for a, b in zip(first, second, strict=True):
            max_diff = max(max_diff, abs(a - b))
        if first and second:
            left_top = max(range(len(first)), key=first.__getitem__)
            right_top = max(range(len(second)), key=second.__getitem__)
            flips += int(left_top != right_top)
    return max_diff, flips


def run(
    *,
    data: str | Path,
    model_id: str,
    adapter_dir: str | None = None,
    max_len: int = 512,
    limit: int | None = 200,
    device: str = "auto",
    repeats: int = 1,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark stock vs fast encoding and return the report."""
    if not cuda_available():
        raise SystemExit("fast-path benchmark needs a CUDA device (and the train extra)")
    examples = load_examples(data, limit=limit)
    texts = [state_text(example.state) for example in examples]
    texts = [text for text in texts if text]
    if not texts:
        raise SystemExit("no non-empty states to benchmark")

    info = CheckpointInfo(
        id="bench",
        languages=["*"],
        context=max_len,
        size_params=0,
        base_model=model_id,
        adapter=adapter_dir,
    )
    stock = load_encoder(info, models_dir=_DEFAULT_MODELS_DIR, device=device)
    fast = load_encoder(info, models_dir=_DEFAULT_MODELS_DIR, device=device, fast=True)

    # Warm both paths (single and batched shapes) before timing steady state.
    warm_sizes = (1, *_BATCH_SIZES)
    _warmup(stock, texts, warm_sizes)
    _warmup(fast, texts, warm_sizes)

    sample = texts[: min(8, len(texts))]
    answer_sample = examples[: min(16, len(examples))]
    answer_diff, answer_flips = _answer_parity(answer_sample, stock, fast)
    report: dict[str, Any] = {
        "n": len(texts),
        "embedding_parity_max_abs": round(_parity(stock(sample), fast(sample)), 6),
        "answer_parity_max_abs": round(answer_diff, 6),
        "answer_top_flips": answer_flips,
        "answer_sample": len(answer_sample),
        "stock": {"latency_ms": _time(stock, texts, repeats=repeats)},
        "fast": {"latency_ms": _time(fast, texts, repeats=repeats)},
        "throughput_items_per_s": {
            "stock": {
                str(size): _throughput(stock, texts, batch_size=size) for size in _BATCH_SIZES
            },
            "fast": {str(size): _throughput(fast, texts, batch_size=size) for size in _BATCH_SIZES},
        },
    }
    stock_p50 = float(report["stock"]["latency_ms"]["p50"])
    fast_p50 = float(report["fast"]["latency_ms"]["p50"])
    report["speedup_p50"] = round(stock_p50 / fast_p50, 3) if fast_p50 else 0.0

    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = (
        "n",
        "embedding_parity_max_abs",
        "answer_parity_max_abs",
        "answer_top_flips",
        "speedup_p50",
    )
    print(json.dumps({key: report[key] for key in summary}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-fast-path-benchmark", description=__doc__)
    parser.add_argument("--data", required=True, help="records JSONL providing states")
    parser.add_argument("--model-id", required=True, help="base encoder model id")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=None, help="output report JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    run(
        data=args.data,
        model_id=args.model_id,
        adapter_dir=args.adapter,
        max_len=args.max_len,
        limit=args.limit,
        device=args.device,
        repeats=args.repeats,
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
