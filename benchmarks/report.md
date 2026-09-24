# jeba Benchmark Report

> **Status: scaffold.** The harness and reproduction commands are in place; the numbers on
> this page are **pending the RTX 3060 training run** (`uv sync --extra train`) and real
> encoder/ONNX weights. Regenerate with `benchmarks/report.py` once results exist.

## Environment

Captured automatically by `training.config.environment()` when the report is generated
(Python, platform, jeba, optional torch/transformers/peft versions, git commit).

## Reproduce

```bash
# 1. Build a labeled evaluation set (deterministic, no GPU needed)
uv run python -m training.generate_data --per-type 2000 --out data/eval.jsonl

# 2. Evaluate each backend (produces benchmarks/results/*.json)
uv run python -m training.evaluate --data data/eval.jsonl --out benchmarks/results/encoder.json --backend encoder
uv run python -m training.evaluate --data data/eval.jsonl --out benchmarks/results/onnx.json    --backend onnx
uv run python -m training.evaluate --data data/eval.jsonl --out benchmarks/results/fake.json     --backend fake

# 3. Render this report
uv run python -m benchmarks.report \
  --entry encoder=benchmarks/results/encoder.json \
  --entry onnx=benchmarks/results/onnx.json \
  --entry fake=benchmarks/results/fake.json \
  --note "Fill in the hardware and commit for the published run." \
  --out benchmarks/report.md
```

## Results

_No published run yet._ Metrics below are reported per scope (`overall` and each primitive:
`noul`, `choice`, `score`), with accuracy, expected calibration error (ECE), and p50/p95
latency in milliseconds.

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| pending | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Public probes (M6 follow-up)

Reproducible probes against **MASSIVE**, **XNLI**, and **typed-decisions** (evaluation only)
will be added under `benchmarks/` and must record the exact command, seed, and hardware in the
generated artifact.
