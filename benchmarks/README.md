# Benchmarks

Evaluation and reproduction harnesses for jeba. Per the testing strategy
(`docs/testing.md`), benchmark runs are serialized (no parallel execution) and artifacts
record hardware, versions, seed, and the exact command.

## Current

- **Accuracy / ECE / latency** — `training/evaluate.py` reports per-primitive and per-language
  metrics over a JSONL of labeled records:

  ```bash
  uv run python -m training.evaluate --data data/eval.jsonl --out benchmarks/results/eval.json --backend fake
  ```

- **Fast path (CUDA graphs)** — `benchmarks/fast_path.py` compares the stock encoder forward with
  the `JEBA_FAST` CUDA-graph path (latency p50/p95, throughput, embedding parity). Requires the
  `train` extra and a CUDA device:

  ```bash
  uv run python -m benchmarks.fast_path --data data/eval_multi.jsonl \
    --model-id jhu-clsp/mmBERT-base --adapter checkpoints/multi \
    --out benchmarks/results/fast_path.json
  ```

## Planned (M6)

- `public_probes.py` — MASSIVE / XNLI / typed-decisions, evaluation only.
- `latency.py` / `throughput.py` — batch-size sweeps per backend.
- `calibration.py` — ECE curves before/after temperature fitting.

Results are written under `benchmarks/results/` (gitignored) and summarized in `BENCHMARKS.md`.
