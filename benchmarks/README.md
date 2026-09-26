# Benchmarks

Evaluation and reproduction harnesses for Tachyone. Per the testing strategy
(`docs/testing.md`), benchmark runs are serialized (no parallel execution) and artifacts
record hardware, versions, seed, and the exact command.

## Current

- **Accuracy / ECE / latency** — `training/evaluate.py` reports per-primitive and per-language
  metrics over a JSONL of labeled records:

  ```bash
  uv run python -m training.evaluate --data data/eval_multi.jsonl \
    --out benchmarks/results/eval_multi.json --backend fake
  ```

- **Report renderer** — `benchmarks/report.py` renders committed JSON artifacts into
  `benchmarks/report.md`:

  ```bash
  uv run python -m benchmarks.report --entry encoder=benchmarks/results/eval.json \
    --out benchmarks/report.md
  ```

- **Fast path (CUDA graphs)** — `benchmarks/fast_path.py` compares the stock encoder forward with
  the `TACHYONE_FAST` CUDA-graph path (latency p50/p95, throughput, embedding parity). Requires the
  `train` extra and a CUDA device:

  ```bash
  uv run python -m benchmarks.fast_path --data data/eval_multi.jsonl \
    --model-id jhu-clsp/mmBERT-base --adapter checkpoints/multi \
    --out benchmarks/results/fast_path.json
  ```

- **Head-to-head comparison** — `benchmarks.compare` answers the same rows with three engines
  through one set of metrics: Tachyone (its wire path), the open peer scorer
  `pngwn/system-one-qwen3.5-4b-scorer` (loaded from a local download through *its own*
  `system_one.py` — nothing third-party is vendored here), and any OpenAI-compatible server such
  as `llama-server` (through Tachyone's `llm` backend, so schema failures count against the
  engine that produced them).

  One process per engine, then render. Benchmark runs are **serialized** — do not overlap them,
  or every latency number is polluted:

  ```bash
  # 1) the public probe (their distribution, 64 rows per task family)
  uv pip install pyarrow                       # parquet reader, imported lazily
  hf download pngwn/system-one-decisions --repo-type dataset --local-dir data/system-one-decisions

  uv run python -m benchmarks.compare run --engine tachyone --per-task 64 \
    --out benchmarks/results/compare_tachyone.json

  uv run python -m benchmarks.compare run --engine systemone --per-task 64 \
    --systemone-script /path/to/scorer/system_one.py \
    --systemone-base /path/to/base --systemone-adapter /path/to/scorer \
    --out benchmarks/results/compare_systemone.json

  llama-server -m small.gguf --port 8081 -ngl 99 --temp 0 --seed 42 &
  uv run python -m benchmarks.compare run --engine llm \
    --llm-base-url http://127.0.0.1:8081/v1 --llm-model small --per-task 16 \
    --out benchmarks/results/compare_llm_small.json

  # 2) Tachyone's own distribution (same engines, our eval records)
  uv run python -m benchmarks.compare run --engine tachyone --format jsonl \
    --data data/eval_en.jsonl --val-data data/train_en.jsonl --per-task 64 \
    --out benchmarks/results/compare_home_tachyone.json

  # 3) render
  uv run python -m benchmarks.compare render benchmarks/results/compare_*.json \
    --out benchmarks/compare_table.md
  ```

  Each artifact records the environment, the fitted temperature, the exact command, peak RSS/VRAM
  (of the process that did the inference — for the LLM that is `llama-server`, not the client) and
  a per-task breakdown.

## Not yet delivered

- `public_probes.py` — MASSIVE / XNLI / typed-decisions, evaluation only.
- `latency.py` / `throughput.py` — batch-size sweeps per backend.
- `calibration.py` — ECE curves before/after temperature fitting.

Results are written under `benchmarks/results/` (gitignored) and summarized in
[`benchmarks/report.md`](report.md).
