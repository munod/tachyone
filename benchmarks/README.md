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
  or every latency number is polluted.

  > **Warning — `--option-batch` above 4 OOMs a 12 GB card.** The peer scorer loads ~8.9 GB of
  > bf16 weights; with the **default** `--option-batch 16` a 77-option question pushed the
  > process past 11.6 GiB and died with `torch.OutOfMemoryError`. **Use `--option-batch 4`**
  > (measured peak VRAM: 8.9 GiB) and, if it still trips,
  > `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The peer's own card reports latency at
  > 4 options; our table mixes option counts up to 77.

  ```bash
  # prerequisites
  uv pip install pyarrow                            # parquet reader, imported lazily
  hf download pngwn/system-one-decisions --repo-type dataset --local-dir data/system-one-decisions
  hf download Qwen/Qwen3.5-4B-Base       --local-dir <peer>/base
  hf download pngwn/system-one-qwen3.5-4b-scorer    --local-dir <peer>/scorer

  # one engine at a time
  uv run python -m benchmarks.compare run --engine tachyone --per-task 64 \
    --out benchmarks/results/compare_tachyone.json

  uv run python -m benchmarks.compare run --engine systemone --per-task 64 \
    --systemone-script <peer>/scorer/system_one.py \
    --systemone-base <peer>/base --systemone-adapter <peer>/scorer \
    --option-batch 4 \
    --out benchmarks/results/compare_systemone.json

  llama-server -m small.gguf --port 8081 -ngl 99 --temp 0 --seed 42 -c 8192 --alias small &
  uv run python -m benchmarks.compare run --engine llm \
    --llm-base-url http://127.0.0.1:8081/v1 --llm-model small --llm-timeout 300 --per-task 4 \
    --out benchmarks/results/compare_llm_small.json

  # Tachyone's own distribution (same engines, our eval records)
  uv run python -m benchmarks.compare run --engine tachyone --format jsonl \
    --data data/eval_en.jsonl --val-data data/train_en.jsonl --per-task 64 \
    --out benchmarks/results/compare_home_tachyone.json

  # render the two tables separately (they are different datasets)
  uv run python -m benchmarks.compare render benchmarks/results/compare_{tachyone,systemone,llm_*}.json \
    --out benchmarks/compare_table.md
  ```

  Each artifact records the environment, the fitted temperature, the exact command, peak RSS/VRAM
  (of the process that did the inference — for the LLM that is `llama-server`, not the client) and
  a per-task breakdown.

### Exact commands behind the published tables (2026-09-26)

`docs/compare.md` §3 was produced by these eight runs on one RTX 3060 12GB, in this order
(serialized; `benchmarks/results/` is gitignored, so the artifacts — and their embedded
`environment.command` — do not survive a clean checkout):

```bash
OUT=benchmarks/results
MODEL_DIR=<directory holding the two GGUFs and System-one-Qwen3.5/>

# --- table A: the peer's public probe ------------------------------------------------------------
uv run python -m benchmarks.compare run --engine tachyone --per-task 64 \
  --out $OUT/compare_tachyone.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m benchmarks.compare run --engine systemone --per-task 64 --option-batch 4 \
  --systemone-script $MODEL_DIR/System-one-Qwen3.5/scorer/system_one.py \
  --systemone-base $MODEL_DIR/System-one-Qwen3.5/base \
  --systemone-adapter $MODEL_DIR/System-one-Qwen3.5/scorer \
  --out $OUT/compare_systemone.json

llama-server -m $MODEL_DIR/Ling-3.0/Ling-3.0-tiny-Q4_K_M.gguf --port 8081 --host 127.0.0.1 \
  -ngl 99 --temp 0 --seed 42 -c 8192 --alias ling-tiny &
uv run python -m benchmarks.compare run --engine llm \
  --llm-base-url http://127.0.0.1:8081/v1 --llm-model ling-tiny --llm-timeout 300 --per-task 4 \
  --out $OUT/compare_llm_ling.json
kill %1                                    # stop llama-server before the next GPU engine

llama-server -m $MODEL_DIR/Ornith-1.5/Ornith-1.5-9B-Q4_K_M.gguf --port 8082 --host 127.0.0.1 \
  -ngl 99 --temp 0 --seed 42 -c 8192 --alias ornith-9b &
uv run python -m benchmarks.compare run --engine llm \
  --llm-base-url http://127.0.0.1:8082/v1 --llm-model ornith-9b --llm-timeout 300 --per-task 4 \
  --out $OUT/compare_llm_ornith.json
kill %1

# --- table B: Tachyone's own records -------------------------------------------------------------
uv run python -m benchmarks.compare run --engine tachyone --format jsonl --per-task 64 \
  --data data/eval_en.jsonl --val-data data/train_en.jsonl \
  --out $OUT/compare_home_tachyone.json

uv run python -m benchmarks.compare run --engine llm --format jsonl --per-task 16 \
  --llm-base-url http://127.0.0.1:8081/v1 --llm-model ling-tiny --llm-timeout 300 \
  --data data/eval_en.jsonl --val-data data/train_en.jsonl \
  --out $OUT/compare_home_llm_ling.json     # and the same with ornith on 8082

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m benchmarks.compare run --engine systemone --format jsonl --per-task 64 \
  --option-batch 4 --data data/eval_en.jsonl --val-data data/train_en.jsonl \
  --systemone-script $MODEL_DIR/System-one-Qwen3.5/scorer/system_one.py \
  --systemone-base $MODEL_DIR/System-one-Qwen3.5/base \
  --systemone-adapter $MODEL_DIR/System-one-Qwen3.5/scorer \
  --out $OUT/compare_home_systemone.json
```

Why the two tables use different caps: the LLM rows ran **4 rows per probe family (n=36)** and
**16 per primitive (n=48)** because a single failing question costs 2–3 autoregressive attempts
(one 52-option attempt took 46 s), while the encoders scored every capped row (n=576 / n=192).
Their rows are subsets of the same ordered rows, and `docs/compare.md` §3 says so.

### The LLM rows were re-run after the B-10 prompt fix (2026-09-26)

`src/tachyone/backends/llm.py` now spells out the `answers` wrapper for all three primitives, so
the four `--engine llm` runs above were repeated with the **same servers, flags and rows** and
their artifacts written to `compare_{,home_}llm_{ling,ornith}_after.json`. Those four are the
numbers in `docs/compare.md` §3; the pre-fix artifacts (`compare_{,home_}llm_*.json` without the
`_after` suffix) hold the before side of the compliance table there:

```bash
# identical to the two llm commands above, only the output names differ
uv run python -m benchmarks.compare run --engine llm \
  --llm-base-url http://127.0.0.1:8081/v1 --llm-model ling-tiny --llm-timeout 300 --per-task 4 \
  --out benchmarks/results/compare_llm_ling_after.json

uv run python -m benchmarks.compare run --engine llm --format jsonl --per-task 16 \
  --llm-base-url http://127.0.0.1:8081/v1 --llm-model ling-tiny --llm-timeout 300 \
  --data data/eval_en.jsonl --val-data data/train_en.jsonl \
  --out benchmarks/results/compare_home_llm_ling_after.json     # and ornith on 8082
```

## Public probes (B-7) — `--probe`

`--format probe` takes a dataset selected by `--probe`. The default `system-one-decisions` is the
peer scorer's own distribution; `typed-decisions`, `massive` and `xnli` are loaded by
`benchmarks/probes.py`, which maps each one onto the harness row shape (`state / task / question /
options / answer_index`, plus `wire` where the dataset already speaks it) so every engine, metric,
temperature fit and renderer is reused untouched.

```bash
# one-time downloads (they are gitignored under data/)
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('LocalLLaMA/typed-decisions', repo_type='dataset', local_dir='data/typed-decisions')"
curl -L -o data/massive/raw/amazon-massive-dataset-1.1.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf data/massive/raw/amazon-massive-dataset-1.1.tar.gz -C data/massive --wildcards '1.1/data/en-US.jsonl' '1.1/data/pt-PT.jsonl' '1.1/data/es-ES.jsonl' '1.1/data/fr-FR.jsonl' '1.1/data/de-DE.jsonl' '1.1/data/it-IT.jsonl' '1.1/data/nl-NL.jsonl'
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/xnli', repo_type='dataset', allow_patterns=['en/*', 'README.md'], local_dir='data/xnli')"

OUT=benchmarks/results
uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe typed-decisions --data data/typed-decisions --per-task 500 \
  --out $OUT/probe_typed_decisions.json
uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe massive --data data/massive --per-task 512 \
  --languages en,pt,es,fr,de,it,nl --out $OUT/probe_massive.json
uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe xnli --data data/xnli --per-task 6000 --out $OUT/probe_xnli.json

# committed report: quality + performance + per-task accuracy/ECE, licences, citations, commands
uv run python -m benchmarks.probes render --out benchmarks/probes.md \
  --synthetic benchmarks/results/en.json $OUT/probe_*.json
```

Notes worth keeping:

- **The temperature split is per dataset** — `--val-split` defaults to `train` for
  typed-decisions, `dev` for MASSIVE and `validation` for XNLI, so the fit never touches the split
  being reported. Override with `--val-split` if you want something else.
- **`task` is per probe**: the dataset's config for typed-decisions, the **language** for MASSIVE
  and XNLI — which is where the per-language accuracy/ECE table in `benchmarks/probes.md` comes
  from.
- **Probe rows never reach `training/`.** These loaders are evaluation-only; using any of this
  data for training would turn the published number into an in-sample one.
- Licences are verified against the upstream sources (XNLI's HF card has no licence tag; it is
  **CC BY-NC 4.0** per `facebookresearch/XNLI`'s `LICENSE`).

## Not yet delivered

- `latency.py` / `throughput.py` — batch-size sweeps per backend.
- `calibration.py` — ECE curves before/after temperature fitting.

Results are written under `benchmarks/results/` (gitignored) and summarized in
[`benchmarks/report.md`](report.md).
