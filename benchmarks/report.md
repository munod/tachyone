# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 9,000 English / 18,000 multilingual deterministic synthetic
> records (fully localized per language via per-record RNG, one-in-six hard negatives), LoRA (r=16)
> plus a low-rank choice head (r=32), 4 epochs, batch 16 effective, bf16 + gradient checkpointing.
> Calibrated ECE is after per-`(primitive, language)` temperature fitting on the held-out synthetic
> split (in-sample; the same split is scored). Latency is per-question (batch=1). B-1 raises
> multilingual `choice` 0.40 → 0.73 and English overall 0.72 → 0.78 (L-002, L-003); multilingual
> `score`/`noul` and per-language calibration (es/nl/de) can still improve.

## Environment

```json
{
  "git_commit": "e5e9864300294d38ddfe1ad8b9a09514435d2d92",
  "jeba": "0.1.1",
  "peft": "0.21.0",
  "platform": "Linux-6.12.108-1-MANJARO-x86_64-with-glibc2.44",
  "pydantic": "2.13.5",
  "python": "3.12.13",
  "torch": "2.14.0+cu130",
  "transformers": "5.17.0"
}
```

## Reproduce

```bash
uv run python -m training.generate_data --seed 1 --per-type 3000 --languages en --out data/train_en.jsonl
uv run python -m training.generate_data --seed 1 --per-type 6000 --languages pt,es,fr,de,it,nl --out data/train_multi.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json    # LoRA r=16 + choice head r=32, epochs=4
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 512 --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 512 --temperature checkpoints/en/temperature_calibration.json --out-report benchmarks/results/en.json
# repeat predict/fit/predict for checkpoints/multi + data/eval_multi.jsonl (--max-len 1024)
uv run python -m benchmarks.report --entry 'english'=benchmarks/results/en.json --entry 'multilingual'=benchmarks/results/multi.json --out benchmarks/report.md
uv run python -m benchmarks.fast_path --data data/eval_multi.jsonl --model-id jhu-clsp/mmBERT-base --adapter checkpoints/multi --repeats 3 --out benchmarks/results/fast_path_multi.json
```

## Results

### english (ModernBERT-large + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.781 | 0.077 | 22.947 | 37.973 |
| choice | 500 | 0.708 | 0.141 | 36.743 | 38.657 |
| noul | 500 | 0.744 | 0.012 | 14.508 | 22.257 |
| score | 500 | 0.892 | 0.090 | 22.591 | 24.917 |
| lang:en | 1500 | 0.781 | 0.077 | 22.947 | 37.973 |

Worst language (accuracy): `en` — accuracy 0.781, ECE 0.077.

### multilingual (mmBERT-base + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.711 | 0.073 | 12.403 | 20.261 |
| choice | 500 | 0.734 | 0.100 | 18.843 | 21.084 |
| noul | 500 | 0.716 | 0.082 | 11.157 | 15.169 |
| score | 500 | 0.682 | 0.067 | 12.160 | 16.022 |
| lang:de | 249 | 0.598 | 0.119 | 12.469 | 19.940 |
| lang:es | 252 | 0.512 | 0.182 | 12.464 | 19.441 |
| lang:fr | 249 | 0.855 | 0.039 | 12.557 | 20.278 |
| lang:it | 249 | 0.827 | 0.068 | 12.249 | 44.317 |
| lang:nl | 249 | 0.606 | 0.178 | 12.389 | 19.985 |
| lang:pt | 252 | 0.865 | 0.045 | 12.395 | 20.041 |

Worst language (accuracy): `es` — accuracy 0.512, ECE 0.182.

## Fast path (CUDA graphs)

mmBERT-base + `checkpoints/multi`, 56 held-out states, single-question batch, RTX 3060 12GB.
`fast` = per-shape CUDA graphs with bf16-resident weights (`JEBA_FAST=1`); capture is warmed
before timing.

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 10.267 | 11.047 | 93.8 | 161.7 | 71.9 |
| fast (CUDA graphs, bf16) | 3.834 | 4.119 | 238.8 | 404.8 | 163.8 |

**Parity:** max absolute answer-probability difference **0.0044** with **0 top-label flips**
(16 held-out questions); embedding max absolute difference is bf16 rounding. Response shape is
unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths, with a **2.68× p50
speedup** for the fast path.
