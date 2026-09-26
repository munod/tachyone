# Tachyone Benchmark Report

> Measured on a single RTX 3060 12GB: 9,000 English / 18,000 multilingual deterministic synthetic records (fully localized per-record RNG, one-in-six hard negatives), LoRA plus a low-rank choice head (r=32), 4 epochs, bf16 + gradient checkpointing. Calibrated ECE is in-sample on the held-out synthetic split. English LoRA r=16; multilingual LoRA raised to r=64 (alpha=128), which lifted overall accuracy 0.702 -> 0.853 and cut es ECE 0.170 -> 0.038. The noisy view applies one surface edit to 15% of states (B-4). nl per-language ECE remains > 0.05.

## Environment

```json
{
  "git_commit": "2b4e77d1f917e891d08f6b11e9bf3742f63a9af4",
  "tachyone": "0.3.0",
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
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_multi.jsonl --adapter checkpoints/multi --max-len 1024 --out-predictions data/preds_multi.jsonl
uv run python -m training.fit_calibration --calibration data/preds_multi.jsonl --out checkpoints/multi/temperature_calibration.json
uv run python -m training.evaluate --data data/eval_en.jsonl --out benchmarks/results/en_split.json --backend encoder
uv run python -m training.evaluate --data data/eval_multi.jsonl --out benchmarks/results/multi_split.json --backend encoder
uv run python -m benchmarks.report --entry encoder=benchmarks/results/en_split.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.859 | 0.023 | 23.307 | 39.010 |
| choice | 500 | 0.948 | 0.020 | 36.849 | 40.910 |
| noul | 500 | 0.718 | 0.020 | 14.841 | 22.032 |
| score | 500 | 0.910 | 0.039 | 22.956 | 25.211 |
| lang:en | 1500 | 0.859 | 0.023 | 23.307 | 39.010 |

Worst language (accuracy): `en` — accuracy 0.859, ECE 0.023.

Noisy view (noise_rate 0.15) — overall: accuracy 0.854, ECE 0.026, p50 23.454 ms.

### multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.853 | 0.038 | 13.254 | 23.189 |
| choice | 500 | 0.684 | 0.083 | 19.627 | 24.108 |
| noul | 500 | 0.960 | 0.038 | 12.003 | 17.053 |
| score | 500 | 0.916 | 0.032 | 12.699 | 18.302 |
| lang:de | 249 | 0.863 | 0.063 | 13.493 | 31.312 |
| lang:es | 252 | 0.956 | 0.038 | 13.169 | 21.753 |
| lang:fr | 249 | 0.871 | 0.051 | 13.397 | 23.058 |
| lang:it | 249 | 0.880 | 0.059 | 13.219 | 47.450 |
| lang:nl | 249 | 0.663 | 0.104 | 13.216 | 22.701 |
| lang:pt | 252 | 0.885 | 0.024 | 13.063 | 21.918 |

Worst language (accuracy): `nl` — accuracy 0.663, ECE 0.104.

Noisy view (noise_rate 0.15) — overall: accuracy 0.847, ECE 0.042, p50 13.529 ms.
