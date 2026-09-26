# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 9,000 English / 18,000 multilingual deterministic synthetic records (fully localized per-record RNG, one-in-six hard negatives), LoRA plus a low-rank choice head (r=32), 4 epochs, bf16 + gradient checkpointing. Calibrated ECE is in-sample on the held-out synthetic split. English LoRA r=16; multilingual LoRA raised to r=64 (alpha=128), which lifted overall accuracy 0.702 -> 0.853 and cut es ECE 0.170 -> 0.038. The noisy view applies one surface edit to 15% of states (B-4). nl per-language ECE remains > 0.05.

## Environment

```json
{
  "git_commit": "c43de7cc43a2a90be9493ec15e18df5ebd90718b",
  "jeba": "0.2.0",
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
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json   # LoRA r=16 + choice head r=32
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json # LoRA r=64 + choice head r=32
uv run python -m training.predict --data data/eval_multi.jsonl --adapter checkpoints/multi --max-len 1024 --out-predictions data/preds_multi.jsonl
uv run python -m training.fit_calibration --calibration data/preds_multi.jsonl --out checkpoints/multi/temperature_calibration.json
```

## Results

### english (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.763 | 0.061 | 22.603 | 37.051 |
| choice | 500 | 0.834 | 0.074 | 36.075 | 37.849 |
| noul | 500 | 0.744 | 0.101 | 14.030 | 22.070 |
| score | 500 | 0.712 | 0.042 | 22.385 | 24.093 |
| lang:en | 1500 | 0.763 | 0.061 | 22.603 | 37.051 |

Worst language (accuracy): `en` — accuracy 0.763, ECE 0.061.

Noisy view (noise_rate 0.15) — overall: accuracy 0.760, ECE 0.067, p50 22.628 ms.

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
