# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 9,000 English / 18,000 multilingual deterministic synthetic records (fully localized per-record RNG, one-in-six hard negatives), LoRA r=16 + low-rank choice head r=32, 4 epochs, bf16 + gradient checkpointing. Calibrated ECE is in-sample on the held-out synthetic split. The noisy view applies one surface edit to 15% of states (B-4). English overall 0.763 / ECE 0.061; multilingual 0.702 / 0.033 (per-language ECE for es/de/nl still above 0.05; NFR-C06 open).

## Environment

```json
{
  "git_commit": "1782296905ba930b64fd442249156a8c4094f8e5",
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
uv run python -m training.generate_data --seed 1 --per-type 6000 --languages pt,es,fr,de,it,nl --noise-rate 0.15 --out data/train_multi_noisy.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi_noisy.json
# predict + fit_calibration per adapter, then evaluate clean + noisy (B-4):
uv run python -m training.predict --data data/eval_multi.jsonl --adapter checkpoints/multi --max-len 1024 --out-predictions data/preds_multi.jsonl
uv run python -m training.fit_calibration --calibration data/preds_multi.jsonl --out checkpoints/multi/temperature_calibration.json
```

## Results

### english (ModernBERT-large + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.763 | 0.061 | 22.603 | 37.051 |
| choice | 500 | 0.834 | 0.074 | 36.075 | 37.849 |
| noul | 500 | 0.744 | 0.101 | 14.030 | 22.070 |
| score | 500 | 0.712 | 0.042 | 22.385 | 24.093 |
| lang:en | 1500 | 0.763 | 0.061 | 22.603 | 37.051 |

Worst language (accuracy): `en` — accuracy 0.763, ECE 0.061.

Noisy view (noise_rate 0.15) — overall: accuracy 0.760, ECE 0.067, p50 22.628 ms.

### multilingual (mmBERT-base + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.702 | 0.033 | 13.328 | 20.772 |
| choice | 500 | 0.454 | 0.062 | 18.875 | 21.323 |
| noul | 500 | 0.738 | 0.023 | 12.342 | 16.416 |
| score | 500 | 0.914 | 0.041 | 12.862 | 16.736 |
| lang:de | 249 | 0.731 | 0.077 | 13.386 | 20.818 |
| lang:es | 252 | 0.472 | 0.170 | 13.284 | 19.756 |
| lang:fr | 249 | 0.775 | 0.028 | 13.436 | 21.314 |
| lang:it | 249 | 0.799 | 0.078 | 13.161 | 45.425 |
| lang:nl | 249 | 0.643 | 0.045 | 13.357 | 20.100 |
| lang:pt | 252 | 0.794 | 0.012 | 13.187 | 20.772 |

Worst language (accuracy): `es` — accuracy 0.472, ECE 0.170.

Noisy view (noise_rate 0.15) — overall: accuracy 0.701, ECE 0.035, p50 13.047 ms.

### multilingual, noise-augmented adapter

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.719 | 0.090 | 13.376 | 20.826 |
| choice | 500 | 0.504 | 0.152 | 18.995 | 20.988 |
| noul | 500 | 0.738 | 0.111 | 12.051 | 27.227 |
| score | 500 | 0.916 | 0.041 | 13.090 | 17.014 |
| lang:de | 249 | 0.530 | 0.148 | 13.485 | 27.548 |
| lang:es | 252 | 0.468 | 0.211 | 13.587 | 19.800 |
| lang:fr | 249 | 0.936 | 0.026 | 13.326 | 21.019 |
| lang:it | 249 | 0.948 | 0.091 | 13.260 | 27.592 |
| lang:nl | 249 | 0.486 | 0.196 | 13.408 | 20.150 |
| lang:pt | 252 | 0.948 | 0.010 | 13.214 | 20.525 |

Worst language (accuracy): `es` — accuracy 0.468, ECE 0.211.

Noisy view (noise_rate 0.15) — overall: accuracy 0.719, ECE 0.087, p50 13.084 ms.
