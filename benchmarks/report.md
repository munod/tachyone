# jeba Benchmark Report

> Initial smoke run on a single RTX 3060 12GB: small deterministic synthetic set, 3 epochs, per-record LoRA updates. Accuracy/ECE are low and expected to improve with more data, epochs, and batched training. Latency is per-question on GPU (batch=1).

## Environment

```json
{
  "git_commit": "c6d4007e9bad267dabba8f12cd1d4f74da3e46e0",
  "jeba": "0.0.1",
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
uv run python -m training.generate_data --seed 1 --per-type 60 --languages en --out data/train_en.jsonl
uv run python -m training.generate_data --seed 1 --per-type 60 --languages pt,es,fr,de --out data/train_multi.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --temperature checkpoints/en/temperature_calibration.json --out-report benchmarks/results/en.json
uv run python -m benchmarks.report --entry 'english'=benchmarks/results/en.json --entry 'multilingual'=benchmarks/results/multi.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 90 | 0.267 | 0.108 | 17.204 | 23.881 |
| choice | 30 | 0.300 | 0.005 | 17.019 | 19.567 |
| noul | 30 | 0.233 | 0.320 | 14.997 | 79.950 |
| score | 30 | 0.267 | 0.000 | 17.597 | 23.881 |

### multilingual (mmBERT-base + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 90 | 0.167 | 0.187 | 12.628 | 19.085 |
| choice | 30 | 0.100 | 0.162 | 12.641 | 19.085 |
| noul | 30 | 0.233 | 0.313 | 11.973 | 35.965 |
| score | 30 | 0.167 | 0.087 | 12.719 | 14.283 |
