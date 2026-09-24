# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 6,000 train / 1,500 eval deterministic synthetic records (localized cue terms per language, a learnable 'other' team with rich descriptions, and distractor clauses). LoRA (r=16) + a dedicated low-rank choice head (r=16, near-identity init), 3 epochs, batch 16, bf16 + gradient checkpointing. Calibrated ECE is after temperature fitting on a held-out split. Latency is per-question (batch=1). The choice head lifts English choice from ~0.25 to 0.78 and multilingual to 0.40 (L-002). open: multilingual 'choice' and 'score' calibration can improve.

## Environment

```json
{
  "git_commit": "a2ee6d3b5edf4ead13624ad4f047856faefe4412",
  "jeba": "0.1.0",
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
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages en --out data/train_en.jsonl
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages pt,es,fr,de,it,nl --out data/train_multi.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json    # LoRA + choice head, epochs=3 batch=16
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 256 --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 256 --temperature checkpoints/en/temperature_calibration.json --out-report benchmarks/results/en.json
# repeat predict/fit/predict for checkpoints/multi + data/eval_multi.jsonl
uv run python -m benchmarks.report --entry 'english'=benchmarks/results/en.json --entry 'multilingual'=benchmarks/results/multi.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.721 | 0.052 | 18.376 | 27.551 |
| choice | 500 | 0.782 | 0.038 | 20.343 | 27.819 |
| noul | 500 | 0.718 | 0.028 | 13.844 | 20.036 |
| score | 500 | 0.664 | 0.149 | 18.201 | 24.557 |

### multilingual (mmBERT-base + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.609 | 0.086 | 12.597 | 17.256 |
| choice | 500 | 0.398 | 0.038 | 14.233 | 17.621 |
| noul | 500 | 0.728 | 0.042 | 11.573 | 16.881 |
| score | 500 | 0.700 | 0.178 | 12.138 | 16.020 |
