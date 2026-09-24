# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 6,000 train / 1,500 eval deterministic synthetic records, LoRA (r=16) 3 epochs, batch 16, bf16 + gradient checkpointing. Labels are synthetic and template-limited; choice has 4 options and score 4 levels (~0.25 chance), noul ~0.5 chance. Calibrated ECE is after temperature fitting on a held-out split. Latency is per-question (batch=1).

## Environment

```json
{
  "git_commit": "4ab2e428006555274b04d913adfce7a5e8aa9b08",
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
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages en --out data/train_en.jsonl
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages pt,es,fr,de --out data/train_multi.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json    # epochs=3 batch=16 max_len=256 grad_ckpt
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --temperature checkpoints/en/temperature_calibration.json --out-report benchmarks/results/en.json
# repeat predict/fit/predict for checkpoints/multi + data/eval_multi.jsonl
uv run python -m benchmarks.report --entry 'english'=benchmarks/results/en.json --entry 'multilingual'=benchmarks/results/multi.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.536 | 0.090 | 17.584 | 24.165 |
| choice | 500 | 0.308 | 0.009 | 17.611 | 19.511 |
| noul | 500 | 0.726 | 0.099 | 14.112 | 21.158 |
| score | 500 | 0.574 | 0.191 | 18.245 | 24.396 |

### multilingual (mmBERT-base + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.533 | 0.016 | 12.246 | 16.159 |
| choice | 500 | 0.320 | 0.010 | 12.300 | 15.482 |
| noul | 500 | 0.726 | 0.012 | 12.100 | 18.452 |
| score | 500 | 0.552 | 0.035 | 12.321 | 16.301 |
