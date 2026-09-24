# jeba Benchmark Report

> Measured on a single RTX 3060 12GB: 6,000 train / 1,500 eval deterministic synthetic records (v2 templates: localized cue terms per language incl. a learnable 'other' team, plus distractor clauses). LoRA (r=16) 3 epochs, batch 16, bf16 + gradient checkpointing. Calibrated ECE is after temperature fitting on a held-out split. Latency is per-question (batch=1). 'choice' stays near chance (~0.25 over 4 teams): the similarity baseline cannot separate the harder team mapping; a dedicated classification head (or the LLM backend) is the next step for that primitive.

## Environment

```json
{
  "git_commit": "9016264b12d2da109c873aac80587b9bca7b8307",
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
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages pt,es,fr,de,it,nl --out data/train_multi.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json    # epochs=3 batch=16 max_len=256 grad_ckpt
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 256 --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --max-len 256 --temperature checkpoints/en/temperature_calibration.json --out-report benchmarks/results/en.json
# repeat predict/fit/predict for checkpoints/multi + data/eval_multi.jsonl
uv run python -m benchmarks.report --entry 'english'=benchmarks/results/en.json --entry 'multilingual'=benchmarks/results/multi.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.613 | 0.059 | 17.522 | 24.900 |
| choice | 500 | 0.250 | 0.009 | 17.614 | 25.359 |
| noul | 500 | 0.700 | 0.146 | 14.012 | 20.643 |
| score | 500 | 0.890 | 0.062 | 18.208 | 24.714 |

### multilingual (mmBERT-base + LoRA)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.493 | 0.034 | 12.267 | 17.023 |
| choice | 500 | 0.256 | 0.065 | 12.860 | 17.791 |
| noul | 500 | 0.730 | 0.026 | 11.765 | 15.693 |
| score | 500 | 0.494 | 0.028 | 12.164 | 16.625 |
