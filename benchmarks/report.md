# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "f041ca6c09140004c3a6d30b8740ed8c6ecf6c60",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.4.0",
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
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi_b12.json
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
uv run python -m training.predict --data data/eval_multi.jsonl --adapter checkpoints/multi --max-len 1024 --out-predictions data/preds_multi.jsonl
uv run python -m training.fit_calibration --calibration data/preds_multi.jsonl --out checkpoints/multi/temperature_calibration.json
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en uv run python -m training.evaluate --data data/eval_en.jsonl --out benchmarks/results/en_split.json --backend encoder --noise-rate 0.15 --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_multi.jsonl --out benchmarks/results/multi_split.json --backend encoder --noise-rate 0.15 --models-dir "$HOME/.cache/tachyone/models"
uv run python -m training.generate_data --seed 1 --per-type 1000 --languages en --domains support,ecommerce,agent_tools,documents,voice --per-domain support=3000 --out data/train_en_domains.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --domains support,ecommerce,agent_tools,documents,voice --out data/eval_en_domains.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en_domains.json
uv run python -m training.predict --data data/eval_en_domains.jsonl --adapter checkpoints/en_domains --out-predictions data/preds_en_domains.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en_domains.jsonl --out checkpoints/en_domains/temperature_calibration.json
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en_domains uv run python -m training.evaluate --data data/eval_en_domains.jsonl --out benchmarks/results/en_domains.json --backend encoder
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_en_domains.json
uv run python -m training.predict --data data/eval_en_domains.jsonl --adapter checkpoints/en_domains_bank --train-data data/train_en_domains.jsonl --out-predictions data/preds_en_domains_bank_gate.jsonl --out-report benchmarks/results/en_domains_bank_gate.json
uv run python -m benchmarks.report --entry encoder=benchmarks/results/en_split.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.964 | 0.023 | 54.384 | 90.428 |
| choice | 500 | 1.000 | 0.000 | 86.316 | 92.598 |
| noul | 500 | 0.946 | 0.038 | 50.858 | 56.172 |
| score | 500 | 0.946 | 0.030 | 53.668 | 58.860 |
| lang:en | 1500 | 0.964 | 0.023 | 54.384 | 90.428 |

Worst language (accuracy): `en` — accuracy 0.964, ECE 0.023.

Noisy view (noise_rate 0.15) — overall: accuracy 0.963, ECE 0.024, p50 54.492 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 0.946 | 0.038 | 241 | 232 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-5 choice-head bank, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.964 | 0.024 | 22.967 | 61.658 |
| choice | 2500 | 1.000 | 0.000 | 55.607 | 63.104 |
| noul | 2500 | 0.946 | 0.042 | 21.024 | 22.745 |
| score | 2500 | 0.946 | 0.029 | 22.949 | 24.857 |
| lang:en | 7500 | 0.964 | 0.024 | 22.967 | 61.658 |
| domain:agent_tools | 1500 | 0.964 | 0.023 | 22.932 | 57.589 |
| domain:documents | 1500 | 0.963 | 0.026 | 23.021 | 59.264 |
| domain:ecommerce | 1500 | 0.964 | 0.027 | 23.030 | 64.439 |
| domain:support | 1500 | 0.964 | 0.023 | 22.871 | 57.404 |
| domain:voice | 1500 | 0.963 | 0.021 | 22.959 | 57.385 |

Worst language (accuracy): `en` — accuracy 0.964, ECE 0.024.

Worst domain (accuracy): `documents` — accuracy 0.963, ECE 0.026.
#### `choice` gate

Answers routed by `gate` — strict accuracy 1.000, fell to shared 0, wrong domain 0 of n=2500.

| Domain | n | Strict accuracy | Fell to shared | Wrong domain |
| --- | --- | --- | --- | --- |
| agent_tools | 500 | 1.000 | 0 | 0 |
| documents | 500 | 1.000 | 0 | 0 |
| ecommerce | 500 | 1.000 | 0 | 0 |
| support | 500 | 1.000 | 0 | 0 |
| voice | 500 | 1.000 | 0 | 0 |

Input text seen in training: n=6709, accuracy 0.961 — never seen: n=791, accuracy 0.991.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 2500 | 0.946 | 0.042 | 1176 | 1189 | 135 | 0 | 0 (0.0%) |


### multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.743 | 0.089 | 16.021 | 46.240 |
| choice | 500 | 0.468 | 0.099 | 19.327 | 53.931 |
| noul | 500 | 0.892 | 0.108 | 11.616 | 20.368 |
| score | 500 | 0.870 | 0.076 | 12.871 | 26.652 |
| lang:de | 249 | 0.799 | 0.118 | 16.437 | 53.958 |
| lang:es | 252 | 0.921 | 0.045 | 15.200 | 47.337 |
| lang:fr | 249 | 0.667 | 0.101 | 15.206 | 45.827 |
| lang:it | 249 | 0.618 | 0.197 | 15.962 | 45.433 |
| lang:nl | 249 | 0.675 | 0.192 | 19.012 | 45.327 |
| lang:pt | 252 | 0.778 | 0.062 | 13.141 | 53.890 |

Worst language (accuracy): `it` — accuracy 0.618, ECE 0.197.

Noisy view (noise_rate 0.15) — overall: accuracy 0.744, ECE 0.092, p50 15.724 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.795 | 0.212 | 33 | 46 | 4 | 0 | 0 (0.0%) |
| es | 84 | 1.000 | 0.003 | 43 | 36 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 0.916 | 0.084 | 43 | 35 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.892 | 0.107 | 39 | 40 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.747 | 0.259 | 46 | 33 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 1.000 | 0.008 | 39 | 40 | 5 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
