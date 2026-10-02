# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "a68fccad56417f3179d6ef9848b46fdec15d4b85",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.5.0",
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
uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en --out-predictions data/preds_en.jsonl
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out checkpoints/en/temperature_calibration.json
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
uv run python -m training.generate_data --seed 1 --per-type 1000 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --per-domain support=6000 --out data/train_multi_domains.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --out data/eval_multi_domains.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi_domains.json
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_multi_domains.json
uv run python -m training.predict --data data/eval_multi_domains.jsonl --adapter checkpoints/multi_b5b_fit_bank --train-data data/train_multi_domains.jsonl --out-predictions data/preds_multi_domains_fit_bank.jsonl --out-report benchmarks/results/multi_domains_fit_bank.json
uv run python -m training.fit_calibration --calibration data/preds_multi_domains_fit_bank.jsonl --out benchmarks/results/calibration_multi_domains_fit_bank.json
uv run python -m benchmarks.report --entry "english (ModernBERT-large + LoRA r=16 + choice head)=benchmarks/results/en_split.json" --entry "english five-domain (B-5 choice-head bank, frozen trunk)=benchmarks/results/en_domains_bank_gate.json" --entry "multilingual (mmBERT-base + LoRA r=64 + choice head)=benchmarks/results/multi_split.json" --entry "multilingual five-domain (B-5b fitted choice-head bank, frozen trunk)=benchmarks/results/multi_domains_fit_bank.json" --out benchmarks/report.md
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
| overall | 1500 | 0.895 | 0.062 | 46.944 | 79.208 |
| choice | 500 | 0.964 | 0.032 | 73.265 | 87.651 |
| noul | 500 | 0.878 | 0.099 | 43.879 | 55.094 |
| score | 500 | 0.844 | 0.056 | 46.337 | 57.866 |
| lang:de | 249 | 0.900 | 0.083 | 47.055 | 78.406 |
| lang:es | 252 | 0.940 | 0.048 | 46.587 | 77.165 |
| lang:fr | 249 | 0.940 | 0.029 | 46.490 | 79.621 |
| lang:it | 249 | 0.932 | 0.056 | 46.824 | 85.826 |
| lang:nl | 249 | 0.715 | 0.167 | 50.776 | 76.504 |
| lang:pt | 252 | 0.944 | 0.034 | 46.620 | 81.543 |

Worst language (accuracy): `nl` — accuracy 0.715, ECE 0.167.

Noisy view (noise_rate 0.15) — overall: accuracy 0.889, ECE 0.067, p50 47.363 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.807 | 0.179 | 33 | 46 | 4 | 0 | 0 (0.0%) |
| es | 84 | 0.940 | 0.040 | 43 | 36 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 0.940 | 0.040 | 43 | 35 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.916 | 0.080 | 39 | 40 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.711 | 0.302 | 46 | 33 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 0.952 | 0.032 | 39 | 40 | 5 | 0 | 0 (0.0%) |


### multilingual five-domain (B-5b fitted choice-head bank, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.997 | 0.001 | 18.559 | 57.545 |
| choice | 2500 | 0.992 | 0.004 | 50.935 | 59.439 |
| noul | 2500 | 1.000 | 0.000 | 16.882 | 17.648 |
| score | 2500 | 1.000 | 0.000 | 18.537 | 20.425 |
| lang:de | 1245 | 0.998 | 0.001 | 18.575 | 57.748 |
| lang:es | 1260 | 0.998 | 0.004 | 18.542 | 57.191 |
| lang:fr | 1245 | 1.000 | 0.004 | 18.585 | 56.926 |
| lang:it | 1245 | 0.998 | 0.001 | 18.556 | 58.509 |
| lang:nl | 1245 | 0.995 | 0.003 | 18.552 | 57.537 |
| lang:pt | 1260 | 0.995 | 0.001 | 18.554 | 57.020 |
| domain:agent_tools | 1500 | 0.997 | 0.004 | 18.581 | 52.449 |
| domain:documents | 1500 | 0.998 | 0.002 | 18.662 | 53.288 |
| domain:ecommerce | 1500 | 0.993 | 0.004 | 18.592 | 59.203 |
| domain:support | 1500 | 1.000 | 0.000 | 18.527 | 51.912 |
| domain:voice | 1500 | 0.999 | 0.001 | 18.457 | 52.240 |

Worst language (accuracy): `nl` — accuracy 0.995, ECE 0.003.

Worst domain (accuracy): `ecommerce` — accuracy 0.993, ECE 0.004.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| ecommerce/pt | 252 | 0.980 | 0.012 |
| documents/nl | 249 | 0.988 | 0.010 |
| ecommerce/nl | 249 | 0.988 | 0.017 |
| agent_tools/it | 249 | 0.992 | 0.007 |
| ecommerce/de | 249 | 0.992 | 0.007 |
| agent_tools/es | 252 | 0.992 | 0.007 |
| ecommerce/it | 249 | 0.996 | 0.006 |
| voice/pt | 252 | 0.996 | 0.005 |
| agent_tools/de | 249 | 1.000 | 0.001 |
| agent_tools/fr | 249 | 1.000 | 0.016 |
| agent_tools/nl | 249 | 1.000 | 0.001 |
| agent_tools/pt | 252 | 1.000 | 0.009 |
| documents/de | 249 | 1.000 | 0.000 |
| documents/es | 252 | 1.000 | 0.007 |
| documents/fr | 249 | 1.000 | 0.000 |
| documents/it | 249 | 1.000 | 0.000 |
| documents/pt | 252 | 1.000 | 0.003 |
| ecommerce/es | 252 | 1.000 | 0.014 |
| ecommerce/fr | 249 | 1.000 | 0.004 |
| support/de | 249 | 1.000 | 0.000 |
| support/es | 252 | 1.000 | 0.000 |
| support/fr | 249 | 1.000 | 0.000 |
| support/it | 249 | 1.000 | 0.000 |
| support/nl | 249 | 1.000 | 0.000 |
| support/pt | 252 | 1.000 | 0.002 |
| voice/de | 249 | 1.000 | 0.000 |
| voice/es | 252 | 1.000 | 0.000 |
| voice/fr | 249 | 1.000 | 0.000 |
| voice/it | 249 | 1.000 | 0.000 |
| voice/nl | 249 | 1.000 | 0.002 |

Worst cell (accuracy): `ecommerce/pt` — accuracy 0.980, ECE 0.012.

#### `choice` gate

Answers routed by `gate` — strict accuracy 1.000, fell to shared 0, wrong domain 0 of n=2500.

| Domain | n | Strict accuracy | Fell to shared | Wrong domain |
| --- | --- | --- | --- | --- |
| agent_tools | 500 | 1.000 | 0 | 0 |
| documents | 500 | 1.000 | 0 | 0 |
| ecommerce | 500 | 1.000 | 0 | 0 |
| support | 500 | 1.000 | 0 | 0 |
| voice | 500 | 1.000 | 0 | 0 |

Input text seen in training: n=4504, accuracy 0.999 — never seen: n=2996, accuracy 0.996.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 1.000 | 0.000 | 181 | 214 | 20 | 0 | 0 (0.0%) |
| es | 420 | 1.000 | 0.000 | 198 | 197 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 1.000 | 0.000 | 178 | 212 | 25 | 0 | 0 (0.0%) |
| it | 415 | 1.000 | 0.000 | 202 | 193 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 1.000 | 0.000 | 211 | 184 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 1.000 | 0.000 | 210 | 185 | 25 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
