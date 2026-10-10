# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "9353a6aa5363fbf2dc9717f92d0d6417ba3de352",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.10.0",
  "torch": "2.14.0+cu130",
  "transformers": "5.17.0"
}
```

## Reproduce

```bash
uv run python -m training.generate_data --seed 1 --per-type 4000 --languages en --template-split train --out data/train_en.jsonl
uv run python -m training.generate_data --seed 1 --per-type 7800 --languages pt,es,fr,de,it,nl --template-split train --out data/train_multi.jsonl
uv run python -m training.generate_data --seed 1 --per-type 2000 --languages en --domains support,ecommerce,agent_tools,documents,voice --per-domain support=4000 --template-split train --out data/train_en_domains.jsonl
uv run python -m training.generate_data --seed 1 --per-type 2400 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --per-domain support=7800 --template-split train --out data/train_multi_domains.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --out data/eval_en.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --domains support,ecommerce,agent_tools,documents,voice --out data/eval_en_domains.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --out data/eval_multi_domains.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --domains support,ecommerce,agent_tools,documents,voice --template-split holdout --out data/eval_en_domains_holdout.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --template-split holdout --out data/eval_multi_domains_holdout.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages en --domains support,ecommerce,agent_tools,documents,voice --template-split train --out data/eval_en_domains_train.jsonl
uv run python -m training.generate_data --seed 2 --per-type 500 --languages pt,es,fr,de,it,nl --domains support,ecommerce,agent_tools,documents,voice --template-split train --out data/eval_multi_domains_train.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en_domains.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi_domains.json
uv run python -m training.interleave_continue --adapter checkpoints/multi_tt --out checkpoints/multi_tt_il --epochs 8,9 --data data/train_multi_domains.jsonl
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_en_domains.json
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_multi_domains.json
uv run python -m training.build_calibration_holdout --public-dir <jevbench-clone>/datasets/public
uv run python -m training.build_prototypes --data data/train_en_domains.jsonl --adapter checkpoints/en_tt --model-id answerdotai/ModernBERT-large --out checkpoints/en_tt/state_prototypes.json
uv run python -m training.predict --data data/calibration_holdout.jsonl --adapter checkpoints/en_tt --prototypes checkpoints/en_tt/state_prototypes.json --out-predictions data/calibration_holdout_preds_entt.jsonl
uv run python -m training.fit_confidence --predictions data/calibration_holdout_preds_entt.jsonl --prototypes checkpoints/en_tt/state_prototypes.json --out-dir checkpoints/en_tt
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_en.jsonl --out benchmarks/results/en_split.json --backend encoder --noise-rate 0.15 --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_multi.jsonl --out benchmarks/results/multi_split.json --backend encoder --noise-rate 0.15 --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_en_domains.jsonl --out benchmarks/results/en_domains.json --backend encoder --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_multi_domains.jsonl --out benchmarks/results/multi_domains.json --backend encoder --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_en_domains_holdout.jsonl --out benchmarks/results/en_domains_holdout.json --backend encoder --models-dir "$HOME/.cache/tachyone/models"
TACHYONE_ADAPTERS=tachyone-en=checkpoints/en,tachyone-multi=checkpoints/multi uv run python -m training.evaluate --data data/eval_multi_domains_holdout.jsonl --out benchmarks/results/multi_domains_holdout.json --backend encoder --models-dir "$HOME/.cache/tachyone/models"
uv run python -m benchmarks.report --entry "english (ModernBERT-large + LoRA r=16 + choice head)=benchmarks/results/en_split.json" --entry "english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)=benchmarks/results/en_domains.json" --entry "english five-domain, holdout phrasing (never in training)=benchmarks/results/en_domains_holdout.json" --entry "multilingual (mmBERT-base + LoRA r=64 + choice head)=benchmarks/results/multi_split.json" --entry "multilingual five-domain (B-15 template-train fitted bank, frozen trunk)=benchmarks/results/multi_domains.json" --entry "multilingual five-domain, holdout phrasing (never in training)=benchmarks/results/multi_domains_holdout.json" --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 1.000 | 0.188 | 58.312 | 92.150 |
| choice | 500 | 1.000 | 0.517 | 77.735 | 93.462 |
| noul | 500 | 1.000 | 0.048 | 54.701 | 61.168 |
| score | 500 | 1.000 | 0.000 | 57.767 | 61.952 |
| lang:en | 1500 | 1.000 | 0.188 | 58.312 | 92.150 |

Worst language (accuracy): `en` — accuracy 1.000, ECE 0.188.

Noisy view (noise_rate 0.15) — overall: accuracy 0.995, ECE 0.185, p50 57.853 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 1.000 | 0.048 | 246 | 227 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.998 | 0.191 | 58.206 | 98.054 |
| choice | 2500 | 0.996 | 0.524 | 88.602 | 99.560 |
| noul | 2500 | 1.000 | 0.048 | 55.119 | 61.089 |
| score | 2500 | 1.000 | 0.000 | 57.312 | 62.456 |
| lang:en | 7500 | 0.998 | 0.191 | 58.206 | 98.054 |
| domain:agent_tools | 1500 | 1.000 | 0.184 | 58.368 | 95.796 |
| domain:documents | 1500 | 1.000 | 0.182 | 57.727 | 93.338 |
| domain:ecommerce | 1500 | 1.000 | 0.210 | 58.310 | 99.787 |
| domain:support | 1500 | 1.000 | 0.188 | 58.206 | 92.507 |
| domain:voice | 1500 | 0.991 | 0.190 | 58.236 | 93.633 |

Worst language (accuracy): `en` — accuracy 0.998, ECE 0.191.

Worst domain (accuracy): `voice` — accuracy 0.991, ECE 0.190.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/en | 1500 | 0.991 | 0.190 |
| agent_tools/en | 1500 | 1.000 | 0.184 |
| documents/en | 1500 | 1.000 | 0.182 |
| ecommerce/en | 1500 | 1.000 | 0.210 |
| support/en | 1500 | 1.000 | 0.188 |

Worst cell (accuracy): `voice/en` — accuracy 0.991, ECE 0.190.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 2500 | 1.000 | 0.048 | 1164 | 1201 | 135 | 0 | 0 (0.0%) |


### english five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.983 | 0.188 | 58.161 | 97.978 |
| choice | 2500 | 0.960 | 0.506 | 88.053 | 98.768 |
| noul | 2500 | 0.996 | 0.066 | 54.927 | 60.864 |
| score | 2500 | 0.992 | 0.008 | 57.373 | 63.038 |
| lang:en | 7500 | 0.983 | 0.188 | 58.161 | 97.978 |
| domain:agent_tools | 1500 | 0.999 | 0.188 | 58.385 | 94.344 |
| domain:documents | 1500 | 0.999 | 0.189 | 58.066 | 93.551 |
| domain:ecommerce | 1500 | 0.999 | 0.214 | 58.157 | 98.980 |
| domain:support | 1500 | 1.000 | 0.193 | 58.248 | 92.900 |
| domain:voice | 1500 | 0.915 | 0.178 | 58.111 | 93.714 |

Worst language (accuracy): `en` — accuracy 0.983, ECE 0.188.

Worst domain (accuracy): `voice` — accuracy 0.915, ECE 0.178.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/en | 1500 | 0.915 | 0.178 |
| documents/en | 1500 | 0.999 | 0.189 |
| agent_tools/en | 1500 | 0.999 | 0.188 |
| ecommerce/en | 1500 | 0.999 | 0.214 |
| support/en | 1500 | 1.000 | 0.193 |

Worst cell (accuracy): `voice/en` — accuracy 0.915, ECE 0.178.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 2500 | 0.996 | 0.066 | 1164 | 1201 | 135 | 0 | 0 (0.0%) |


### multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.930 | 0.038 | 52.782 | 83.679 |
| choice | 500 | 0.982 | 0.030 | 75.471 | 92.004 |
| noul | 500 | 0.928 | 0.016 | 48.888 | 57.674 |
| score | 500 | 0.880 | 0.094 | 52.525 | 60.025 |
| lang:de | 249 | 0.908 | 0.065 | 53.320 | 82.369 |
| lang:es | 252 | 0.948 | 0.042 | 52.681 | 83.320 |
| lang:fr | 249 | 0.980 | 0.048 | 52.207 | 80.901 |
| lang:it | 249 | 0.932 | 0.051 | 52.842 | 91.481 |
| lang:nl | 249 | 0.855 | 0.091 | 53.335 | 88.821 |
| lang:pt | 252 | 0.956 | 0.043 | 52.476 | 81.390 |

Worst language (accuracy): `nl` — accuracy 0.855, ECE 0.091.

Noisy view (noise_rate 0.15) — overall: accuracy 0.925, ECE 0.043, p50 53.324 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.880 | 0.047 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| es | 84 | 0.929 | 0.025 | 40 | 39 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 1.000 | 0.030 | 38 | 40 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.976 | 0.028 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.843 | 0.066 | 43 | 36 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 0.940 | 0.019 | 35 | 44 | 5 | 0 | 0 (0.0%) |


### multilingual five-domain (B-17 teacher-expanded banks, B-15 recipe)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.922 | 0.026 | 53.501 | 87.689 |
| choice | 2500 | 0.957 | 0.030 | 78.209 | 95.899 |
| noul | 2500 | 0.933 | 0.010 | 50.170 | 59.718 |
| score | 2500 | 0.876 | 0.072 | 52.666 | 60.470 |
| lang:de | 1245 | 0.918 | 0.036 | 53.527 | 87.145 |
| lang:es | 1260 | 0.958 | 0.029 | 53.318 | 84.360 |
| lang:fr | 1245 | 0.950 | 0.018 | 53.246 | 85.149 |
| lang:it | 1245 | 0.933 | 0.031 | 53.463 | 87.964 |
| lang:nl | 1245 | 0.807 | 0.049 | 55.057 | 92.333 |
| lang:pt | 1260 | 0.965 | 0.026 | 52.853 | 84.726 |
| domain:agent_tools | 1500 | 0.907 | 0.020 | 54.026 | 90.699 |
| domain:documents | 1500 | 0.936 | 0.021 | 52.604 | 86.588 |
| domain:ecommerce | 1500 | 0.921 | 0.027 | 53.730 | 89.486 |
| domain:support | 1500 | 0.930 | 0.038 | 53.033 | 86.167 |
| domain:voice | 1500 | 0.916 | 0.041 | 53.821 | 84.781 |

Worst language (accuracy): `nl` — accuracy 0.807, ECE 0.049.

Worst domain (accuracy): `agent_tools` — accuracy 0.907, ECE 0.020.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| agent_tools/nl | 249 | 0.779 | 0.074 |
| voice/nl | 249 | 0.779 | 0.096 |
| documents/nl | 249 | 0.811 | 0.041 |
| ecommerce/nl | 249 | 0.811 | 0.041 |
| support/nl | 249 | 0.855 | 0.091 |
| agent_tools/de | 249 | 0.871 | 0.056 |
| ecommerce/de | 249 | 0.908 | 0.060 |
| support/de | 249 | 0.908 | 0.065 |
| documents/it | 249 | 0.916 | 0.036 |
| agent_tools/fr | 249 | 0.920 | 0.028 |
| voice/fr | 249 | 0.924 | 0.040 |
| voice/es | 252 | 0.925 | 0.051 |
| voice/it | 249 | 0.928 | 0.035 |
| support/it | 249 | 0.932 | 0.051 |
| documents/de | 249 | 0.936 | 0.042 |
| ecommerce/it | 249 | 0.936 | 0.035 |
| ecommerce/pt | 252 | 0.948 | 0.042 |
| support/es | 252 | 0.948 | 0.042 |
| agent_tools/pt | 252 | 0.952 | 0.022 |
| agent_tools/it | 249 | 0.956 | 0.028 |
| ecommerce/es | 252 | 0.956 | 0.046 |
| support/pt | 252 | 0.956 | 0.043 |
| documents/fr | 249 | 0.960 | 0.030 |
| agent_tools/es | 252 | 0.964 | 0.044 |
| ecommerce/fr | 249 | 0.968 | 0.024 |
| voice/de | 249 | 0.968 | 0.020 |
| voice/pt | 252 | 0.972 | 0.054 |
| support/fr | 249 | 0.980 | 0.048 |
| documents/es | 252 | 0.996 | 0.030 |
| documents/pt | 252 | 0.996 | 0.025 |

Worst cell (accuracy): `voice/nl` — accuracy 0.779, ECE 0.096.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.928 | 0.014 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.969 | 0.005 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.964 | 0.016 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.923 | 0.020 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.848 | 0.040 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.967 | 0.015 | 200 | 195 | 25 | 0 | 0 (0.0%) |


### multilingual five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.926 | 0.032 | 53.273 | 87.861 |
| choice | 2500 | 0.953 | 0.036 | 78.272 | 98.947 |
| noul | 2500 | 0.939 | 0.007 | 50.558 | 58.305 |
| score | 2500 | 0.887 | 0.063 | 52.553 | 60.445 |
| lang:de | 1245 | 0.928 | 0.031 | 53.111 | 86.664 |
| lang:es | 1260 | 0.950 | 0.031 | 53.138 | 86.449 |
| lang:fr | 1245 | 0.949 | 0.036 | 52.987 | 84.765 |
| lang:it | 1245 | 0.947 | 0.026 | 53.058 | 91.821 |
| lang:nl | 1245 | 0.831 | 0.055 | 54.183 | 98.055 |
| lang:pt | 1260 | 0.954 | 0.032 | 52.950 | 84.886 |
| domain:agent_tools | 1500 | 0.931 | 0.043 | 53.058 | 82.918 |
| domain:documents | 1500 | 0.935 | 0.044 | 51.994 | 84.508 |
| domain:ecommerce | 1500 | 0.873 | 0.046 | 54.086 | 98.239 |
| domain:support | 1500 | 0.961 | 0.031 | 53.454 | 82.634 |
| domain:voice | 1500 | 0.932 | 0.030 | 53.516 | 81.976 |

Worst language (accuracy): `nl` — accuracy 0.831, ECE 0.055.

Worst domain (accuracy): `ecommerce` — accuracy 0.873, ECE 0.046.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| ecommerce/nl | 249 | 0.627 | 0.120 |
| agent_tools/nl | 249 | 0.819 | 0.100 |
| ecommerce/de | 249 | 0.855 | 0.086 |
| documents/nl | 249 | 0.863 | 0.081 |
| voice/nl | 249 | 0.871 | 0.072 |
| support/pt | 252 | 0.885 | 0.085 |
| agent_tools/fr | 249 | 0.888 | 0.081 |
| ecommerce/it | 249 | 0.896 | 0.044 |
| documents/es | 252 | 0.909 | 0.095 |
| voice/de | 249 | 0.916 | 0.042 |
| documents/de | 249 | 0.924 | 0.091 |
| voice/fr | 249 | 0.928 | 0.072 |
| ecommerce/pt | 252 | 0.933 | 0.058 |
| documents/it | 249 | 0.940 | 0.033 |
| voice/es | 252 | 0.940 | 0.040 |
| ecommerce/es | 252 | 0.952 | 0.045 |
| voice/it | 249 | 0.956 | 0.031 |
| agent_tools/it | 249 | 0.960 | 0.033 |
| agent_tools/de | 249 | 0.964 | 0.032 |
| support/es | 252 | 0.968 | 0.042 |
| ecommerce/fr | 249 | 0.972 | 0.018 |
| support/nl | 249 | 0.972 | 0.026 |
| agent_tools/pt | 252 | 0.972 | 0.019 |
| documents/fr | 249 | 0.976 | 0.026 |
| support/de | 249 | 0.980 | 0.031 |
| support/fr | 249 | 0.980 | 0.047 |
| agent_tools/es | 252 | 0.980 | 0.038 |
| voice/pt | 252 | 0.980 | 0.042 |
| support/it | 249 | 0.984 | 0.031 |
| documents/pt | 252 | 1.000 | 0.027 |

Worst cell (accuracy): `ecommerce/nl` — accuracy 0.627, ECE 0.120.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.981 | 0.009 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.974 | 0.014 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.981 | 0.020 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.954 | 0.011 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.790 | 0.059 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.955 | 0.002 | 200 | 195 | 25 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
