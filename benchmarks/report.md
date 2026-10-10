# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "725c57c87b18e20401f68c3756c42b0763350c12",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.11.0",
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
| overall | 1500 | 1.000 | 0.017 | 58.464 | 92.843 |
| choice | 500 | 1.000 | 0.004 | 88.646 | 95.737 |
| noul | 500 | 1.000 | 0.048 | 54.422 | 60.986 |
| score | 500 | 1.000 | 0.000 | 57.790 | 63.146 |
| lang:en | 1500 | 1.000 | 0.017 | 58.464 | 92.843 |

Worst language (accuracy): `en` — accuracy 1.000, ECE 0.017.

Noisy view (noise_rate 0.15) — overall: accuracy 0.995, ECE 0.013, p50 58.604 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 1.000 | 0.048 | 246 | 227 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.998 | 0.016 | 58.004 | 97.937 |
| choice | 2500 | 0.996 | 0.001 | 89.980 | 99.926 |
| noul | 2500 | 1.000 | 0.048 | 54.877 | 60.680 |
| score | 2500 | 1.000 | 0.000 | 57.348 | 62.896 |
| lang:en | 7500 | 0.998 | 0.016 | 58.004 | 97.937 |
| domain:agent_tools | 1500 | 1.000 | 0.015 | 58.000 | 93.229 |
| domain:documents | 1500 | 1.000 | 0.013 | 57.834 | 94.203 |
| domain:ecommerce | 1500 | 1.000 | 0.016 | 58.354 | 100.111 |
| domain:support | 1500 | 1.000 | 0.017 | 57.544 | 92.283 |
| domain:voice | 1500 | 0.991 | 0.030 | 58.105 | 93.379 |

Worst language (accuracy): `en` — accuracy 0.998, ECE 0.016.

Worst domain (accuracy): `voice` — accuracy 0.991, ECE 0.030.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/en | 1500 | 0.991 | 0.030 |
| agent_tools/en | 1500 | 1.000 | 0.015 |
| documents/en | 1500 | 1.000 | 0.013 |
| ecommerce/en | 1500 | 1.000 | 0.016 |
| support/en | 1500 | 1.000 | 0.017 |

Worst cell (accuracy): `voice/en` — accuracy 0.991, ECE 0.030.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 2500 | 1.000 | 0.048 | 1164 | 1201 | 135 | 0 | 0 (0.0%) |


### english five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.983 | 0.032 | 58.274 | 98.200 |
| choice | 2500 | 0.960 | 0.037 | 82.856 | 99.364 |
| noul | 2500 | 0.996 | 0.066 | 55.264 | 61.465 |
| score | 2500 | 0.992 | 0.008 | 57.429 | 62.797 |
| lang:en | 7500 | 0.983 | 0.032 | 58.274 | 98.200 |
| domain:agent_tools | 1500 | 0.999 | 0.017 | 58.475 | 93.494 |
| domain:documents | 1500 | 0.999 | 0.015 | 58.391 | 94.392 |
| domain:ecommerce | 1500 | 0.999 | 0.016 | 58.009 | 99.691 |
| domain:support | 1500 | 1.000 | 0.019 | 57.815 | 92.270 |
| domain:voice | 1500 | 0.915 | 0.120 | 58.456 | 94.002 |

Worst language (accuracy): `en` — accuracy 0.983, ECE 0.032.

Worst domain (accuracy): `voice` — accuracy 0.915, ECE 0.120.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/en | 1500 | 0.915 | 0.120 |
| documents/en | 1500 | 0.999 | 0.015 |
| agent_tools/en | 1500 | 0.999 | 0.017 |
| ecommerce/en | 1500 | 0.999 | 0.016 |
| support/en | 1500 | 1.000 | 0.019 |

Worst cell (accuracy): `voice/en` — accuracy 0.915, ECE 0.120.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 2500 | 0.996 | 0.066 | 1164 | 1201 | 135 | 0 | 0 (0.0%) |


### multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.930 | 0.039 | 53.120 | 84.152 |
| choice | 500 | 0.982 | 0.014 | 76.529 | 92.117 |
| noul | 500 | 0.928 | 0.016 | 50.052 | 60.106 |
| score | 500 | 0.880 | 0.094 | 51.909 | 61.521 |
| lang:de | 249 | 0.908 | 0.056 | 53.208 | 90.938 |
| lang:es | 252 | 0.948 | 0.032 | 52.946 | 81.073 |
| lang:fr | 249 | 0.980 | 0.038 | 53.144 | 80.608 |
| lang:it | 249 | 0.932 | 0.061 | 52.914 | 90.742 |
| lang:nl | 249 | 0.855 | 0.097 | 53.744 | 91.148 |
| lang:pt | 252 | 0.956 | 0.031 | 52.125 | 80.703 |

Worst language (accuracy): `nl` — accuracy 0.855, ECE 0.097.

Noisy view (noise_rate 0.15) — overall: accuracy 0.925, ECE 0.043, p50 53.203 ms.
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
| overall | 7500 | 0.922 | 0.034 | 53.497 | 88.180 |
| choice | 2500 | 0.957 | 0.039 | 78.028 | 96.744 |
| noul | 2500 | 0.933 | 0.010 | 50.538 | 59.734 |
| score | 2500 | 0.876 | 0.072 | 52.523 | 61.246 |
| lang:de | 1245 | 0.918 | 0.039 | 53.357 | 85.642 |
| lang:es | 1260 | 0.958 | 0.020 | 53.176 | 84.654 |
| lang:fr | 1245 | 0.950 | 0.013 | 53.339 | 85.123 |
| lang:it | 1245 | 0.933 | 0.031 | 53.533 | 87.539 |
| lang:nl | 1245 | 0.807 | 0.101 | 54.783 | 93.437 |
| lang:pt | 1260 | 0.965 | 0.015 | 53.168 | 84.303 |
| domain:agent_tools | 1500 | 0.907 | 0.039 | 53.337 | 83.609 |
| domain:documents | 1500 | 0.936 | 0.028 | 53.428 | 87.898 |
| domain:ecommerce | 1500 | 0.921 | 0.035 | 53.733 | 90.782 |
| domain:support | 1500 | 0.930 | 0.039 | 53.492 | 89.043 |
| domain:voice | 1500 | 0.916 | 0.053 | 53.513 | 89.277 |

Worst language (accuracy): `nl` — accuracy 0.807, ECE 0.101.

Worst domain (accuracy): `agent_tools` — accuracy 0.907, ECE 0.039.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| agent_tools/nl | 249 | 0.779 | 0.132 |
| voice/nl | 249 | 0.779 | 0.157 |
| documents/nl | 249 | 0.811 | 0.091 |
| ecommerce/nl | 249 | 0.811 | 0.104 |
| support/nl | 249 | 0.855 | 0.097 |
| agent_tools/de | 249 | 0.871 | 0.071 |
| ecommerce/de | 249 | 0.908 | 0.049 |
| support/de | 249 | 0.908 | 0.056 |
| documents/it | 249 | 0.916 | 0.054 |
| agent_tools/fr | 249 | 0.920 | 0.028 |
| voice/fr | 249 | 0.924 | 0.035 |
| voice/es | 252 | 0.925 | 0.061 |
| voice/it | 249 | 0.928 | 0.030 |
| support/it | 249 | 0.932 | 0.061 |
| documents/de | 249 | 0.936 | 0.042 |
| ecommerce/it | 249 | 0.936 | 0.026 |
| ecommerce/pt | 252 | 0.948 | 0.029 |
| support/es | 252 | 0.948 | 0.032 |
| agent_tools/pt | 252 | 0.952 | 0.012 |
| agent_tools/it | 249 | 0.956 | 0.020 |
| ecommerce/es | 252 | 0.956 | 0.034 |
| support/pt | 252 | 0.956 | 0.031 |
| documents/fr | 249 | 0.960 | 0.018 |
| agent_tools/es | 252 | 0.964 | 0.033 |
| ecommerce/fr | 249 | 0.968 | 0.013 |
| voice/de | 249 | 0.968 | 0.026 |
| voice/pt | 252 | 0.972 | 0.044 |
| support/fr | 249 | 0.980 | 0.038 |
| documents/es | 252 | 0.996 | 0.019 |
| documents/pt | 252 | 0.996 | 0.014 |

Worst cell (accuracy): `voice/nl` — accuracy 0.779, ECE 0.157.

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
| overall | 7500 | 0.926 | 0.033 | 53.174 | 86.106 |
| choice | 2500 | 0.953 | 0.032 | 77.680 | 97.631 |
| noul | 2500 | 0.939 | 0.007 | 49.941 | 57.988 |
| score | 2500 | 0.887 | 0.063 | 52.571 | 59.990 |
| lang:de | 1245 | 0.928 | 0.030 | 53.164 | 84.618 |
| lang:es | 1260 | 0.950 | 0.020 | 52.980 | 84.410 |
| lang:fr | 1245 | 0.949 | 0.025 | 53.104 | 84.754 |
| lang:it | 1245 | 0.947 | 0.030 | 53.073 | 86.606 |
| lang:nl | 1245 | 0.831 | 0.086 | 54.134 | 95.222 |
| lang:pt | 1260 | 0.954 | 0.021 | 52.924 | 83.984 |
| domain:agent_tools | 1500 | 0.931 | 0.036 | 53.053 | 81.988 |
| domain:documents | 1500 | 0.935 | 0.038 | 52.907 | 82.965 |
| domain:ecommerce | 1500 | 0.873 | 0.084 | 53.481 | 96.456 |
| domain:support | 1500 | 0.961 | 0.023 | 53.344 | 82.772 |
| domain:voice | 1500 | 0.932 | 0.023 | 52.880 | 82.065 |

Worst language (accuracy): `nl` — accuracy 0.831, ECE 0.086.

Worst domain (accuracy): `ecommerce` — accuracy 0.873, ECE 0.084.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| ecommerce/nl | 249 | 0.627 | 0.317 |
| agent_tools/nl | 249 | 0.819 | 0.101 |
| ecommerce/de | 249 | 0.855 | 0.076 |
| documents/nl | 249 | 0.863 | 0.072 |
| voice/nl | 249 | 0.871 | 0.064 |
| support/pt | 252 | 0.885 | 0.072 |
| agent_tools/fr | 249 | 0.888 | 0.070 |
| ecommerce/it | 249 | 0.896 | 0.081 |
| documents/es | 252 | 0.909 | 0.084 |
| voice/de | 249 | 0.916 | 0.049 |
| documents/de | 249 | 0.924 | 0.082 |
| voice/fr | 249 | 0.928 | 0.062 |
| ecommerce/pt | 252 | 0.933 | 0.046 |
| documents/it | 249 | 0.940 | 0.052 |
| voice/es | 252 | 0.940 | 0.029 |
| ecommerce/es | 252 | 0.952 | 0.033 |
| voice/it | 249 | 0.956 | 0.022 |
| agent_tools/it | 249 | 0.960 | 0.024 |
| agent_tools/de | 249 | 0.964 | 0.023 |
| support/es | 252 | 0.968 | 0.031 |
| ecommerce/fr | 249 | 0.972 | 0.008 |
| support/nl | 249 | 0.972 | 0.018 |
| agent_tools/pt | 252 | 0.972 | 0.009 |
| documents/fr | 249 | 0.976 | 0.015 |
| support/de | 249 | 0.980 | 0.022 |
| support/fr | 249 | 0.980 | 0.036 |
| agent_tools/es | 252 | 0.980 | 0.028 |
| voice/pt | 252 | 0.980 | 0.032 |
| support/it | 249 | 0.984 | 0.022 |
| documents/pt | 252 | 1.000 | 0.015 |

Worst cell (accuracy): `ecommerce/nl` — accuracy 0.627, ECE 0.317.

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
