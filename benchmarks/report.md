# tachyone Benchmark Report

> B-15 (2026-10-08): both checkpoints retrained on template_split=train; the holdout entries are phrasing never seen in any training. The router sends empty/no-signal states to the English checkpoint by design (~14% of five-domain multilingual rows), so multilingual entries pin BOTH local checkpoints - the published pair. English choice confidence is fitted on the pooled never-trained holdout (L-015): in-domain choice ECE reads 0.5243 honestly instead of in-sample (P3 in-domain <= 0.05 recorded NOT met for the rebuilt asset); accuracy is untouched. B-1 per-language ECE on holdout is NOT met for multilingual - numbers published in BACKLOG B-15.

## Environment

```json
{
  "git_commit": "71974e7dae1558ccb3d32282af9c67dfd6dd1b4b",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.8.0",
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
| overall | 1500 | 1.000 | 0.188 | 57.664 | 92.621 |
| choice | 500 | 1.000 | 0.517 | 88.233 | 94.792 |
| noul | 500 | 1.000 | 0.048 | 53.993 | 60.493 |
| score | 500 | 1.000 | 0.000 | 56.994 | 63.024 |
| lang:en | 1500 | 1.000 | 0.188 | 57.664 | 92.621 |

Worst language (accuracy): `en` — accuracy 1.000, ECE 0.188.

Noisy view (noise_rate 0.15) — overall: accuracy 0.995, ECE 0.185, p50 57.376 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 1.000 | 0.048 | 246 | 227 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.998 | 0.191 | 57.680 | 96.701 |
| choice | 2500 | 0.996 | 0.524 | 88.743 | 98.431 |
| noul | 2500 | 1.000 | 0.048 | 54.413 | 60.744 |
| score | 2500 | 1.000 | 0.000 | 56.735 | 62.275 |
| lang:en | 7500 | 0.998 | 0.191 | 57.680 | 96.701 |
| domain:agent_tools | 1500 | 1.000 | 0.184 | 58.020 | 92.960 |
| domain:documents | 1500 | 1.000 | 0.182 | 57.580 | 95.103 |
| domain:ecommerce | 1500 | 1.000 | 0.210 | 57.782 | 98.720 |
| domain:support | 1500 | 1.000 | 0.188 | 57.422 | 91.448 |
| domain:voice | 1500 | 0.991 | 0.190 | 57.847 | 94.879 |

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
| overall | 7500 | 0.983 | 0.188 | 58.012 | 97.506 |
| choice | 2500 | 0.960 | 0.506 | 89.061 | 98.673 |
| noul | 2500 | 0.996 | 0.066 | 55.238 | 61.157 |
| score | 2500 | 0.992 | 0.008 | 56.905 | 62.443 |
| lang:en | 7500 | 0.983 | 0.188 | 58.012 | 97.506 |
| domain:agent_tools | 1500 | 0.999 | 0.188 | 57.876 | 92.456 |
| domain:documents | 1500 | 0.999 | 0.189 | 58.245 | 92.818 |
| domain:ecommerce | 1500 | 0.999 | 0.214 | 58.110 | 99.361 |
| domain:support | 1500 | 1.000 | 0.193 | 58.020 | 93.846 |
| domain:voice | 1500 | 0.915 | 0.178 | 57.819 | 92.183 |

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
| overall | 1500 | 0.911 | 0.051 | 47.383 | 78.472 |
| choice | 500 | 1.000 | 0.040 | 72.549 | 90.785 |
| noul | 500 | 0.860 | 0.049 | 44.101 | 58.893 |
| score | 500 | 0.872 | 0.103 | 46.796 | 59.579 |
| lang:de | 249 | 0.876 | 0.086 | 49.365 | 90.132 |
| lang:es | 252 | 0.909 | 0.061 | 47.067 | 77.152 |
| lang:fr | 249 | 0.968 | 0.040 | 47.190 | 76.526 |
| lang:it | 249 | 0.888 | 0.104 | 47.439 | 89.618 |
| lang:nl | 249 | 0.863 | 0.096 | 48.773 | 78.806 |
| lang:pt | 252 | 0.960 | 0.021 | 47.219 | 77.954 |

Worst language (accuracy): `nl` — accuracy 0.863, ECE 0.096.

Noisy view (noise_rate 0.15) — overall: accuracy 0.909, ECE 0.051, p50 47.458 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.783 | 0.103 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| es | 84 | 0.845 | 0.095 | 40 | 39 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 0.964 | 0.062 | 38 | 40 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.831 | 0.148 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.795 | 0.096 | 43 | 36 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 0.940 | 0.047 | 35 | 44 | 5 | 0 | 0 (0.0%) |


### multilingual five-domain (B-15 template-train fitted bank, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.916 | 0.019 | 47.996 | 83.556 |
| choice | 2500 | 0.956 | 0.038 | 71.965 | 93.079 |
| noul | 2500 | 0.914 | 0.012 | 44.701 | 58.520 |
| score | 2500 | 0.877 | 0.065 | 47.145 | 61.421 |
| lang:de | 1245 | 0.912 | 0.026 | 48.128 | 86.149 |
| lang:es | 1260 | 0.944 | 0.028 | 47.625 | 80.220 |
| lang:fr | 1245 | 0.945 | 0.024 | 47.842 | 80.750 |
| lang:it | 1245 | 0.914 | 0.052 | 47.797 | 82.355 |
| lang:nl | 1245 | 0.816 | 0.041 | 50.636 | 91.770 |
| lang:pt | 1260 | 0.961 | 0.017 | 47.338 | 79.475 |
| domain:agent_tools | 1500 | 0.895 | 0.034 | 48.025 | 90.649 |
| domain:documents | 1500 | 0.952 | 0.018 | 48.008 | 83.150 |
| domain:ecommerce | 1500 | 0.918 | 0.034 | 48.573 | 82.533 |
| domain:support | 1500 | 0.911 | 0.051 | 47.392 | 79.512 |
| domain:voice | 1500 | 0.903 | 0.043 | 48.686 | 79.859 |

Worst language (accuracy): `nl` — accuracy 0.816, ECE 0.041.

Worst domain (accuracy): `agent_tools` — accuracy 0.895, ECE 0.034.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| agent_tools/nl | 249 | 0.707 | 0.110 |
| ecommerce/nl | 249 | 0.763 | 0.068 |
| agent_tools/de | 249 | 0.847 | 0.082 |
| support/nl | 249 | 0.863 | 0.096 |
| voice/it | 249 | 0.871 | 0.089 |
| voice/nl | 249 | 0.871 | 0.090 |
| documents/nl | 249 | 0.876 | 0.025 |
| support/de | 249 | 0.876 | 0.086 |
| voice/fr | 249 | 0.884 | 0.052 |
| support/it | 249 | 0.888 | 0.104 |
| support/es | 252 | 0.909 | 0.061 |
| voice/es | 252 | 0.909 | 0.077 |
| documents/it | 249 | 0.920 | 0.046 |
| ecommerce/de | 249 | 0.924 | 0.071 |
| voice/pt | 252 | 0.925 | 0.074 |
| ecommerce/it | 249 | 0.928 | 0.070 |
| agent_tools/fr | 249 | 0.936 | 0.050 |
| ecommerce/es | 252 | 0.948 | 0.062 |
| documents/de | 249 | 0.956 | 0.031 |
| voice/de | 249 | 0.956 | 0.044 |
| agent_tools/es | 252 | 0.956 | 0.035 |
| agent_tools/pt | 252 | 0.956 | 0.028 |
| support/pt | 252 | 0.960 | 0.021 |
| agent_tools/it | 249 | 0.964 | 0.063 |
| documents/fr | 249 | 0.964 | 0.037 |
| support/fr | 249 | 0.968 | 0.040 |
| ecommerce/pt | 252 | 0.968 | 0.015 |
| ecommerce/fr | 249 | 0.976 | 0.026 |
| documents/pt | 252 | 0.996 | 0.036 |
| documents/es | 252 | 1.000 | 0.037 |

Worst cell (accuracy): `agent_tools/nl` — accuracy 0.707, ECE 0.110.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.911 | 0.022 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.940 | 0.044 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.952 | 0.018 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.877 | 0.088 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.841 | 0.020 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.960 | 0.027 | 200 | 195 | 25 | 0 | 0 (0.0%) |


### multilingual five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.835 | 0.103 | 47.795 | 86.344 |
| choice | 2500 | 0.896 | 0.087 | 74.488 | 96.385 |
| noul | 2500 | 0.784 | 0.123 | 44.664 | 58.523 |
| score | 2500 | 0.827 | 0.110 | 46.989 | 59.998 |
| lang:de | 1245 | 0.894 | 0.052 | 47.471 | 84.313 |
| lang:es | 1260 | 0.801 | 0.146 | 47.567 | 80.963 |
| lang:fr | 1245 | 0.922 | 0.056 | 47.431 | 81.268 |
| lang:it | 1245 | 0.798 | 0.162 | 47.681 | 90.605 |
| lang:nl | 1245 | 0.758 | 0.113 | 51.041 | 95.694 |
| lang:pt | 1260 | 0.840 | 0.120 | 47.548 | 80.496 |
| domain:agent_tools | 1500 | 0.899 | 0.045 | 47.776 | 80.247 |
| domain:documents | 1500 | 0.948 | 0.024 | 47.816 | 80.347 |
| domain:ecommerce | 1500 | 0.896 | 0.035 | 48.197 | 95.944 |
| domain:support | 1500 | 0.771 | 0.186 | 47.113 | 79.837 |
| domain:voice | 1500 | 0.663 | 0.275 | 47.881 | 78.283 |

Worst language (accuracy): `nl` — accuracy 0.758, ECE 0.113.

Worst domain (accuracy): `voice` — accuracy 0.663, ECE 0.275.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/es | 252 | 0.472 | 0.429 |
| voice/pt | 252 | 0.484 | 0.471 |
| voice/it | 249 | 0.494 | 0.478 |
| ecommerce/nl | 249 | 0.627 | 0.113 |
| support/it | 249 | 0.691 | 0.317 |
| support/nl | 249 | 0.739 | 0.245 |
| support/es | 252 | 0.742 | 0.186 |
| voice/nl | 249 | 0.743 | 0.164 |
| support/fr | 249 | 0.763 | 0.226 |
| support/de | 249 | 0.779 | 0.187 |
| agent_tools/nl | 249 | 0.819 | 0.108 |
| agent_tools/pt | 252 | 0.841 | 0.088 |
| agent_tools/es | 252 | 0.849 | 0.118 |
| documents/nl | 249 | 0.863 | 0.085 |
| voice/de | 249 | 0.863 | 0.066 |
| ecommerce/it | 249 | 0.904 | 0.064 |
| support/pt | 252 | 0.909 | 0.055 |
| documents/de | 249 | 0.924 | 0.072 |
| voice/fr | 249 | 0.928 | 0.076 |
| documents/it | 249 | 0.940 | 0.024 |
| ecommerce/de | 249 | 0.948 | 0.035 |
| agent_tools/de | 249 | 0.956 | 0.058 |
| agent_tools/it | 249 | 0.960 | 0.059 |
| ecommerce/es | 252 | 0.960 | 0.056 |
| ecommerce/fr | 249 | 0.968 | 0.018 |
| ecommerce/pt | 252 | 0.968 | 0.016 |
| agent_tools/fr | 249 | 0.972 | 0.045 |
| documents/fr | 249 | 0.980 | 0.040 |
| documents/es | 252 | 0.980 | 0.025 |
| documents/pt | 252 | 1.000 | 0.038 |

Worst cell (accuracy): `voice/es` — accuracy 0.472, ECE 0.429.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.870 | 0.066 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.755 | 0.167 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.892 | 0.092 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.745 | 0.228 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.660 | 0.166 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.781 | 0.135 | 200 | 195 | 25 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
