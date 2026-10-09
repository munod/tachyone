# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "b37cf0ac74350b0917b7f08f244152ecd6b3c452",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.9.0",
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
| overall | 1500 | 1.000 | 0.188 | 57.929 | 92.313 |
| choice | 500 | 1.000 | 0.517 | 83.266 | 96.124 |
| noul | 500 | 1.000 | 0.048 | 55.407 | 61.335 |
| score | 500 | 1.000 | 0.000 | 57.003 | 62.131 |
| lang:en | 1500 | 1.000 | 0.188 | 57.929 | 92.313 |

Worst language (accuracy): `en` — accuracy 1.000, ECE 0.188.

Noisy view (noise_rate 0.15) — overall: accuracy 0.995, ECE 0.185, p50 57.818 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 1.000 | 0.048 | 246 | 227 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.998 | 0.191 | 58.243 | 98.361 |
| choice | 2500 | 0.996 | 0.524 | 88.016 | 101.665 |
| noul | 2500 | 1.000 | 0.048 | 55.283 | 61.175 |
| score | 2500 | 1.000 | 0.000 | 57.490 | 62.777 |
| lang:en | 7500 | 0.998 | 0.191 | 58.243 | 98.361 |
| domain:agent_tools | 1500 | 1.000 | 0.184 | 58.306 | 93.265 |
| domain:documents | 1500 | 1.000 | 0.183 | 58.362 | 92.707 |
| domain:ecommerce | 1500 | 1.000 | 0.210 | 58.230 | 101.799 |
| domain:support | 1500 | 1.000 | 0.188 | 58.367 | 92.775 |
| domain:voice | 1500 | 0.991 | 0.190 | 58.024 | 92.704 |

Worst language (accuracy): `en` — accuracy 0.998, ECE 0.191.

Worst domain (accuracy): `voice` — accuracy 0.991, ECE 0.190.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/en | 1500 | 0.991 | 0.190 |
| agent_tools/en | 1500 | 1.000 | 0.184 |
| documents/en | 1500 | 1.000 | 0.183 |
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
| overall | 7500 | 0.983 | 0.188 | 57.991 | 98.030 |
| choice | 2500 | 0.960 | 0.506 | 89.271 | 98.952 |
| noul | 2500 | 0.996 | 0.066 | 55.164 | 61.136 |
| score | 2500 | 0.992 | 0.008 | 57.235 | 62.338 |
| lang:en | 7500 | 0.983 | 0.188 | 57.991 | 98.030 |
| domain:agent_tools | 1500 | 0.999 | 0.188 | 57.931 | 93.814 |
| domain:documents | 1500 | 0.999 | 0.189 | 57.911 | 93.738 |
| domain:ecommerce | 1500 | 0.999 | 0.214 | 57.926 | 99.236 |
| domain:support | 1500 | 1.000 | 0.193 | 58.084 | 93.479 |
| domain:voice | 1500 | 0.915 | 0.178 | 58.137 | 93.904 |

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
| overall | 1500 | 0.911 | 0.047 | 52.992 | 80.309 |
| choice | 500 | 1.000 | 0.041 | 73.772 | 91.257 |
| noul | 500 | 0.860 | 0.035 | 49.198 | 59.898 |
| score | 500 | 0.872 | 0.100 | 51.917 | 59.168 |
| lang:de | 249 | 0.876 | 0.084 | 53.275 | 86.591 |
| lang:es | 252 | 0.909 | 0.057 | 52.885 | 79.238 |
| lang:fr | 249 | 0.968 | 0.046 | 52.471 | 81.025 |
| lang:it | 249 | 0.888 | 0.059 | 52.596 | 83.454 |
| lang:nl | 249 | 0.863 | 0.092 | 53.240 | 79.985 |
| lang:pt | 252 | 0.960 | 0.030 | 52.749 | 80.437 |

Worst language (accuracy): `nl` — accuracy 0.863, ECE 0.092.

Noisy view (noise_rate 0.15) — overall: accuracy 0.909, ECE 0.052, p50 52.600 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.783 | 0.103 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| es | 84 | 0.845 | 0.096 | 40 | 39 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 0.964 | 0.064 | 38 | 40 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.831 | 0.032 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.795 | 0.096 | 43 | 36 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 0.940 | 0.047 | 35 | 44 | 5 | 0 | 0 (0.0%) |


### multilingual five-domain (B-15 template-train fitted bank, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.916 | 0.018 | 53.321 | 88.260 |
| choice | 2500 | 0.956 | 0.039 | 77.721 | 96.350 |
| noul | 2500 | 0.914 | 0.023 | 50.344 | 59.443 |
| score | 2500 | 0.877 | 0.059 | 52.512 | 61.155 |
| lang:de | 1245 | 0.912 | 0.029 | 53.285 | 88.420 |
| lang:es | 1260 | 0.944 | 0.031 | 53.054 | 84.673 |
| lang:fr | 1245 | 0.945 | 0.024 | 53.164 | 85.793 |
| lang:it | 1245 | 0.914 | 0.061 | 53.323 | 89.816 |
| lang:nl | 1245 | 0.816 | 0.037 | 55.042 | 92.384 |
| lang:pt | 1260 | 0.961 | 0.023 | 52.982 | 84.313 |
| domain:agent_tools | 1500 | 0.895 | 0.020 | 53.530 | 91.247 |
| domain:documents | 1500 | 0.952 | 0.022 | 53.257 | 87.720 |
| domain:ecommerce | 1500 | 0.918 | 0.025 | 53.397 | 93.801 |
| domain:support | 1500 | 0.911 | 0.047 | 52.976 | 81.756 |
| domain:voice | 1500 | 0.903 | 0.044 | 53.410 | 85.433 |

Worst language (accuracy): `nl` — accuracy 0.816, ECE 0.037.

Worst domain (accuracy): `agent_tools` — accuracy 0.895, ECE 0.020.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| agent_tools/nl | 249 | 0.707 | 0.108 |
| ecommerce/nl | 249 | 0.763 | 0.063 |
| agent_tools/de | 249 | 0.847 | 0.080 |
| support/nl | 249 | 0.863 | 0.092 |
| voice/it | 249 | 0.871 | 0.072 |
| voice/nl | 249 | 0.871 | 0.104 |
| documents/nl | 249 | 0.876 | 0.027 |
| support/de | 249 | 0.876 | 0.084 |
| voice/fr | 249 | 0.884 | 0.055 |
| support/it | 249 | 0.888 | 0.059 |
| support/es | 252 | 0.909 | 0.057 |
| voice/es | 252 | 0.909 | 0.084 |
| documents/it | 249 | 0.920 | 0.100 |
| ecommerce/de | 249 | 0.924 | 0.075 |
| voice/pt | 252 | 0.925 | 0.068 |
| ecommerce/it | 249 | 0.928 | 0.083 |
| agent_tools/fr | 249 | 0.936 | 0.042 |
| ecommerce/es | 252 | 0.948 | 0.070 |
| documents/de | 249 | 0.956 | 0.028 |
| voice/de | 249 | 0.956 | 0.035 |
| agent_tools/es | 252 | 0.956 | 0.043 |
| agent_tools/pt | 252 | 0.956 | 0.027 |
| support/pt | 252 | 0.960 | 0.030 |
| agent_tools/it | 249 | 0.964 | 0.090 |
| documents/fr | 249 | 0.964 | 0.033 |
| support/fr | 249 | 0.968 | 0.046 |
| ecommerce/pt | 252 | 0.968 | 0.021 |
| ecommerce/fr | 249 | 0.976 | 0.016 |
| documents/pt | 252 | 0.996 | 0.038 |
| documents/es | 252 | 1.000 | 0.044 |

Worst cell (accuracy): `agent_tools/nl` — accuracy 0.707, ECE 0.108.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.911 | 0.022 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.940 | 0.044 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.952 | 0.022 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.877 | 0.099 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.841 | 0.021 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.960 | 0.027 | 200 | 195 | 25 | 0 | 0 (0.0%) |


### multilingual five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.835 | 0.075 | 53.352 | 86.891 |
| choice | 2500 | 0.896 | 0.044 | 78.664 | 98.718 |
| noul | 2500 | 0.784 | 0.111 | 49.848 | 57.961 |
| score | 2500 | 0.827 | 0.082 | 52.807 | 60.202 |
| lang:de | 1245 | 0.894 | 0.050 | 53.303 | 85.620 |
| lang:es | 1260 | 0.801 | 0.088 | 53.324 | 84.701 |
| lang:fr | 1245 | 0.922 | 0.051 | 53.218 | 84.740 |
| lang:it | 1245 | 0.798 | 0.083 | 53.334 | 91.549 |
| lang:nl | 1245 | 0.758 | 0.108 | 54.161 | 96.678 |
| lang:pt | 1260 | 0.840 | 0.080 | 52.926 | 84.852 |
| domain:agent_tools | 1500 | 0.899 | 0.041 | 52.790 | 81.766 |
| domain:documents | 1500 | 0.948 | 0.041 | 53.887 | 85.016 |
| domain:ecommerce | 1500 | 0.896 | 0.026 | 53.691 | 97.986 |
| domain:support | 1500 | 0.771 | 0.158 | 52.925 | 82.210 |
| domain:voice | 1500 | 0.663 | 0.190 | 53.569 | 81.713 |

Worst language (accuracy): `nl` — accuracy 0.758, ECE 0.108.

Worst domain (accuracy): `voice` — accuracy 0.663, ECE 0.190.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| voice/es | 252 | 0.472 | 0.304 |
| voice/pt | 252 | 0.484 | 0.332 |
| voice/it | 249 | 0.494 | 0.316 |
| ecommerce/nl | 249 | 0.627 | 0.112 |
| support/it | 249 | 0.691 | 0.240 |
| support/nl | 249 | 0.739 | 0.229 |
| support/es | 252 | 0.742 | 0.129 |
| voice/nl | 249 | 0.743 | 0.157 |
| support/fr | 249 | 0.763 | 0.218 |
| support/de | 249 | 0.779 | 0.172 |
| agent_tools/nl | 249 | 0.819 | 0.105 |
| agent_tools/pt | 252 | 0.841 | 0.097 |
| agent_tools/es | 252 | 0.849 | 0.097 |
| documents/nl | 249 | 0.863 | 0.090 |
| voice/de | 249 | 0.863 | 0.048 |
| ecommerce/it | 249 | 0.904 | 0.073 |
| support/pt | 252 | 0.909 | 0.060 |
| documents/de | 249 | 0.924 | 0.076 |
| voice/fr | 249 | 0.928 | 0.068 |
| documents/it | 249 | 0.940 | 0.085 |
| ecommerce/de | 249 | 0.948 | 0.049 |
| agent_tools/de | 249 | 0.956 | 0.044 |
| agent_tools/it | 249 | 0.960 | 0.096 |
| ecommerce/es | 252 | 0.960 | 0.065 |
| ecommerce/fr | 249 | 0.968 | 0.026 |
| ecommerce/pt | 252 | 0.968 | 0.032 |
| agent_tools/fr | 249 | 0.972 | 0.058 |
| documents/fr | 249 | 0.980 | 0.025 |
| documents/es | 252 | 0.980 | 0.031 |
| documents/pt | 252 | 1.000 | 0.038 |

Worst cell (accuracy): `voice/es` — accuracy 0.472, ECE 0.304.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.870 | 0.066 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 0.755 | 0.167 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.892 | 0.098 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.745 | 0.093 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.660 | 0.166 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.781 | 0.135 | 200 | 195 | 25 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
