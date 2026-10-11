# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "5eb26d75217410bc537e7979dc4872c7955ddbb5",
  "peft": "0.21.0",
  "platform": "Linux-6.8.0-146-generic-x86_64-with-glibc2.39",
  "pydantic": "2.13.5",
  "python": "3.12.3",
  "tachyone": "0.12.0",
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
| overall | 1500 | 1.000 | 0.017 | 58.152 | 95.652 |
| choice | 500 | 1.000 | 0.004 | 91.124 | 98.594 |
| noul | 500 | 1.000 | 0.048 | 55.987 | 62.328 |
| score | 500 | 1.000 | 0.000 | 55.009 | 61.408 |
| lang:en | 1500 | 1.000 | 0.017 | 58.152 | 95.652 |

Worst language (accuracy): `en` — accuracy 1.000, ECE 0.017.

Noisy view (noise_rate 0.15) — overall: accuracy 0.995, ECE 0.013, p50 58.526 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 1.000 | 0.048 | 246 | 227 | 27 | 0 | 0 (0.0%) |


### english five-domain (B-15 template-train fitted bank + P3 confidence, frozen trunk)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.998 | 0.016 | 58.791 | 101.389 |
| choice | 2500 | 0.996 | 0.001 | 84.824 | 102.752 |
| noul | 2500 | 1.000 | 0.048 | 56.232 | 61.897 |
| score | 2500 | 1.000 | 0.000 | 57.688 | 63.126 |
| lang:en | 7500 | 0.998 | 0.016 | 58.791 | 101.389 |
| domain:agent_tools | 1500 | 1.000 | 0.015 | 59.056 | 95.739 |
| domain:documents | 1500 | 1.000 | 0.013 | 58.981 | 96.862 |
| domain:ecommerce | 1500 | 1.000 | 0.016 | 59.003 | 103.292 |
| domain:support | 1500 | 1.000 | 0.017 | 58.536 | 94.911 |
| domain:voice | 1500 | 0.991 | 0.030 | 58.340 | 96.189 |

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
| overall | 7500 | 0.983 | 0.032 | 59.156 | 102.119 |
| choice | 2500 | 0.960 | 0.037 | 93.307 | 103.063 |
| noul | 2500 | 0.996 | 0.066 | 56.666 | 62.113 |
| score | 2500 | 0.992 | 0.008 | 58.462 | 63.168 |
| lang:en | 7500 | 0.983 | 0.032 | 59.156 | 102.119 |
| domain:agent_tools | 1500 | 0.999 | 0.017 | 59.216 | 96.508 |
| domain:documents | 1500 | 0.999 | 0.015 | 59.372 | 97.117 |
| domain:ecommerce | 1500 | 0.999 | 0.016 | 59.550 | 103.579 |
| domain:support | 1500 | 1.000 | 0.019 | 58.794 | 96.524 |
| domain:voice | 1500 | 0.915 | 0.120 | 58.658 | 96.183 |

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
| overall | 1500 | 0.999 | 0.006 | 54.481 | 93.320 |
| choice | 500 | 0.998 | 0.004 | 81.845 | 95.882 |
| noul | 500 | 1.000 | 0.001 | 51.753 | 56.705 |
| score | 500 | 1.000 | 0.013 | 53.862 | 58.648 |
| lang:de | 249 | 1.000 | 0.007 | 54.453 | 93.320 |
| lang:es | 252 | 1.000 | 0.006 | 54.249 | 92.157 |
| lang:fr | 249 | 1.000 | 0.012 | 54.503 | 93.357 |
| lang:it | 249 | 1.000 | 0.003 | 54.801 | 94.478 |
| lang:nl | 249 | 1.000 | 0.003 | 53.607 | 92.647 |
| lang:pt | 252 | 0.996 | 0.005 | 55.169 | 94.128 |

Worst language (accuracy): `pt` — accuracy 0.996, ECE 0.005.

Noisy view (noise_rate 0.15) — overall: accuracy 0.993, ECE 0.005, p50 53.901 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 1.000 | 0.000 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| es | 84 | 1.000 | 0.000 | 40 | 39 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 1.000 | 0.000 | 38 | 40 | 5 | 0 | 0 (0.0%) |
| it | 83 | 1.000 | 0.000 | 37 | 42 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 1.000 | 0.004 | 43 | 36 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 1.000 | 0.000 | 35 | 44 | 5 | 0 | 0 (0.0%) |


### multilingual five-domain (B-17 teacher-expanded banks, B-15 recipe)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.996 | 0.003 | 53.236 | 87.361 |
| choice | 2500 | 0.993 | 0.004 | 80.225 | 89.508 |
| noul | 2500 | 0.999 | 0.001 | 50.918 | 56.248 |
| score | 2500 | 0.996 | 0.008 | 52.597 | 57.883 |
| lang:de | 1245 | 0.993 | 0.005 | 53.040 | 87.543 |
| lang:es | 1260 | 1.000 | 0.006 | 53.312 | 87.586 |
| lang:fr | 1245 | 0.994 | 0.007 | 53.650 | 87.338 |
| lang:it | 1245 | 0.995 | 0.001 | 53.162 | 88.035 |
| lang:nl | 1245 | 0.996 | 0.000 | 53.074 | 86.790 |
| lang:pt | 1260 | 0.997 | 0.003 | 53.211 | 86.947 |
| domain:agent_tools | 1500 | 0.995 | 0.003 | 52.718 | 83.510 |
| domain:documents | 1500 | 0.996 | 0.004 | 54.249 | 85.234 |
| domain:ecommerce | 1500 | 0.993 | 0.003 | 52.286 | 88.716 |
| domain:support | 1500 | 0.999 | 0.006 | 53.405 | 83.899 |
| domain:voice | 1500 | 0.995 | 0.003 | 53.626 | 85.157 |

Worst language (accuracy): `de` — accuracy 0.993, ECE 0.005.

Worst domain (accuracy): `ecommerce` — accuracy 0.993, ECE 0.003.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| agent_tools/fr | 249 | 0.980 | 0.004 |
| ecommerce/de | 249 | 0.980 | 0.016 |
| documents/de | 249 | 0.988 | 0.008 |
| voice/it | 249 | 0.988 | 0.009 |
| voice/nl | 249 | 0.988 | 0.010 |
| ecommerce/pt | 252 | 0.992 | 0.005 |
| agent_tools/de | 249 | 0.996 | 0.013 |
| agent_tools/it | 249 | 0.996 | 0.002 |
| documents/fr | 249 | 0.996 | 0.006 |
| documents/it | 249 | 0.996 | 0.000 |
| documents/nl | 249 | 0.996 | 0.006 |
| ecommerce/fr | 249 | 0.996 | 0.008 |
| ecommerce/it | 249 | 0.996 | 0.000 |
| ecommerce/nl | 249 | 0.996 | 0.001 |
| voice/fr | 249 | 0.996 | 0.012 |
| agent_tools/pt | 252 | 0.996 | 0.002 |
| support/pt | 252 | 0.996 | 0.005 |
| agent_tools/es | 252 | 1.000 | 0.004 |
| agent_tools/nl | 249 | 1.000 | 0.003 |
| documents/es | 252 | 1.000 | 0.006 |
| documents/pt | 252 | 1.000 | 0.006 |
| ecommerce/es | 252 | 1.000 | 0.005 |
| support/de | 249 | 1.000 | 0.007 |
| support/es | 252 | 1.000 | 0.006 |
| support/fr | 249 | 1.000 | 0.012 |
| support/it | 249 | 1.000 | 0.003 |
| support/nl | 249 | 1.000 | 0.003 |
| voice/de | 249 | 1.000 | 0.007 |
| voice/es | 252 | 1.000 | 0.006 |
| voice/pt | 252 | 1.000 | 0.003 |

Worst cell (accuracy): `ecommerce/de` — accuracy 0.980, ECE 0.016.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.998 | 0.002 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 1.000 | 0.000 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 1.000 | 0.000 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.995 | 0.004 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 1.000 | 0.005 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 1.000 | 0.001 | 200 | 195 | 25 | 0 | 0 (0.0%) |


### multilingual five-domain, holdout phrasing (never in training)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | 0.983 | 0.006 | 53.779 | 88.386 |
| choice | 2500 | 0.981 | 0.005 | 80.816 | 89.890 |
| noul | 2500 | 0.992 | 0.005 | 51.741 | 56.484 |
| score | 2500 | 0.975 | 0.009 | 53.126 | 57.586 |
| lang:de | 1245 | 0.972 | 0.007 | 53.725 | 88.594 |
| lang:es | 1260 | 0.983 | 0.005 | 53.653 | 87.971 |
| lang:fr | 1245 | 0.976 | 0.012 | 53.982 | 88.369 |
| lang:it | 1245 | 0.993 | 0.003 | 53.834 | 89.296 |
| lang:nl | 1245 | 0.986 | 0.007 | 53.606 | 87.304 |
| lang:pt | 1260 | 0.987 | 0.003 | 53.882 | 88.618 |
| domain:agent_tools | 1500 | 0.983 | 0.010 | 53.686 | 85.445 |
| domain:documents | 1500 | 0.987 | 0.007 | 53.842 | 85.078 |
| domain:ecommerce | 1500 | 0.967 | 0.015 | 54.482 | 89.688 |
| domain:support | 1500 | 0.995 | 0.007 | 53.574 | 83.834 |
| domain:voice | 1500 | 0.983 | 0.011 | 53.233 | 84.455 |

Worst language (accuracy): `de` — accuracy 0.972, ECE 0.007.

Worst domain (accuracy): `ecommerce` — accuracy 0.967, ECE 0.015.
#### domain x language

| Cell | n | Accuracy | ECE |
| --- | --- | --- | --- |
| ecommerce/de | 249 | 0.871 | 0.061 |
| agent_tools/fr | 249 | 0.912 | 0.065 |
| documents/es | 252 | 0.925 | 0.059 |
| voice/nl | 249 | 0.936 | 0.046 |
| ecommerce/pt | 252 | 0.964 | 0.024 |
| support/pt | 252 | 0.976 | 0.013 |
| voice/fr | 249 | 0.980 | 0.011 |
| voice/it | 249 | 0.984 | 0.007 |
| ecommerce/it | 249 | 0.988 | 0.007 |
| agent_tools/it | 249 | 0.992 | 0.006 |
| ecommerce/fr | 249 | 0.992 | 0.007 |
| ecommerce/nl | 249 | 0.992 | 0.012 |
| ecommerce/es | 252 | 0.992 | 0.010 |
| agent_tools/de | 249 | 0.996 | 0.006 |
| documents/fr | 249 | 0.996 | 0.005 |
| support/de | 249 | 0.996 | 0.002 |
| voice/de | 249 | 0.996 | 0.016 |
| agent_tools/pt | 252 | 0.996 | 0.003 |
| agent_tools/es | 252 | 1.000 | 0.007 |
| agent_tools/nl | 249 | 1.000 | 0.003 |
| documents/de | 249 | 1.000 | 0.012 |
| documents/it | 249 | 1.000 | 0.003 |
| documents/nl | 249 | 1.000 | 0.005 |
| documents/pt | 252 | 1.000 | 0.006 |
| support/es | 252 | 1.000 | 0.018 |
| support/fr | 249 | 1.000 | 0.009 |
| support/it | 249 | 1.000 | 0.004 |
| support/nl | 249 | 1.000 | 0.006 |
| voice/es | 252 | 1.000 | 0.007 |
| voice/pt | 252 | 1.000 | 0.011 |

Worst cell (accuracy): `ecommerce/de` — accuracy 0.871, ECE 0.061.

#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 415 | 0.998 | 0.002 | 194 | 201 | 20 | 0 | 0 (0.0%) |
| es | 420 | 1.000 | 0.000 | 189 | 206 | 25 | 0 | 0 (0.0%) |
| fr | 415 | 0.995 | 0.005 | 176 | 214 | 25 | 0 | 0 (0.0%) |
| it | 415 | 0.990 | 0.007 | 200 | 195 | 20 | 0 | 0 (0.0%) |
| nl | 415 | 0.973 | 0.017 | 206 | 189 | 20 | 0 | 0 (0.0%) |
| pt | 420 | 0.998 | 0.002 | 200 | 195 | 25 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
