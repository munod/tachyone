# tachyone Benchmark Report

## Environment

```json
{
  "git_commit": "0d26145303f2f6dcd752eb9c04a45aa2ede46bc6",
  "peft": "0.21.0",
  "platform": "Linux-6.12.108-1-MANJARO-x86_64-with-glibc2.44",
  "pydantic": "2.13.5",
  "python": "3.12.13",
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
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
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
uv run python -m benchmarks.report --entry encoder=benchmarks/results/en_split.json --out benchmarks/report.md
```

## Results

### english (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.945 | 0.034 | 23.424 | 37.989 |
| choice | 500 | 0.960 | 0.024 | 36.986 | 38.353 |
| noul | 500 | 0.992 | 0.025 | 14.838 | 23.309 |
| score | 500 | 0.882 | 0.067 | 23.201 | 25.212 |
| lang:en | 1500 | 0.945 | 0.034 | 23.424 | 37.989 |

Worst language (accuracy): `en` — accuracy 0.945, ECE 0.034.

Noisy view (noise_rate 0.15) — overall: accuracy 0.942, ECE 0.035, p50 23.499 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | 500 | 0.992 | 0.025 | 241 | 232 | 27 | 0 | 0 (0.0%) |


### multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.718 | 0.042 | 15.866 | 49.029 |
| choice | 500 | 0.468 | 0.022 | 20.246 | 55.600 |
| noul | 500 | 0.832 | 0.120 | 12.511 | 20.748 |
| score | 500 | 0.854 | 0.059 | 13.723 | 28.951 |
| lang:de | 249 | 0.687 | 0.101 | 16.443 | 55.795 |
| lang:es | 252 | 0.810 | 0.040 | 13.996 | 48.167 |
| lang:fr | 249 | 0.699 | 0.050 | 15.259 | 47.077 |
| lang:it | 249 | 0.751 | 0.044 | 16.025 | 47.652 |
| lang:nl | 249 | 0.586 | 0.110 | 19.746 | 46.429 |
| lang:pt | 252 | 0.774 | 0.025 | 14.019 | 54.784 |

Worst language (accuracy): `nl` — accuracy 0.586, ECE 0.110.

Noisy view (noise_rate 0.15) — overall: accuracy 0.713, ECE 0.047, p50 16.191 ms.
#### `noul` per language — accuracy beside the label audit

| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 83 | 0.747 | 0.200 | 33 | 46 | 4 | 0 | 0 (0.0%) |
| es | 84 | 0.940 | 0.034 | 43 | 36 | 5 | 0 | 0 (0.0%) |
| fr | 83 | 0.855 | 0.094 | 43 | 35 | 5 | 0 | 0 (0.0%) |
| it | 83 | 0.843 | 0.123 | 39 | 40 | 4 | 0 | 0 (0.0%) |
| nl | 83 | 0.663 | 0.235 | 46 | 33 | 4 | 0 | 0 (0.0%) |
| pt | 84 | 0.940 | 0.035 | 39 | 40 | 5 | 0 | 0 (0.0%) |



## See also

The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, Brier, latency, throughput, memory and contract compliance on two evaluation sets — is published in [`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from `benchmarks/README.md`.
