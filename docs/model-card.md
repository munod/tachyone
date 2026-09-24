---
license: apache-2.0
library_name: jeba
language:
  - en
  - pt
  - es
  - fr
  - de
  - multilingual
tags:
  - decision-engine
  - system-one
  - calibration
  - multilingual
  - local-first
pipeline_tag: text-classification
---

# jeba (System One decision engine)

> **Status: pre-release.** This card describes the intended release. **Weights and measured
> metrics are pending** the RTX 3060 training run (`uv sync --extra train`); numbers below are
> placeholders to be replaced by `benchmarks/report.md`.

## Model details

- **Developed by:** The jeba Authors.
- **Model type:** non-autoregressive encoder with three task distributions (`noul`, `choice`,
  `score`), answering typed questions about a state in one forward pass.
- **Trunk:** ModernBERT-large (English) and mmBERT-base (100+ languages); see ADR-0007.
- **Adapters:** [`munod/jeba-en`](https://huggingface.co/munod/jeba-en),
  [`munod/jeba-multi`](https://huggingface.co/munod/jeba-multi) (LoRA; load base + adapter).
- **Licence:** Apache-2.0.
- **Repository:** <https://github.com/munod/jeba>

## Uses

jeba answers atomic `choice` / `score` / `noul` questions about a state and returns typed values
with probabilities and `confidence`. It speaks the TypeSafe Jev `/v1/systemone` wire protocol as
a drop-in and runs **locally/offline** with no API key. Compose several atomic answers in code
rather than asking one broad question.

**Out of scope:** free-form text generation, multi-step reasoning, and any decision requiring
extended deliberation — decompose those into atomic questions and combine results in code.

## Bias, risks, and limitations

- Probabilities are only meaningful after **calibration**; the shipped temperature must be
  applied (see `docs/training.md`).
- Synthetic training data can inherit generator biases; public probes are evaluation-only.
- Confidence is a property of the distribution, not a guarantee of correctness.

## Training

Deterministic synthetic JSONL (`training/generate_data.py`) supervised with an RLCD
proper-scoring objective (`training/finetune_rlcd.py`), then temperature-calibrated on a held-out
split (`training/fit_calibration.py`). Configs and seed live under `training/configs/`.

## Evaluation

Reported by `training/evaluate.py` and rendered by `benchmarks/report.py` (accuracy, ECE, p50/p95
latency per primitive and language).

**Full-scale run (single RTX 3060 12GB):** 6,000 train / 1,500 eval deterministic synthetic
records (localized cue terms per language incl. a learnable `other` team, plus distractor
clauses), LoRA (r=16), 3 epochs, batch 16, bf16 + gradient checkpointing.

| Checkpoint | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- |
| English (ModernBERT-large + LoRA) | 0.613 | 0.059 | 17.5 |
| Multilingual (mmBERT-base + LoRA) | 0.493 | 0.034 | 12.3 |

Per primitive (English): `score` 0.890, `noul` 0.700, `choice` 0.250; (multilingual): `noul`
0.730, `score` 0.494, `choice` 0.256. `choice` remains near chance (~0.25 over four teams) —
the similarity baseline does not separate the harder team mapping; a dedicated classification
head (or the LLM backend) is the next step for that primitive. Full tables and environment are
in [`benchmarks/report.md`](https://github.com/munod/jeba/blob/main/benchmarks/report.md).

## Citation

```bibtex
@misc{jeba2026,
  title        = {jeba: a local-first System One decision engine},
  author       = {The jeba Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/munod/jeba}}
}
```
