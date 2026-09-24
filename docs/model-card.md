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

**Initial smoke run (single RTX 3060 12GB, small synthetic set, 3 epochs):**

| Checkpoint | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- |
| English (ModernBERT-large + LoRA) | 0.267 | 0.108 | 17.2 |
| Multilingual (mmBERT-base + LoRA) | 0.167 | 0.187 | 12.6 |

These are deliberately small-scale; accuracy is expected to improve substantially with more
data, epochs, and batched training. Full numbers and the environment are in
[`benchmarks/report.md`](https://github.com/munod/jeba/blob/main/benchmarks/report.md).

## Citation

```bibtex
@misc{jeba2026,
  title        = {jeba: a local-first System One decision engine},
  author       = {The jeba Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/munod/jeba}}
}
```
