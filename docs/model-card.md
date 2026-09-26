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

> **Status: released (`v0.2.0`).** Trained on a single RTX 3060 12GB and published as LoRA
> adapters ([`munod/jeba-en`](https://huggingface.co/munod/jeba-en),
> [`munod/jeba-multi`](https://huggingface.co/munod/jeba-multi)); measured numbers below come
> from `benchmarks/report.md`.

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

**Full-scale run (single RTX 3060 12GB):** 9,000 English / 18,000 multilingual train / 1,500 eval
deterministic synthetic records (fully localized per language, a learnable `other` team with rich
descriptions, per-record RNG, one-in-six distractor clauses), LoRA (r=16) plus a dedicated
low-rank `choice` head (r=32, near-identity init), 4 epochs, batch 16, bf16 + gradient
checkpointing.

| Checkpoint | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- |
| English (ModernBERT-large + LoRA + choice head) | 0.763 | 0.061 | 22.6 |
| Multilingual (mmBERT-base + LoRA + choice head) | 0.702 | 0.033 | 13.3 |

Per primitive (English): `choice` 0.834, `noul` 0.744, `score` 0.712; (multilingual): `choice`
0.454, `noul` 0.738, `score` 0.914. The localized, per-record-RNG data (B-1) lifted multilingual
`choice` from 0.40 to 0.73 and English overall from 0.72 to 0.78. Per-language calibration for a
few multilingual languages (`es` 0.170, `de` 0.077, `nl` 0.045) and multilingual `choice` remain
the next targets (NFR-C06 open).
The CUDA-graph fast path (`JEBA_FAST=1`) gives a 2.7× p50 speedup with 0 top-label flips.

**Robustness (B-4).** On a noisy view (one surface edit — typo/accents/casing — applied to 15% of
states) English drops only 0.763 → 0.760 and multilingual 0.702 → 0.701, so the released adapters
are already robust to this noise model. A noise-augmented multilingual adapter (trained on the same
data with `noise_rate=0.15`) reached 0.719 on both views but calibrated worse (ECE 0.090 vs 0.033)
and regressed on `de`/`nl`, so it is **not** the released adapter; see `benchmarks/report.md`.

Full tables and environment are in
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
