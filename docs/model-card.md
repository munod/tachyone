---
license: apache-2.0
library_name: tachyone
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

# Tachyone (System One decision engine)

> **Status: released (`v0.4.0`), revision 2026-09-28.** Trained on a single RTX 3060 12GB and
> published as LoRA adapters ([`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
> [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)); measured numbers below come
> from `benchmarks/report.md`.
>
> **This revision is B-11 (ADR-0014):** every `noul` label is derived from the text it
> accompanies (`request` → 1, `neutral`/empty → 0) instead of the loop index, so all datasets were
> regenerated and both adapters retrained. The labels contradict nothing now — the audit shipped
> with the evaluation reports **0 contradictory rows** — but that also means these numbers are
> **not comparable with pre-B-11 measurements**: the old labels contradicted 121 of 241
> request-toned English rows and left *every* `noul` label in `es`/`de`/`nl` at 0.

## Model details

- **Developed by:** The Tachyone Authors.
- **Model type:** non-autoregressive encoder with three task distributions (`noul`, `choice`,
  `score`), answering typed questions about a state in one forward pass.
- **Trunk:** ModernBERT-large (English) and mmBERT-base (100+ languages); see ADR-0007.
- **Adapters:** [`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
  [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi) (LoRA; load base + adapter).
- **License:** Apache-2.0.
- **Repository:** <https://github.com/munod/tachyone>

## Uses

Tachyone answers atomic `choice` / `score` / `noul` questions about a state and returns typed values
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
descriptions, per-record RNG, one-in-six distractor clauses), LoRA (r=16 English, r=64 multilingual)
plus a dedicated low-rank `choice` head (r=32, near-identity init); 4 epochs for English and
**8 for multilingual** (the 4-epoch run under-fit `score`: 0.648 → 0.854), batch 16, bf16 +
gradient checkpointing.

| Checkpoint | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- |
| English (ModernBERT-large + LoRA r=16 + choice head) | **0.945** | 0.034 | 23.4 |
| Multilingual (mmBERT-base + LoRA r=64 + choice head) | **0.718** | 0.042 | 15.9 |

Per primitive (English): `choice` 0.960, `noul` 0.992, `score` 0.882; (multilingual): `choice`
0.468, `noul` 0.832, `score` 0.854. **Label audit:** every `noul` row is judged against its own
text — **0 contradictory** in both eval sets (positive rates 0.482 / 0.486), per language in
`benchmarks/report.md`.

The multilingual `choice` figure is the honest cost of the fix, and it is published rather than
hidden: `choice` labels never changed, so trading ~0.20 of `choice` for ~0.22 of `noul` nets 0.718
against **0.714 for the pre-B-11 weights on the same corrected eval** (and the external probes see
the same trade — MASSIVE, a 60-way `choice` task, 0.033 → 0.013). Three of six languages meet
ECE ≤ 0.05 (`pt` 0.025, `es` 0.040, `it` 0.044); `fr` 0.0502, `de` 0.101 and `nl` 0.110 remain
above target (NFR-C06 partially open), and multilingual `noul` ECE (0.120) is declared with them.
The CUDA-graph fast path (`TACHYONE_FAST=1`) gives a 2.72× p50 speedup (9.19 → 3.37 ms) with 0
top-label flips.

**Robustness (B-4).** On a noisy view (one surface edit — typo/accents/casing — applied to 15% of
states) English drops only 0.945 → 0.942 and multilingual 0.718 → 0.713, so the released
adapters are robust to this noise model.

Full tables and environment are in
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

## Citation

```bibtex
@misc{tachyone2026,
  title        = {tachyone: a local-first System One decision engine},
  author       = {The tachyone Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/munod/tachyone}}
}
```
