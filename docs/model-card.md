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

> **Status: released (`v0.4.0`), revision 2026-09-29.** Trained on a single RTX 3060 12GB and
> published as LoRA adapters ([`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
> [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)); measured numbers below come
> from `benchmarks/report.md`.
>
> **This revision is B-11 + B-12 (ADR-0014, ADR-0015): every label in all three primitives is
> derived from the text it accompanies** — `noul` from its phrase bank (`request` → 1,
> `neutral`/empty → 0), `score` from the tone's level (empty → middle), `choice` from the option
> the state names (empty → the catch-all `other`). All datasets were regenerated and the label
> audit published with the evaluation reports **0 contradictory rows**. These numbers are
> **not comparable with pre-B-11/pre-B-12 measurements**: the old labels contradicted 121 of 241
> request-toned English rows, left every `noul` label in `es`/`de`/`nl` at 0, and gave 7.8% of
> `score` rows a "near-tie" the text never showed.
>
> **Provenance, stated plainly:** the English adapter carries the **B-11 weights** and the
> multilingual adapter the **B-12 retrain** — each is the *best measured* checkpoint of its
> recipe (training on the corrected labels makes `noul`+`score` trivial and costs `choice`; an
> identical-recipe control landed 13 points lower, L-006).

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

**Full-scale run (single RTX 3060 12GB):** 21,000 five-domain English / 18,000 multilingual train
/ 1,500 eval deterministic synthetic records (fully localized per language, a learnable `other`
option with rich descriptions, per-record RNG, one-in-six distractor clauses), LoRA (r=16 English,
r=64 multilingual) plus a dedicated low-rank `choice` head (near-identity init); **8 epochs for
multilingual**, batch 16, bf16 + gradient checkpointing. The **English artifact** is the B-5 bank
(ADR-0016): run 5's six-epoch trunk kept **frozen** while the `choice` head was re-fitted as a
bank — shared + one head per domain, rank 128, 8 epochs at lr 1e-4 — by
`training/fit_choice_bank.py`, whose recipe ships next to the weights as `choice_bank_fit.json`.

| Checkpoint | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- |
| English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) | **0.964** | 0.023 | 54.4 |
| Multilingual (mmBERT-base + LoRA r=64 + choice head) | **0.743** | 0.089 | 16.0 |

Per primitive (English): `choice` **1.000**, `noul` 0.946, `score` 0.946; (multilingual): `choice`
0.468, `noul` 0.892, `score` 0.870. **Label audit:** every `noul` row is judged against its own
text — **0 contradictory** in both eval sets (positive rates 0.482 / 0.486), per language in
`benchmarks/report.md`.

**What this English row trades (published in full, not summarized away).** On the support-only
split the previous artifact scored 0.972 — `noul` 0.992, `score` 0.978, `choice` 0.946. The
five-domain bank scores 0.964 there: `choice` becomes **1.000** while `noul`/`score` give up 4.6
and 3.2 points, and in exchange the adapter covers four domains it could not answer at all
before — on the five-domain split the previous artifact scores **0.511** overall (worst domain
0.328) against this one's **0.964** (worst domain 0.963). The gate that routes each `choice`
question to its domain head scores **strict 1.000** (0 to shared, 0 wrong domain) over 2,500 rows,
and accuracy on the 473 rows whose text never occurs in training is **0.998**. Full tables:
[`docs/benchmarks.md`](https://github.com/munod/tachyone/blob/main/docs/benchmarks.md).

`choice` is the weak primitive on the multilingual side (0.468) and it is a *training* trade, not
a label problem: `choice` labels never changed, and retraining on the corrected labels makes
`noul` (1.000) and `score` (0.998) trivial — the same tone detector — while the shared trunk
starves `choice` (an identical-recipe English control landed at 0.841, the five-domain retrain's
`choice` collapsed to 0.303). The English side did **not** retrain the trunk to escape that: it
re-fitted the `choice` head on a **frozen** trunk (B-5 / ADR-0016), which lifts `choice` to 1.000
and leaves `noul`/`score` where the trunk already had them — the cost and the gain in the table
above are one artifact, not two runs (provenance: `choice_bank_fit.json` next to the weights).
One of six languages meets ECE ≤ 0.05 (`es` 0.045); `pt` 0.062,
`fr` 0.101, `de` 0.118, `nl` 0.192 and `it` 0.197 remain above target (NFR-C06), and multilingual
`choice`/`noul` ECE (0.099 / 0.108) is declared with them. The CUDA-graph fast path
(`TACHYONE_FAST=1`) gives a 2.65× p50 speedup (8.97 → 3.39 ms) with 0 top-label flips.

**Robustness (B-4).** On a noisy view (one surface edit — typo/accents/casing — applied to 15% of
states) English drops only 0.964 → 0.963 and multilingual 0.743 → 0.744, so the released
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
