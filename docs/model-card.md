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

> **Status: released, revision 2026-10-02 (B-5b).** Trained on a single RTX 3060 12GB (the
> B-5b cycle on a single NVIDIA L4 23GB) and published as LoRA adapters
> ([`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
> [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)); measured numbers below come
> from `benchmarks/report.md`.
>
> **This revision is B-5b: the multilingual checkpoint now covers five domains.** Its trunk is
> joint-trained on 30,000 multilingual five-domain records (support keeps its 18,000; four new
> domains 3,000 each) and its `choice` heads were re-fitted on the **frozen** trunk by
> `training/fit_choice_bank.py` — the same construction as the English artifact, recipe shipped
> next to the weights as `choice_bank_fit.json`. On the five-domain split it scores **0.9975**
> (previous adapter zero-shot: 0.561), gate strict **1.000**, all six languages at ECE ≤ 0.004.
>
> **Labels (B-11 + B-12, ADR-0014/ADR-0015): every label in all three primitives is derived
> from the text it accompanies** — `noul` from its phrase bank (`request` → 1,
> `neutral`/empty → 0), `score` from the tone's level (empty → middle), `choice` from the option
> the state names (empty → the catch-all `other`). All datasets were regenerated and the label
> audit published with the evaluation reports **0 contradictory rows**. These numbers are
> **not comparable with pre-B-11/pre-B-12 measurements**: the old labels contradicted 121 of 241
> request-toned English rows, left every `noul` label in `es`/`de`/`nl` at 0, and gave 7.8% of
> `score` rows a "near-tie" the text never showed.
>
> **Provenance, stated plainly:** each adapter carries its recipe next to the weights
> (`finetune_config.json` / `choice_bank_fit.json`) — the English adapter is the B-5 bank over
> run 5's frozen trunk; the multilingual adapter is the B-5b joint run plus the frozen-trunk
> head fit. Both heads-first constructions exist because training the corrected labels makes
> `noul`+`score` trivial and costs `choice` (an identical-recipe control landed 13 points lower,
> L-006) — the fix that worked was re-fitting the head on a trunk that already knew the data.

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

**Full-scale run (single RTX 3060 12GB; B-5b on a single L4 23GB):** 21,000 five-domain English /
30,000 five-domain multilingual / 18,000 support-only multilingual train / 1,500–7,500 eval
deterministic synthetic records (fully localized per language — all five domains now ship the
seven training languages, a learnable `other` option with rich descriptions, per-record RNG,
one-in-six distractor clauses), LoRA (r=16 English, r=64 multilingual) plus a dedicated low-rank
`choice` head (near-identity init); **8 epochs for multilingual**, batch 16, bf16 + gradient
checkpointing. **Both published artifacts are bank-over-frozen-trunk:** the trunk is trained
jointly, then `training/fit_choice_bank.py` re-fits the `choice` heads on cached encodings
(rank 128, 8 epochs at lr 1e-4), shipping its recipe as `choice_bank_fit.json`.

| Checkpoint | Split | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- | --- |
| English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) | support | **0.964** | 0.023 | 54.4 |
| English, five-domain split | 5 domains | **0.964** | 0.024 | 23.0 |
| Multilingual (mmBERT-base + LoRA r=64 + fitted bank) — routed runtime | support | **0.895** | 0.062 | 46.9 |
| Multilingual, five-domain split | 5 domains | **0.9975** | 0.0014 | 18.6 |

Per primitive (English): `choice` **1.000**, `noul` 0.946, `score` 0.946; (multilingual,
five-domain): `choice` **0.9924**, `noul` **1.000**, `score` **1.000**. **Label audit:** every
`noul` row is judged against its own text — **0 contradictory** in every eval set (positive
rates 0.472–0.486), per language in `benchmarks/report.md`.

**What this English row trades (published in full, not summarized away).** On the support-only
split the previous artifact scored 0.972 — `noul` 0.992, `score` 0.978, `choice` 0.946. The
five-domain bank scores 0.964 there: `choice` becomes **1.000** while `noul`/`score` give up 4.6
and 3.2 points, and in exchange the adapter covers four domains it could not answer at all
before — on the five-domain split the previous artifact scores **0.511** overall (worst domain
0.328) against this one's **0.964** (worst domain 0.963). The gate that routes each `choice`
question to its domain head scores **strict 1.000** (0 to shared, 0 wrong domain) over 2,500 rows,
and accuracy on the 473 rows whose text never occurs in training is **0.998**. Full tables:
[`docs/benchmarks.md`](https://github.com/munod/tachyone/blob/main/docs/benchmarks.md).

**The multilingual B-5b gates (fixed before training, all pass).** The previous adapter measured
**0.561** zero-shot on the new five-domain split (worst new domain `agent_tools` 0.438, support
0.8413) — the gates were `support` ≥ 0.8413, worst new domain ≥ 0.70, per-domain ECE ≤ 0.05
(exceptions declared), gate strict published. The published artifact posts support **1.0000**,
worst new domain **0.993**, per-domain ECE **0.0005–0.0038** (zero exceptions) and gate
**strict 1.000** over 2,500 `choice` rows; accuracy on the 2,996 rows whose text never occurs in
training is **0.996**. Per-language ECE on that split is **0.0010–0.0041 — all six languages
≤ 0.05** for the first time on a five-domain set; on the support-only *routed* split (13–15% of
rows fall through to the English checkpoint by language routing, `.specs/project/BACKLOG.md`
L-013) `nl` still sits at accuracy 0.715 / ECE 0.167 and remains open under NFR-C06 / BACKLOG
B-1. The CUDA-graph fast path (`TACHYONE_FAST=1`) gives **3.76×** p50 on English and **3.66×**
on the multilingual checkpoint (8.97 → 3.39 ms / 12.9 → 3.5 ms), both with **0 top-label flips**.

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
