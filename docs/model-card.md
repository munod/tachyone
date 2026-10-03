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

> **Status: released (`v0.8.0`), revision 2026-10-02 — English [`1c88ebef`](https://huggingface.co/munod/tachyone-en/commit/1c88ebef8f15f68e8a6583c564d636221f22c29f) (B-13 mixture + P3 evidence confidence), multilingual [`3693def1`](https://huggingface.co/munod/tachyone-multi/commit/3693def1bd04d8208216f05459c5b83388ff9fd6) (card).** Trained on a
> single RTX 3060 12GB (B-5b and B-13 on NVIDIA L4 23GB) and published as LoRA adapters
> ([`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
> [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)); measured numbers below come
> from `benchmarks/report.md`.
>
> **This revision adds B-13 for the English checkpoint: the mixture retrain.** Same recipe and
> seed as the B-5 run, one factor changed — the training data: 35,540 records (the 21,000
> five-domain records **plus** 11,000 pinned public records — MultiNLI, BoolQ, Banking77, licences
> in `training/data/jev_sources.lock.json` — and 3,540 executable-rule-tree family records), then
> the identical frozen-trunk `choice`-bank fit. It sweeps both English splits **0.964 → 1.000**
> (gate strict 1.000; unseen-text rows 0.9987) — and, measured against a fresh same-recipe
> control that moves +0.6, it lifts Intelligence on the 231 public JevBench items
> (**evaluation-only, never trained on**) from **8.4 to 15.0**. The probes below are the honest
> external check: XNLI **0.341 → 0.566**, typed-decisions **0.269 → 0.367**.
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
> (`finetune_config.json` / `choice_bank_fit.json`) — the English adapter is the B-13 mixture
> bank over the B-13 joint run's frozen trunk; the multilingual adapter is the B-5b joint run
> plus the frozen-trunk head fit. Both heads-first constructions exist because training the
> corrected labels makes `noul`+`score` trivial and costs `choice` (an identical-recipe control
> landed 13 points lower, L-006) — the fix that worked was re-fitting the head on a trunk that
> already knew the data.

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
| English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) | support | **1.000** | 0.006 | 58.0 |
| English, five-domain split | 5 domains | **1.000** | 0.009 | 26.3 |
| Multilingual (mmBERT-base + LoRA r=64 + fitted bank) — routed runtime | support | **0.895** | 0.062 | 46.9 |
| Multilingual, five-domain split | 5 domains | **0.9975** | 0.0014 | 18.6 |

Per primitive (English): `choice` **1.000**, `noul` **1.000**, `score` **1.000**; (multilingual,
five-domain): `choice` **0.9924**, `noul` **1.000**, `score` **1.000**. **Label audit:** every
`noul` row is judged against its own text — **0 contradictory** in every eval set (positive
rates 0.472–0.486), per language in `benchmarks/report.md`.

**Confidence comes from evidence (P3, 2026-10-02).** The English adapter ships two extra assets:
`state_prototypes.json` (K=32 centroids of its own 21,000 training states) and
`confidence_calibration.json` (that bank plus a fitted `noul` map). A `noul` answer keeps its
direction — the cosine still decides yes/no — and takes its magnitude from
`strength = max_k cos(state, centroid)`, clamped to `[0.5, 1]`; `choice` runs at T=1.5 and
`score` is pinned at 0.1 because its answer *is* the expected value. Consequences, all
measured: accuracy is **byte-identical** to the previous revision everywhere (argmax and
expected value never move), in-domain ECE reads **0.006 / 0.009** instead of an in-sample
0.000 (L-015 — the old number was fitted on the very rows it scored), and off-domain
confidence drops to match what the model actually knows: XNLI raw ECE **0.426 → 0.264**,
typed-decisions **0.578 → 0.416**, MASSIVE **0.583 → 0.296**, JevBench's Calibration axis
**0.0 → 52.6** on the 231 public items (its recorded gate of 60 was **not** met —
`.specs/features/jevbench/spec.md` *P3 results* records why).

**What the English B-13 row is (in-sample, stated plainly).** Both English splits read 1.000
because the held-out synthetic sets share rows with training data (the known 81% `choice`
row collision) — saturation here is a ceiling, not a claim: accuracy on the 791 rows whose
text never occurs in training is **0.9987**, and the external check is the public one — the
231 JevBench items move Intelligence **8.4 → 15.0** against a fresh control's **+0.6**
(attribution table in `.specs/features/jevbench/tasks.md` JB-7), with XNLI **0.566** and
typed-decisions **0.367** as the probe deltas. The B-5 trade this row replaces is closed:
the previous bank won `choice` 0.946 → 1.000 at the cost of `noul`/`score` (0.992/0.978 →
0.946/0.946) on support; B-13 posts **1.000 across all three primitives** on both splits,
gate **strict 1.000** (0 to shared, 0 wrong domain over 2,500 `choice` rows). Full tables:
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
L-013) three of six languages remain above the 0.05 target — `nl` accuracy 0.715 / ECE 0.167,
`de` 0.083, `it` 0.056 — open under NFR-C06 / BACKLOG B-1. The CUDA-graph fast path
(`TACHYONE_FAST=1`) gives **2.54×** p50 on English (17.31 → 6.81 ms) and **3.66×** on the
multilingual five-domain path (13.97 → 3.82 ms), both with **0 top-label flips** — the gate and
the keyed heads execute after the encode, which the graphed path never sees.

**Robustness (B-4).** On a noisy view (one surface edit — typo/accents/casing — applied to 15% of
states) English drops only 1.000 → 0.997 and multilingual 0.743 → 0.744, so the released
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
