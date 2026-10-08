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

> **Status: released (`v0.9.0`), revision 2026-10-08 — both checkpoints are the B-15
> template-train retrain (English carries its P3 stack rebuilt on the new trunk); revision
> commits are listed in [`docs/huggingface.md`](https://github.com/munod/tachyone/blob/main/docs/huggingface.md).**
> Trained on NVIDIA L4 23GB and published as LoRA adapters
> ([`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en),
> [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)); measured numbers below come
> from `benchmarks/report.md`.
>
> **This revision is B-15: both checkpoints retrained on `template_split: "train"`, and the
> holdout number is honest for the first time.** The data recipe was corrected end to end
> (sampler strides decoupled so every language trains every label, evals made
> leave-one-template-out, phrase content ×3, volume 1.7×) and every training set regenerated so
> training and the holdout eval share **0%** of templates. The multilingual lineage carries the
> two-epoch **interleaved touch-up** that fixed the B-14 `noul` cell inversion, and the trainer
> now ships a per-cell validation monitor. On phrasing never seen in any training
> (`eval_*_domains_holdout`): **multilingual 0.8355, English 0.9825** — **+0.127 / +0.131 over
> the previously released adapters**, with the in-template → holdout gap at −0.1105 / −0.0179.
> Disclosures, never smoothed: English `choice` confidence is refitted on the rebuilt
> never-trained holdout and its pooled fit pegged the grid at **T = 10.0**, so in-domain
> `choice` ECE reads **0.5243** honestly instead of in-sample (P3's in-domain ≤ 0.05
> acceptance recorded **NOT met** for the rebuilt asset; accuracy is untouched); and **B-1's
> per-language ECE ≤ 0.05 on holdout is NOT met** for the multilingual arm (`choice` es/it/pt,
> `score` de/es/it/nl — full numbers in `.specs/project/BACKLOG.md` **B-15**).
>
> **The v0.8.0 English artifact was B-13: the mixture retrain (history).** Same recipe and
> seed as the B-5 run, one factor changed — the training data: 35,540 records (the 21,000
> five-domain records **plus** 11,000 pinned public records — MultiNLI, BoolQ, Banking77, licences
> in `training/data/jev_sources.lock.json` — and 3,540 executable-rule-tree family records), then
> the identical frozen-trunk `choice`-bank fit. It sweeps both English splits **0.964 → 1.000**
> (gate strict 1.000; unseen-text rows 0.9987) — and, measured against a fresh same-recipe
> control that moves +0.6, it lifts Intelligence on the 231 public JevBench items
> (**evaluation-only, never trained on**) from **8.4 to 15.0**. The probes below are the honest
> external check: XNLI **0.341 → 0.566**, typed-decisions **0.269 → 0.367**.
>
> **The v0.8.0 multilingual artifact was B-5b: five-domain coverage (history).** Its trunk is
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
> (`choice_bank_fit.json`) — the English adapter is the **B-15 five-domain bank** over the
> template-train trunk (`en_tt`, 6 epochs) with the P3 stack rebuilt on it; the multilingual
> adapter is the **B-15 joint run** (52,200 records, 8 epochs, then the 2-epoch interleaved
> touch-up → `multi_tt_il`) plus the frozen-trunk head fit. Both heads-first constructions exist because training the
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

**Full-scale run (NVIDIA L4 23GB):** 36,000 five-domain English / 52,200 five-domain
multilingual records generated with `template_split: "train"` (the holdout evals share **0%** of
training templates) / 1,500–7,500 eval deterministic synthetic records (fully localized per
language — all five domains ship the seven training languages, a learnable `other` option with
rich descriptions, per-record RNG, one-in-six distractor clauses), LoRA (r=16 English, r=64
multilingual) plus a dedicated low-rank `choice` head (near-identity init); **6 epochs English /
8 epochs multilingual**, batch 8 × grad-accum 4, bf16 + gradient checkpointing, and a
**per-cell validation monitor** (`cell_monitor.jsonl`) so one `(primitive, domain, language)`
collapse cannot hide behind the aggregate. **Both published artifacts are
bank-over-frozen-trunk:** the trunk is trained jointly, then `training/fit_choice_bank.py`
re-fits the `choice` heads on cached encodings (rank 128, 8 epochs at lr 1e-4), shipping its
recipe as `choice_bank_fit.json`; the multilingual trunk additionally carries the two-epoch
interleaved touch-up (`training/interleave_continue.py`).

| Checkpoint | Split | Accuracy | ECE (calibrated) | p50 (ms) |
| --- | --- | --- | --- | --- |
| English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) | support | **1.0000** | 0.188 | 57.7 |
| English, five-domain split | 5 domains | **0.9983** | 0.191 | 57.7 |
| English, **holdout phrasing** (never in training) | 5 domains | **0.9825** | 0.188 | 58.0 |
| Multilingual (mmBERT-base + LoRA r=64 + fitted bank) — routed runtime | support | **0.9107** | 0.051 | 47.4 |
| Multilingual, five-domain split | 5 domains | **0.9156** | 0.019 | 48.0 |
| Multilingual, **holdout phrasing** (never in training) | 5 domains | **0.8355** | 0.103 | 47.8 |

The English ECE column reads the **pooled never-trained fit** (below): `choice` is flattened to
T=10.0 to calibrate held-out rows, which shows up honestly in-domain (0.5243 `choice` ECE —
accuracy untouched). The multilingual rows are the **routed system**: ~13–15% of rows fall
through to the English checkpoint by language routing (empty state or no Latin-language signal
— `.specs/project/BACKLOG.md` **L-013**), which is why the five-domain multilingual row sits
below the multilingual checkpoint's own **0.9885** when it answers everything.

Per primitive (English, five-domain): `choice` **0.9956**, `noul` **0.9996**, `score`
**0.9996**; (multilingual, five-domain): `choice` **0.9564**, `noul` **0.9136**, `score`
**0.8768**; (holdout phrasing): English `choice` **0.9600** / `noul` **0.9956** / `score`
**0.9920**, multilingual `choice` **0.8960** / `noul` **0.7836** / `score` **0.8268**.
**Label audit:** every `noul` row is judged against its own text — **0 contradictory** in every
eval set (positive rates 0.472–0.486), per language in `benchmarks/report.md`.

**Confidence comes from evidence (P3 — stack rebuilt on the v0.9.0 trunk).** The English
adapter ships two extra assets: `state_prototypes.json` (K=32 spherical k-means centroids of
its own 36,000 training states) and `confidence_calibration.json` (that bank plus a fitted
`noul` map, with sha256 provenance of both inputs). A `noul` answer keeps its direction — the
cosine still decides yes/no — and takes its magnitude from
`strength = max_k cos(state, centroid)`, clamped to `[0.5, 1]`. The temperatures come from the
**rebuilt never-trained holdout** (a stride slice of the new training file plus never-trained
public rows, re-asserted against the public items before a byte was written): `choice`
**T = 10.0** — the pooled fit pegged the top of the pre-registered `DEFAULT_GRID`, a direct
consequence of training on `template_split: "train"` while the holdout keeps its hard rows —
`noul` **T = 0.75** with the map, `score` **pinned at 0.1** because its answer *is* the
expected value (the pin's own record: rounded-EV accuracy 0.53 at T=1 vs 0.9996 shipped).
Accuracy is untouched by construction (argmax and direction never move). **Recorded NOT met:**
P3's in-domain ECE ≤ 0.05 acceptance for the rebuilt asset — `choice` reads **0.5243**
in-domain under T=10.0 (the honest price of calibrating held-out rows; the fix is a fit-basis
redesign, not another fit — see `.specs/features/jevbench/spec.md` *P3 results* for why no
legal fit set can see bench difficulty). The multilingual adapter has no evidence stack; its
served `temperature_calibration.json` is fitted on **15,000 pooled in-template + holdout
predictions** (`choice` 6.0, `noul` 0.25, `score` 0.25 — fitted where the model is *not*
saturated, per L-015). The public probes were **re-measured on this v0.9.0 artifact
(2026-10-08)**: XNLI 0.566 → **0.333** (exactly chance — the B-13 mixture's MultiNLI layer was
not carried into the recomposed recipe, follow-up recorded in `.specs/project/BACKLOG.md`
**B-15**; published, never re-fixed), typed-decisions 0.367 → **0.306**, MASSIVE 0.051 →
**0.074** (chance 0.017, the best reading yet), with off-domain `ECE raw` down across the board
(**0.096 / 0.042 / 0.068**). JevBench Calibration 0.0 → 52.6 with its recorded gate of 60
**not** met remains a **v0.8.0-artifact** measurement — JevBench was set aside by decision
after its maintainer changed the submission methodology.

**What the English rows mean (in-sample, stated plainly).** The support and five-domain splits
read 1.0000 / 0.9983 because their phrasing is *seen* — training excluded the holdout template
pool, but these splits still draw the seen templates — so saturation here is a ceiling, not a
claim. The honest number is the **holdout row: 0.9825** overall (`choice` 0.9600), phrasing no
training ever emitted. The external checks quoted by earlier revisions — JevBench Intelligence
**8.4 → 15.0** against a fresh control's **+0.6** — were measured on the **v0.8.0 mixture
artifact** (JevBench set aside by decision after its maintainer changed the submission
methodology). The probes **were** re-measured on this artifact (2026-10-08): XNLI **0.333**
(chance 0.333), typed-decisions **0.306**, MASSIVE **0.074** (chance 0.017). Full tables:
[`docs/benchmarks.md`](https://github.com/munod/tachyone/blob/main/docs/benchmarks.md).

**The v0.8.0 multilingual gates (B-5b — history; v0.9.0's gates are B-15's).** The previous adapter measured
**0.561** zero-shot on the new five-domain split (worst new domain `agent_tools` 0.438, support
0.8413) — the gates were `support` ≥ 0.8413, worst new domain ≥ 0.70, per-domain ECE ≤ 0.05
(exceptions declared), gate strict published. That artifact posted support **1.0000**,
worst new domain **0.993**, per-domain ECE **0.0005–0.0038** (zero exceptions) and gate
**strict 1.000** over 2,500 `choice` rows. **v0.9.0 (B-15) posts on the unchanged current
benchmark: multilingual support 0.9760, worst new domain 0.9747, per-domain ECE 0.0019–0.0202
(zero exceptions), strict 1.000, every language ≥ baseline; overall 0.9885 / ECE 0.0069 when
the multilingual checkpoint answers everything** — the routed rows above show the pair as
served. On the support-only *routed* split (13–15% of
rows fall through to the English checkpoint by language routing, `.specs/project/BACKLOG.md`
**L-013**) the per-language story is unchanged in kind: three of six languages sit above the
0.05 ECE target on that split, and on the **holdout** the B-1 per-language ECE criterion is
**NOT met** (recorded with numbers in BACKLOG **B-1**) — open under NFR-C06 / BACKLOG B-1.
The CUDA-graph fast path
(`TACHYONE_FAST=1`) gives **2.53×** p50 on English (17.31 → 6.83 ms) and **3.71×** on the
multilingual five-domain path (14.23 → 3.84 ms) — re-measured on the v0.9.0 artifacts
(2026-10-08): **0 top-label flips** multilingual, **1 of 16 sampled** on English with max
answer-probability difference 0.024 (disclosed as measured; NFR-P01 met by both paths). The gate and
the keyed heads execute after the encode, which the graphed path never sees.

**Robustness (B-4).** On a noisy view (one surface edit — typo/accents/casing — applied to 15% of
states) English moves 1.0000 → 0.9953 and the multilingual routed pair 0.9107 → 0.9087, so the
released adapters are robust to this noise model.

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
