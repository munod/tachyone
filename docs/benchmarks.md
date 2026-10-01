# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

For the **head-to-head** against the open System One scorer and two local LLMs (accuracy, ECE,
Brier, latency, throughput, memory and contract compliance on two evaluation sets), see
[`compare.md`](compare.md).

**Setup:** single RTX 3060 12GB · 9,000 English / 18,000 multilingual / 21,000 five-domain
deterministic synthetic train records (fully localized per language, per-record RNG) · held-out
evals of 1,500 / 1,500 / 7,500 · LoRA (r=16 English, **r=64 multilingual**) plus a low-rank
`choice` head (r=32; r=128 for the five-domain run) · 4 epochs English, **8 multilingual**, 6
five-domain · bf16 + gradient checkpointing. Calibrated ECE is after per-`(primitive, language)`
temperature fitting on the held-out split (in-sample). The **noisy view** applies one surface edit
(typo/accents/casing) to 15% of states (B-4).

**Labels (B-11 + B-12 / [ADR-0014](adr/ADR-0014-noul-label-from-text.md) +
[ADR-0015](adr/ADR-0015-score-choice-labels-from-text.md)):** every label in all three primitives is
derived from the text it accompanies — `noul`: `request` → 1, `neutral`/empty → 0; `score`: the
level of the tone the state was written in, empty → the middle level; `choice`: the option the
state names, empty → the catch-all `other` (which `ecommerce` gained as a fifth option). The
pre-B-11 `noul` rule was index-parity (every `es`/`de`/`nl` label at 0, 121 of 241 English
request-toned rows contradicted) and `score` carried an `index % 13` near-tie that contradicted
7.8% of rows. All datasets were regenerated and every table below ships with a label audit
reporting **0 contradictory rows**; numbers measured before either fix live in
`.specs/project/BACKLOG.md` **B-11**/**B-12** and are **not** comparable line-by-line with these.

**Which weights are published.** The B-12 cycle retrained everything and kept **the best measured
checkpoint per adapter**, because training on the corrected labels makes `noul` and `score`
trivially learnable (both hit 1.000 — they are the same tone detector once the near-tie noise is
gone) and the shared trunk pays for it in `choice`: an English control of the *identical* published
recipe landed at 0.841 (L-006), and the five-domain retrain's `choice` collapsed to 0.303. So the
English and five-domain adapters publish the **B-11 weights** (trained where `score` still had the
7.8% near-tie noise they now beat) and the multilingual adapter publishes the **B-12 retrain**
(it wins on merit: 0.743 vs 0.739). Provenance is stated with every table.

## English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) — B-5 artifact

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.964** | 0.023 | 54.384 | 90.428 |
| `choice` | 500 | **1.000** | 0.000 | 86.316 | 92.598 |
| `noul` | 500 | 0.946 | 0.038 | 50.858 | 56.172 |
| `score` | 500 | 0.946 | 0.030 | 53.668 | 58.860 |

Noisy view (noise_rate 0.15): overall accuracy **0.963**, ECE **0.024**.

> **Latency re-measured 2026-10-01 on the L4 reference box.** These two columns are *not*
> comparable with the 23 ms this table showed before — that was a different box, and today the
> adapter it replaces measures **51.8 ms** p50 on the same path in the same session against these
> **54.4 ms**, so the new artifact costs ≈ +5%, not 2.4×. Accuracy and ECE are machine-independent
> and are what the comparison rests on.

`noul` label audit of the same 500 rows: **0 contradictory** (241 request / 232 neutral / 27 empty,
0 unreadable), positive rate 0.482. The whole history of this table in six rows — weights ×
labels, because neither half is comparable on its own:

| English checkpoint × labels | overall | `noul` | `choice` | `score` |
| --- | ---: | ---: | ---: | ---: |
| B-11 weights × pre-B-11 labels (what `v0.4.0` published) | 0.859 | 0.718 | 0.948 | 0.910 |
| B-11 weights × B-11 labels (ADR-0014) | 0.945 | 0.992 | 0.960 | 0.882 |
| B-11 weights × B-12 labels (previous published) | 0.972 | 0.992 | 0.946 | 0.978 |
| B-12 retrain × B-12 labels | 0.962 | 1.000 | 0.888 | 0.998 |
| control — identical recipe, seed 2 | 0.841 | 1.000 | 0.732 | 0.792 |
| **B-5 five-domain bank × B-12 labels — published** | **0.964** | 0.946 | **1.000** | 0.946 |

**The trade this row makes, stated plainly.** The published English adapter is now the five-domain
artifact: on this support-only split it **loses** 0.008 overall (`noul` 0.992 → 0.946,
`score` 0.978 → 0.946) and **gains** `choice` 0.946 → 1.000 — and on the five-domain split it goes
from 0.511 to 0.964 (see the multi-domain section below). Both halves are published; neither is
hidden inside a single headline number.

Two lessons are visible in it: the labels were the cap, not the model (`noul` 0.718 → 0.946 by
relabeling alone, `score` 0.882 → 0.978 by the same weights on corrected labels); and the
retraining *gamble* is real — the control differs from the retrain only in `out_dir` and lands 13
points lower.

## Multilingual (mmBERT-base + LoRA r=64 + choice head, 8 epochs) — B-12 retrain

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.743** | 0.089 | 16.021 | 46.240 |
| `choice` | 500 | 0.468 | 0.099 | 19.327 | 53.931 |
| `noul` | 500 | 0.892 | 0.108 | 11.616 | 20.368 |
| `score` | 500 | 0.870 | 0.076 | 12.871 | 26.652 |
| lang `es` | 252 | 0.921 | 0.045 | 15.200 | 47.337 |
| lang `de` | 249 | 0.799 | 0.118 | 16.437 | 53.958 |
| lang `pt` | 252 | 0.778 | 0.062 | 13.141 | 53.890 |
| lang `nl` | 249 | 0.675 | 0.192 | 19.012 | 45.327 |
| lang `fr` | 249 | 0.667 | 0.101 | 15.206 | 45.827 |
| lang `it` | 249 | 0.618 | 0.197 | 15.962 | 45.433 |

Noisy view (noise_rate 0.15): overall accuracy **0.744**, ECE **0.092**.

**Why this table is not comparable with the 0.853 it replaces.** The pre-B-11 labels left *every*
`noul` row in `es`/`de`/`nl` at 0 (L-010), so 0.960 `noul` and 0.853 overall were partly a
language→label shortcut. The full ladder, weights × labels:

| multilingual checkpoint × labels | overall | ECE | `noul` | `choice` | `score` |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre-B-11 weights × pre-B-11 labels (`v0.4.0`) | 0.853 | 0.038 | 0.960 | 0.684 | 0.916 |
| pre-B-11 weights × B-11 labels | 0.714 | 0.076 | 0.614 | **0.674** | 0.854 |
| B-11 weights (8 ep) × B-11 labels | 0.718 | 0.042 | 0.832 | 0.468 | 0.854 |
| B-11 weights × B-12 labels | 0.739 | 0.066 | 0.892 | 0.454 | 0.870 |
| **B-12 retrain × B-12 labels — published** | **0.743** | 0.089 | **0.892** | **0.468** | **0.870** |

The B-12 retrain is the only checkpoint in the cycle that wins on merit *and* has clean
provenance, so it is the published one; its ECE (0.089 vs 0.066 for the weights it replaces) is
worse and declared.

**Per language:** `noul` accuracy reads es 1.000, pt 1.000, fr 0.916, it 0.892, de 0.795,
nl 0.747 against **0 contradictory labels** in all six (the audit's request/neutral/empty split is
33–46 / 33–46 / 4–5 per language). The corrected labels are *easy* where the model reads tone —
and the ECE says the calibration did not follow: only `es` (0.045) meets **ECE ≤ 0.05**;
`pt` 0.062, `fr` 0.101, `de` 0.118, `nl` 0.192 and `it` 0.197 are above (NFR-C06, 1 of 6, worse
than the 3 of 6 the B-11 cycle reported).

## Fast path (CUDA graphs)

mmBERT + `checkpoints/multi`, 189 held-out states, batch=1; `TACHYONE_FAST=1` uses per-shape CUDA
graphs with bf16-resident weights. Capture is warmed before timing. Re-measured 2026-10-01 on the
L4 reference box (`benchmarks/results/fast_path.json`) — latency is box-bound (the columns before
this date were the RTX 3060 build); the speedup ratio and the parity are not.

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 14.381 | 16.549 | 68.8 | 158.4 | 76.2 |
| fast (CUDA graphs, bf16) | 3.823 | 3.930 | 250.7 | 572.3 | 208.9 |

**Parity:** max absolute answer-probability difference **0.000372** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **3.76× p50 speedup** for the fast path (2.65× on the 3060 build).

**The released B-5 artifact was checked in the same pass** (ModernBERT + the keyed bank,
`benchmarks/results/fast_path_en.json`): **0 top-label flips**, max answer difference
**0.000649**, **2.59×** p50 (17.63 → 6.81 ms). The gate and the per-domain heads run *after* the
encode, so the CUDA-graph path neither sees nor perturbs them — which is what ADR-0016 predicted
when it kept the head as plain JSON computed post-encode.

## Public probes (B-7)

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. Full tables (including per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | **0.269** | 0.20–0.50 | 0.542 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | 0.011 | 0.017 | 0.354 |
| XNLI (`en`) | CC BY-NC 4.0 | 5010 | **0.341** | 0.333 | 0.650 |

Evaluation only — no probe row has ever reached `training/`. Read each number against its
**chance** level and against the in-sample 0.964 above, not against the other rows: these are the
released adapters zero-shot on tasks they were never trained for.

**Re-run on the B-5 artifact (2026-10-01, right after `munod/tachyone-en` `c00c174d`).** The
multilingual adapter did not move; the English head did. Accuracy is flat-to-slightly-down and
still near chance: typed-decisions **0.330 → 0.269** (best config 0.412 → 0.344, worst 0.250 →
0.188), XNLI **0.333 → 0.341** (its chance is 0.333), MASSIVE **unchanged at 0.011 against the
0.017 chance** (per language 0.004–0.039 over 60 intents). What visibly moved is **confidence**:
`Conf` 0.399 → 0.811 (typed-decisions), 0.354 → **0.991** (XNLI) and 0.021 → 0.365 (MASSIVE), so
`ECE raw` goes 0.543 → 0.542, 0.268 → **0.650** and 0.192 → **0.354**, `Brier` 0.699 → 1.188,
0.670 → 1.304 and 0.983 → 1.272 — while all three temperature fits still pin T=20.0 and can no
longer buy it back (`ECE cal` 0.069 → 0.170, 0.021 → 0.294, 0.010 → 0.016). The B-5 head was
fitted to be sharply right *in* domain; *off* domain it is now sharply confident. That is the same
trade the in-domain table buys, published here instead of absorbed — MASSIVE is still the 60-way
`choice` task whose weakest cell is the multilingual `choice` head (0.468), so the external probe
and the internal one still tell the same story.

## Multi-domain experiment (B-5a / ADR-0016) — **released 2026-10-01**

Five domains trained into one adapter — `support`, `ecommerce`, `agent_tools`, `documents`,
`voice` — English only, 21,000 records, evaluated on `data/eval_en_domains.jsonl` (7,500 rows;
its `support` half is the same records as `eval_en.jsonl`, so **0.964** above is the comparable
support number). **Released as `munod/tachyone-en` (2026-10-01):** run 5's trunk kept frozen and
its `choice` head re-fitted as a **bank** — shared + five domain heads behind a deterministic gate
(ADR-0016), 8 epochs at lr 1e-4 against the B-12-corrected labels.

| Domain | support-only (previous) | run 5 (unreleased) | **B-5 bank — published** |
| --- | ---: | ---: | ---: |
| `support` | **0.972** | 0.886 | **0.964** |
| `agent_tools` | 0.328 | 0.935 | **0.964** |
| `documents` | 0.359 | 0.915 | **0.963** |
| `ecommerce` | 0.358 | 0.842 | **0.964** |
| `voice` | 0.539 | 0.815 | **0.963** |
| **overall** | 0.511 | 0.879 | **0.964** |

| Primitive | support-only | run 5 | **B-5 bank** | ECE (bank) |
| --- | ---: | ---: | ---: | ---: |
| `choice` | 0.452 | 0.745 | **0.9996** | 0.0003 ✅ |
| `noul` | 0.697 | 0.946 | 0.946 | 0.042 ✅ |
| `score` | 0.385 | 0.946 | 0.946 | 0.029 ✅ |

**Gates (ADR-0016 cycle, four arms on one harness).** Worst new domain ≥ 0.70 → **pass** (0.963).
Support ≥ 0.85 → **pass** (0.964). Gap to the released adapter's `support` cell →
**0.086 → 0.008**, i.e. 91% closed by moving the head alone. The gate's own number → **strict
1.000, 0 fell to shared, 0 wrong domain** over 2,500 `choice` rows, and the oracle arm
(`--head-hint domain`) is numerically identical — routing was never the bottleneck. Per-domain ECE
**0.021–0.027: all five domains under the 0.05 target for the first time** (was 0.036–0.080 with
three declared exceptions).

**What actually closed it — read this before quoting the table.** Two arms were fitted on the
*frozen* run-5 trunk from a single encode pass: a shared-head **control** and the **bank**. The
control lands at 0.9633 overall against the bank's 0.9637 — **3 of 2,500 `choice` rows**. The gap
closed because *one* head was re-fitted on the corrected labels, not because of per-domain capacity
(lesson L-012): the control exists precisely to make the structural claim falsifiable, and it
falsified it. What the bank adds is the forward-compatible asset format (B-5b's language keys
reuse it unchanged) and a gate that measures 1.000 at no cost. On the 473 `choice` rows whose text
never occurs in training the ranking holds — support-only weights 0.892 → control 0.992 → **bank
0.998** — so the gain is not the eval/train row collision (81% of `choice` rows are byte-identical
to a training row) buying a memorized answer.

What the six original runs established still stands: the binding constraint on `choice` is neither
capacity nor time — `choice_rank` 32 → 128 → 256 with 4 → 6 → 8 epochs moved support 0.655 →
0.785 → 0.795 on the pre-B-11 labels while the fourth-option leak simply **moved between runs**.
The full six-run curve, the per-(domain, primitive) cross tables, the isolate's four arms and the
three structural options live in `.specs/project/BACKLOG.md` **B-5**.

## Known limitations

- Per-language ECE meets the `0.05` target for **one of six** multilingual languages (`es` 0.045);
  `pt` 0.062, `fr` 0.101, `de` 0.118, `nl` 0.192 and `it` 0.197 are above it (NFR-C06). Multilingual
  `choice` ECE 0.099, `noul` ECE 0.108 — all declared.
- **`choice` is the weak primitive everywhere multilingual** (0.468): the corrected labels made
  `noul`/`score` easy and the shared trunk spends itself on them. The English side fixed its own
  `choice` by **re-fitting the head on a frozen trunk** (0.745 → 0.9996) — but that fix cost
  `noul`/`score` on support (0.992/0.978 → 0.946/0.946), which is why the published row now shows
  both numbers instead of the best one. The multilingual `choice` fix is backlog `B-1`, and B-5b
  (bank keys per language) reuses the asset format shipped here.
- Calibrated ECE is measured in-sample on the held-out synthetic split. The public probes above
  are the opposite case — public distributions, evaluation-only, no shared states with training.
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.236** while scoring **0.972** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
