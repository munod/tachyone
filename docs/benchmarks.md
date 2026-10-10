# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

For the **head-to-head** against the open System One scorer and two local LLMs (accuracy, ECE,
Brier, latency, throughput, memory and contract compliance on two evaluation sets), see
[`compare.md`](compare.md).

**Setup (v0.9.0, B-15):** NVIDIA L4 23GB · 12,000 English support / 36,000 English
five-domain / 23,400 multilingual support / 52,200 multilingual five-domain deterministic
synthetic train records, all generated with `template_split: "train"` (the holdout evals share
**0%** of training templates; fully localized per language, per-record RNG) · evals of
1,500 / 1,500 support rows (en / multi), 7,500 / 7,500 five-domain rows, **7,500 / 7,500
holdout rows (phrasing never in training)** and 7,500 / 7,500 in-template rows · LoRA
(r=16 English, **r=64 multilingual**) plus a low-rank `choice` head (r=128 for both five-domain
runs) · **6 epochs English / 8 multilingual** (the multi trunk also carries a 2-epoch
interleaved touch-up, `training/interleave_continue.py`) · the published multilingual artifact
= that trunk + `choice` heads re-fitted on the **frozen** trunk (rank 128, 8 epochs, lr 1e-4;
recipe in `choice_bank_fit.json`) · bf16 + gradient checkpointing + a per-cell validation
monitor. Calibrated ECE is the confidence the runtime actually ships: the English adapter
carries the **P3 stack rebuilt on the new trunk** (holdout regenerated 2026-10-08 against the
new training file; `choice` **T=10.0** — the pooled fit pegged the grid, in-domain `choice`
ECE **0.5243** recorded **NOT met** with accuracy untouched — `noul` 0.75, `score` pinned 0.1)
and the multilingual one a temperature fitted on **15,000 pooled in-template + holdout
predictions** (`choice` 6.0, `noul` 0.25, `score` 0.25 — never in-sample, L-015). Tables in
the dated sections below were measured on the **v0.8.0 artifacts** (single RTX 3060 12GB for
the speed rows) and are kept as history. The **noisy view** applies one surface edit
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
English adapter publishes the **B-5 bank over run 5's trunk** (a trunk trained before B-11/B-12 —
the retrain-on-corrected-labels option was tried and lost, BACKLOG B-12) and the multilingual
adapter now publishes the **B-5b fitted bank** — the 30k five-domain joint run plus `choice`
heads re-fitted on its frozen trunk — superseding the B-12 retrain (its weights × labels ladder
lives in `.specs/project/BACKLOG.md` **B-12**). Provenance is stated with every table.

## v0.9.0 — B-15: template-train retrain and the honest holdout

Both checkpoints retrained on `template_split: "train"` after the data-recipe corrections
(sampler strides decoupled, leave-one-template-out evals, content ×3, volume 1.7×). Measured
through the **routed runtime with both local checkpoints pinned** — the router sends empty or
no-signal states to the English checkpoint by design (**L-013**, ~13–15% of five-domain
multilingual rows), so the measured pair is the published pair.

| Checkpoint | Split | Overall | ECE | p50 (ms) |
| --- | --- | --- | --- | --- |
| English, support split | 1 domain | **1.0000** | 0.188 | 57.7 |
| English, five-domain | 5 domains | **0.9983** | 0.191 | 57.7 |
| English, **holdout phrasing** | 5 domains | **0.9825** | 0.188 | 58.0 |
| Multilingual, support split (routed) | 1 domain | **0.9107** | 0.051 | 47.4 |
| Multilingual, five-domain (routed) | 5 domains | **0.9156** | 0.019 | 48.0 |
| Multilingual, **holdout phrasing** (routed) | 5 domains | **0.8355** | 0.103 | 47.8 |

Per language — five-domain current → holdout (accuracy / ECE):

| lang | current | holdout |
| --- | --- | --- |
| `pt` | 0.9611 / 0.0173 | 0.8405 / 0.1204 |
| `fr` | 0.9454 / 0.0238 | 0.9221 / 0.0556 |
| `es` | 0.9444 / 0.0280 | 0.8008 / 0.1458 |
| `it` | 0.9141 / 0.0523 | 0.7976 / 0.1622 |
| `de` | 0.9116 / 0.0262 | 0.8940 / 0.0519 |
| `nl` | 0.8161 / 0.0413 | 0.7582 / 0.1127 |
| `en` (English ckpt) | 0.9983 / 0.1906 | 0.9825 / 0.1884 |

**Gates.** Current benchmark — all pre-registered gates PASS (multilingual `support` 0.9760 ≥
0.7887, worst domain 0.9747 ≥ 0.70, per-domain ECE 0.0019–0.0202 ≤ 0.05 with **zero
exceptions**, strict 1.000, every language ≥ baseline; English `support` 1.0000 ≥ 0.9733).
Holdout — B-1's `choice` ≥ 0.60 PASS (0.9248), English overall ≥ 0.72 PASS, the `noul`
per-language tripwire ≥ 0.60 PASS everywhere (worst 0.7807), `wrong_domain` = 0 with
`fell_to_shared` = 0. **NOT met, recorded:** B-1's per-language ECE ≤ 0.05 for
`choice`/`score` on holdout (multilingual: `choice` es/it/pt, `score` de/es/it/nl, worst
`score/it` 0.1495; English passes at 0.0155 / 0.0067) and P3's in-domain ≤ 0.05 for the
rebuilt confidence asset (English `choice` 0.5243). Never re-fixed — full trail in
`.specs/project/BACKLOG.md` **B-15**.

**Reference points.** The previously released adapters score **0.7623** (multilingual) and
**0.8508** (English) on the same holdout rows — B-15 is **+0.127 / +0.131** over them. The
v0.8.0 arm's holdout score (0.9876 / 0.9995) is **in-sample** (it trained on all templates)
and never was a generalization number. The in-template → holdout gap for B-15 itself is
**−0.1105** (multilingual) / **−0.0179** (English). Full report with reproduction commands:
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

## v0.11.0 — B-17: teacher bank expansion — B-1 closes (2026-10-10)

Part 2 of the generalization cycle: **352 teacher-authored phrases** (`qwen3.6:35b` through
the local Ollama, 8 urgency-filtered) appended to every `neutral`/`request` bank — five
domains × six non-English languages, inserted **before** the held-out last phrase, so the
gate rows never moved (holdout sha pins intact). Multi retrained with the unchanged B-15
recipe (8 epochs + interleaved touch-up); English ran no training. Gates pre-registered in
`cd69459`, **15/15 PASS**; full record in BACKLOG **B-17**.

| Axis | B-16 (2026-10-09) | B-17 (2026-10-10) |
| --- | --- | --- |
| **B-1 cells ≤ 0.05 (holdout)** | 11/12 — worst `score/it` 0.0521 | **12/12 — worst `score/fr` 0.0257 · criterion 3 CLOSED** |
| `score/it` (ECE / accuracy) | 0.0521 / 0.8699 | **0.0014 / 0.9928** |
| holdout overall, checkpoint-alone | 0.8892 | **0.9828** (generalization gap −0.1105 → **−0.0165**) |
| holdout overall, routed pair | 0.8355 / 0.0747 | **0.9264 / 0.0318** |
| `choice` holdout | 0.9248 | **0.9808** |
| `noul` tripwire (worst) | 0.7807 | **0.9735** |
| languages, current split | 0.9799–0.9960 | **0.9928–1.0000** (`es` 1.0000) |
| in-template multi ECE | 0.0228 (disclosed trade) | **0.0036** — zero exceptions, both slices |
| English | byte-identical to B-15 | **byte-identical to B-16** |

The holdout cell table (checkpoint-alone, n = 415–420):

| Cell | ECE | | Cell | ECE |
| --- | ---: | --- | --- | ---: |
| `choice/de` | 0.0225 | | `score/de` | 0.0007 |
| `choice/es` | 0.0000 | | `score/es` | 0.0165 |
| `choice/fr` | 0.0055 | | `score/fr` | **0.0257** (worst) |
| `choice/it` | 0.0007 | | `score/it` | **0.0014** (was 0.1495 at B-15) |
| `choice/nl` | 0.0049 | | `score/nl` | 0.0025 |
| `choice/pt` | 0.0035 | | `score/pt` | 0.0056 |

**Disclosures (measured, never re-fixed).** The frozen yardstick held — holdout sha pins
`982236ca…` / `3119e73c…`, the gate rows never moved, and the expansion targeted a diagnosed
mechanism (the per-domain error table is in BACKLOG **B-17**). Off-domain probes: en-routed
rows unchanged (typed-decisions 0.306, XNLI 0.333); **MASSIVE 0.074 → 0.058 with `ECE raw`
0.046 → 0.197** — the in-domain gain re-sharpened off-domain confidence (the L-007/L-015
trade), still 3.4× its 0.017 chance. Accuracy improved everywhere the gates read; nothing was
smoothed.

## v0.9.0 — B-16: serve-key detector + the 2D confidence map (2026-10-09)

No retrain — the calibration **key** and the confidence **derivation** changed for the
multilingual `choice`/`score` answers. Gates pre-registered in `0f1593d`; full record in
BACKLOG **B-16**; the gate harness is `training.predict` (checkpoint-alone — the protocol
B-15's gates were read under, **L-020**), the ECE rows below are the routed pair.

| Axis | B-15 (2026-10-08) | B-16 (2026-10-09) | Note |
| --- | --- | --- | --- |
| Holdout per-language ECE cells, `choice`/`score` × 6 | **7 failing**, worst `score/it` 0.1495 | **1 failing**, `score/it` **0.0521** | B-1 criterion 3: 11/12 ≤ 0.05 |
| Routed holdout `choice` ECE | 0.087 | **0.044** | pair-as-served, `report.md` |
| Routed holdout overall ECE | 0.103 | **0.075** | accuracies byte-identical |
| Calibration-key detection diagonal | 65.8% (state-only) | **99.80% / 99.95%** | state + localized question (G2 ≥ 0.99) |
| multi in-template overall ECE (gate harness) | 0.0069 | **0.0228** | disclosed trade of the pooled-basis map; per-domain ECE still zero exceptions ≤ 0.05 |
| accuracy — every axis | — | **byte-identical** | `choice` 0.9248 · all six languages · `noul` tripwire (`it` 0.7807) · strict 1.0000 |

The holdout cell table (checkpoint-alone, n = 415–420 per cell):

| Cell | ECE | | Cell | ECE |
| --- | ---: | --- | --- | ---: |
| `choice/de` | 0.0118 | | `score/de` | 0.0211 |
| `choice/es` | 0.0467 | | `score/es` | 0.0369 |
| `choice/fr` | 0.0024 | | `score/fr` | 0.0095 |
| `choice/it` | 0.0320 | | **`score/it`** | **0.0521** |
| `choice/nl` | 0.0079 | | `score/nl` | 0.0296 |
| `choice/pt` | 0.0475 | | `score/pt` | 0.0141 |

**Disclosures (recorded, never re-fixed).** `score/it` **0.0521 > 0.05** stays NOT met —
tied to the holdout accuracy gap (0.8699) that Part 2 of the data cycle attacks; English is
untouched (its per-domain ECE values are byte-identical to the v0.9.0 disclosed P3 state,
`choice` 0.5243 — the fit-basis redesign remains queued); in-template multi ECE moved
0.0069 → 0.0228 under the pooled-basis map (every gate line still passes); probes re-ran the
same day with **unchanged accuracy** (MASSIVE T 1.75 → 1.85 under the new key). On the
routed pair the same cells read 9/12 — the gap scales monotonically with each cell's
state-only-undetected share (the English fallback carries the disclosed P3 confidence onto
foreign rows), which quantifies the queued **routing-quality** follow-up: 13% of
choice/score holdout rows.

## English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank) — B-13 artifact (v0.8.0, history)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **1.000** | 0.006 | 57.974 | 91.525 |
| `choice` | 500 | **1.000** | 0.017 | 85.739 | 94.256 |
| `noul` | 500 | **1.000** | 0.002 | 54.709 | 60.845 |
| `score` | 500 | **1.000** | 0.000 | 57.271 | 62.797 |

Noisy view (noise_rate 0.15): overall accuracy **0.997**, ECE **0.006**.

> **Accuracy is byte-identical to the pre-P3 run; ECE is not, and the reason is published
> rather than hidden.** Before P3 this column read 0.000 because the shipped temperature had
> been fitted *on this very split* (L-015 — an in-sample fit buys a perfect axis at home and
> zeroes it off-domain). The P3 confidence is fitted on a holdout the model never trained on,
> so home ECE is what an honest fit leaves: **0.006** overall, every primitive ≤ 0.017. The
> off-domain half of that trade is the point — see *Public probes* below and the JevBench
> calibration axis in `.specs/features/jevbench/spec.md` (*P3 results*).
> Latency re-measured 2026-10-02 on the same L4 reference box and path: **58.0 ms** p50
> against the B-5 artifact's 54.4 ms — the columns are box-bound, accuracy and ECE are not.

`noul` label audit of the same 500 rows: **0 contradictory** (241 request / 232 neutral / 27 empty,
0 unreadable), positive rate 0.482. The whole history of this table in seven rows — weights ×
labels, because neither half is comparable on its own:

| English checkpoint × labels | overall | `noul` | `choice` | `score` |
| --- | ---: | ---: | ---: | ---: |
| B-11 weights × pre-B-11 labels (what `v0.4.0` published) | 0.859 | 0.718 | 0.948 | 0.910 |
| B-11 weights × B-11 labels (ADR-0014) | 0.945 | 0.992 | 0.960 | 0.882 |
| B-11 weights × B-12 labels (previous published) | 0.972 | 0.992 | 0.946 | 0.978 |
| B-12 retrain × B-12 labels | 0.962 | 1.000 | 0.888 | 0.998 |
| control — identical recipe, seed 2 | 0.841 | 1.000 | 0.732 | 0.792 |
| B-5 five-domain bank × B-12 labels | 0.964 | 0.946 | 1.000 | 0.946 |
| **B-13 mixture retrain × B-12 labels — published** | **1.000** | **1.000** | **1.000** | **1.000** |

**What the B-13 row is (and what it is not).** The published English adapter is now the
mixture retrain: the same run-5 recipe and seed, trained on 35,540 records (the 21,000
five-domain records **plus** 11,000 pinned public records — MultiNLI/BoolQ/Banking77 — and
3,540 rule-tree family records built for the JevBench preparation, B-13), then the same
frozen-trunk `choice`-bank fit. It sweeps the support split **0.964 → 1.000** and the
five-domain split **0.964 → 1.000** with gate strict 1.000 (see below) — but this split is
**in-sample synthetic** (the known train/eval row collision), so saturation here buys
nothing external: on the 231 public JevBench items the same weights move Intelligence
**8.4 → 15.0** against a fresh control's **+0.6** (measurement and attribution in
`.specs/features/jevbench/tasks.md` JB-7). Read the public probes below as the honest
check on this table.

The B-5 trade note it replaced still explains why the five-domain artifact existed: on its
release it **lost** 0.008 on support (`noul` 0.992 → 0.946) to gain `choice` 0.946 → 1.000
and the whole five-domain split 0.511 → 0.964; B-13 inherits both halves at 1.000.

Two lessons from the history above still stand: the labels were the cap, not the model
(`noul` 0.718 → 0.946 by relabeling alone); and the retraining *gamble* is real — the
control differs from the retrain only in `out_dir` and lands 13 points lower.

## Multilingual (mmBERT-base + LoRA r=64 + choice head, 8 epochs) — B-5b fitted bank, five domains (v0.8.0, history)

The published multilingual artifact is the **B-5b cycle**: one joint mmBERT run on **30,000
five-domain multilingual records** (8 epochs, LoRA r=64) whose `choice` heads were then re-fitted
on the **frozen** trunk by `training/fit_choice_bank.py` (rank 128, 8 epochs, lr 1e-4 — the fit
ships its own recipe next to the weights as `choice_bank_fit.json`), promoted to `checkpoints/multi`
on **2026-10-02**. The **B-12 retrain** it replaces — 0.743 overall / 0.468 `choice` on this same
support split — is superseded: it was a *routed* number that no longer even reproduces (rerouted on
the same rows it reads 0.736, BACKLOG **B-5b** anchor correction), and its weights × labels ladder
lives in `.specs/project/BACKLOG.md` **B-12**. Both halves below are published because they answer
different questions on different harnesses.

**Support-only split — routed runtime harness** (`data/eval_multi.jsonl`, 1,500 rows): the
historical section's eval set, re-measured on the new weights, routed at runtime by
`training.evaluate --backend encoder` — **13.4% of rows fall through to `tachyone-en`** on this
split (14.8% on the five-domain rows). Rule **L-013**: never gate one checkpoint's quality on a
routed harness — read this table as the product-level number, not as the adapter's.

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.895** | 0.062 | 46.944 | 79.208 |
| `choice` | 500 | 0.964 | 0.032 | 73.265 | 87.651 |
| `noul` | 500 | 0.878 | 0.099 | 43.879 | 55.094 |
| `score` | 500 | 0.844 | 0.056 | 46.337 | 57.866 |
| lang `de` | 249 | 0.900 | 0.083 | 47.055 | 78.406 |
| lang `es` | 252 | 0.940 | 0.048 | 46.587 | 77.165 |
| lang `fr` | 249 | 0.940 | 0.029 | 46.490 | 79.621 |
| lang `it` | 249 | 0.932 | 0.056 | 46.824 | 85.826 |
| lang `nl` | 249 | 0.715 | 0.167 | 50.776 | 76.504 |
| lang `pt` | 252 | 0.944 | 0.034 | 46.620 | 81.543 |

Worst language: `nl` — accuracy 0.715, ECE **0.167** (open, backlog **B-1**). Noisy view (noise_rate
0.15): overall accuracy **0.889**, ECE **0.067** (p50 47.363 ms). `noul` label audit of the same
rows: **0 contradictory** in all six languages (request/neutral/empty = 33–46 / 33–46 / 4–5 per
language); `noul` reads pt 0.952, es 0.940, fr 0.940, it 0.916, de 0.807, nl 0.711.

**Five-domain split — explicit adapter harness** (`data/eval_multi_domains.jsonl`, 7,500 rows).
This is the publish gate: the adapter it replaces scored **0.5609** overall on exactly these rows
through this harness — zero-shot new domains 0.438 (`agent_tools`) to 0.565 (`ecommerce`),
`support` 0.8413 — and every gate was fixed from that baseline *before* training (BACKLOG
**B5B-4**). Its `support` half is the same records as `eval_multi.jsonl`, so the `support` row
below is what those same 1,500 rows score once routing stops mixing checkpoints.

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 7500 | **0.997** | 0.001 | 18.559 | 57.545 |
| `choice` | 2500 | 0.992 | 0.004 | 50.935 | 59.439 |
| `noul` | 2500 | 1.000 | 0.000 | 16.882 | 17.648 |
| `score` | 2500 | 1.000 | 0.000 | 18.537 | 20.425 |
| domain `support` | 1500 | 1.000 | 0.000 | 18.527 | 51.912 |
| domain `agent_tools` | 1500 | 0.997 | 0.004 | 18.581 | 52.449 |
| domain `documents` | 1500 | 0.998 | 0.002 | 18.662 | 53.288 |
| domain `ecommerce` | 1500 | 0.993 | 0.004 | 18.592 | 59.203 |
| domain `voice` | 1500 | 0.999 | 0.001 | 18.457 | 52.240 |
| lang `de` | 1245 | 0.998 | 0.001 | 18.575 | 57.748 |
| lang `es` | 1260 | 0.998 | 0.004 | 18.542 | 57.191 |
| lang `fr` | 1245 | 1.000 | 0.004 | 18.585 | 56.926 |
| lang `it` | 1245 | 0.998 | 0.001 | 18.556 | 58.509 |
| lang `nl` | 1245 | 0.995 | 0.003 | 18.552 | 57.537 |
| lang `pt` | 1260 | 0.995 | 0.001 | 18.554 | 57.020 |

**Gates — fixed before training, every one passes:** `support` **1.0000 ≥ 0.8413** ✓ · worst new
domain **0.9927 ≥ 0.70** ✓ · per-domain ECE **0.0005–0.0038, all five under 0.05 with zero
declared exceptions — the first time either checkpoint family posts that** ✓ · gate strict
**1.000** — 0 fell to shared, 0 wrong domain over 2,500 `choice` rows (every domain 1.000 on its
500) ✓. Per language **0.995–1.000 at ECE 0.001–0.004 — six of six ≤ 0.05 on this split**
(NFR-C06). Worst (domain, language) cell: `ecommerce/pt` — accuracy 0.980, ECE 0.012. Input text
seen in training: n=4504, accuracy 0.999 — never seen: n=2996, accuracy **0.996**, so the headline
is not the known train/eval row collision buying it. `noul` label audit: **0 contradictory rows**
in all six languages (`noul` = 1.000 everywhere; request/neutral/empty = 178–211 / 184–214 / 20–25
per language).

> **Latency in both tables was measured on the L4 reference box (2026-10-02).** The support
> split's 46.9 ms p50 is *not* comparable with the 16.0 ms the pre-B-5b table showed on the RTX
> 3060 build; accuracy and ECE are machine-independent and are what the comparisons rest on.

## Fast path (CUDA graphs) — v0.9.0 artifacts, re-measured 2026-10-08

mmBERT + `checkpoints/multi` (the B-15 arm as published), 189 held-out states, batch=1;
`TACHYONE_FAST=1` uses per-shape CUDA graphs with bf16-resident weights. Capture is warmed
before timing. Re-measured 2026-10-08 on the L4 reference box (`benchmarks/results/fast_path.json`;
the v0.8.0 artifacts are archived as `*_preb15.json`) — latency is box-bound; the speedup ratio
and the parity are not.

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 13.907 | 15.877 | 70.6 | 159.3 | 74.4 |
| fast (CUDA graphs, bf16) | 3.824 | 3.940 | 250.3 | 554.4 | 202.7 |

**Parity:** max absolute answer-probability difference **0.001120** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **3.64× p50 speedup** for the fast path (3.76× on the same box with the v0.8.0 weights,
2.65× on the 3060 build).

**The English artifact** (`benchmarks/results/fast_path_en.json`, ModernBERT + the fitted bank
`checkpoints/en`): **2.533×** p50 (17.31 → 6.83 ms) — and, disclosed exactly as measured, **1
top-label flip** among the 16 sampled questions, max answer-probability difference **0.024133**
(the v0.8.0 run on the same states read 0 flips / 0.000784). Parity is a property of the
*weights* as much as of the path: with the new arm, one question's fp32/bf16 answers diverged
far enough to cross the argmax where the old arm's did not. The gate and the per-domain
heads still run *after* the encode, so the CUDA-graph path neither sees nor perturbs them — which
is what ADR-0016 predicted when it kept the head as plain JSON computed post-encode; NFR-P01 is
met by both paths (17.31 → 6.83 ms).

**The multilingual five-domain run** (`benchmarks/results/fast_path_multi.json`, mmBERT + the
fitted bank, `eval_multi_domains`, max_len 1024, n=284): **3.708×** p50 (14.23 → 3.84 ms),
**0 top-label flips** (16 questions), max answer-probability difference **0.001120** — the gate
and the keyed heads still execute *after* the encode, exactly as ADR-0016 predicted for the
keyed format. The earlier runs stay archived: v0.8.0 support-split 3.762× / 0 flips, English
2.504× / 0 flips, B-5b five-domain 3.659× / 0 flips (all `*_preb15.json`), plus the older
B-5/B-13 English runs (`fast_path_en_b5.json`, `fast_path_en_prep3.json`).

## Public probes (B-7) — v0.9.0 artifacts, re-measured 2026-10-08

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. (B-13's mixture trains on MultiNLI's **train** split; XNLI's test pairs are
translated MultiNLI *development* rows — standard split discipline, no probe row was ever
trained or calibrated on, and each probe stays evaluation-only.) Full tables (including
per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | **0.306** | 0.20–0.50 | 0.096 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | **0.058** | 0.017 | 0.197 |
| XNLI (`en`) | CC BY-NC 4.0 | 5010 | **0.333** | 0.333 | 0.068 |

Evaluation only — no probe row has ever reached `training/`. Read each number against its
**chance** level and against the in-sample 1.000 above, not against the other rows: these are the
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
trade the in-domain table buys, published here instead of absorbed.

**Re-run on the B-5b artifact (2026-10-02, right after `munod/tachyone-multi` `9a3ef5a5`).**
Only MASSIVE can move — typed-decisions and XNLI are English rows that route to the English
adapter, untouched since B-5, so their rows above are the same measurements. MASSIVE goes
**0.011 → 0.039** against its 0.017 chance, with **every language above chance** (0.025–0.055;
the B-5a run spanned 0.004–0.039), while the confidence trade deepens: `Conf` 0.365 → **0.557**,
`ECE raw` 0.354 → **0.518**, `Brier` 1.272 → 1.393. The fit no longer needs the grid ceiling
(**T=14.55**, the only probe off it): `ECE cal` 0.0165 → 0.005. Read beside the internal table —
in-domain `choice` **0.992**, external 60-way intent routing still near chance — the honest
reading is that the public probe measures option-space coverage the synthetic training never
contains; the adapter is no longer *below* chance anywhere, and both facts are published side by
side.

**Re-run on the B-13 artifact (2026-10-02, right after `munod/tachyone-en` got the mixture
retrain).** All three move together for the first time since B-12. **XNLI 0.341 → 0.566**
(chance 0.333) — the MultiNLI-trained reading transfers, and XNLI's test pairs are translated
MultiNLI *development* rows the trainer never saw. **typed-decisions 0.269 → 0.367** (still
inside its 0.20–0.50 per-config band, up 9.8 points overall). **MASSIVE 0.039 → 0.051** against
the 0.017 chance with every language above it (`en` 0.117, `it` 0.055, `de` 0.047, `es`/`fr`
0.041, `nl` 0.031, `pt` 0.027 — the en-locale rows ride the new English adapter). Confidence
stays sharp off-domain: `Conf` **0.945 / 0.992 / 0.634**, `ECE raw` **0.578 / 0.426 / 0.583**;
`ECE cal` improves for typed-decisions (0.170 → 0.139) and XNLI (0.294 → 0.144) and worsens for
MASSIVE (0.005 → 0.038, T=11.6). The direction is finally right on accuracy while the
calibration trade remains exactly where L-007 and L-015 say it is: the data bought
*capability off-domain*, not *confidence off-domain*.

**Re-run on the P3 confidence (2026-10-02, same adapter weights).** Only the confidence moved —
every accuracy above is byte-identical — and it moved in the direction L-015 asked for: `ECE raw`
**0.578 → 0.416**, **0.583 → 0.296**, **0.426 → 0.264** (typed-decisions / MASSIVE / XNLI), with
mean confidence pulled back from **0.945 / 0.634 / 0.992** to **0.782 / 0.348 / 0.718** and
`Brier` 1.189 / 1.470 / 0.857 → **0.968 / 1.194 / 0.709**. The harness's own per-probe fits relax
with it (MASSIVE comes off the ceiling, T 11.6 → **7.55**; XNLI 20.0 → **3.10**) and `ECE cal`
reads **0.029 / 0.050 / 0.121**. Read against the internal table, the honest summary is unchanged:
the adapter is still *less* confident off-domain than on its own turf — that is what the evidence
map is for — and it is no longer confident about being wrong.

**Re-run on the v0.9.0 artifact (2026-10-08, right after the B-15 retrain).** The published pair
was pinned locally through `TACHYONE_ADAPTERS`, byte-identical to Hub `tachyone-en` `e47d6393` /
`tachyone-multi` `c5c05fd2`. The B-15 arm trains on the recomposed **synthetic-only** recipe,
while the B-13 artifact it replaced carried the real-source layer (MultiNLI/BoolQ/Banking77,
`training/configs/jev_mixture.json`) — the probes read that trade directly: **XNLI
0.566 → 0.333**, exactly back to its chance level, the MultiNLI transfer went with the mixture
layer the retrain did not carry (recorded as a follow-up in BACKLOG **B-15**; published here,
never re-fixed); **typed-decisions 0.367 → 0.306**, the per-config mix moving rather than
collapsing (0.274/0.450/0.290/0.456 → 0.256/0.260/0.446/0.262); **MASSIVE 0.051 → 0.074**
against its 0.017 chance — the best reading on that probe so far, every language above chance
(0.057–0.104, `de` 0.104 leading; the `en`-locale row alone moved down 0.117 → 0.080). What
improved across all three is confidence: `ECE raw` **0.416 / 0.296 / 0.264 → 0.096 / 0.042 /
0.068** with mean confidence pulled from **0.782 / 0.348 / 0.718 → 0.398 / 0.107 / 0.401**, so
the shipped stack is no longer sharply wrong off-domain. The harness's per-probe fits now read
T=20 (typed-decisions, ceiling), **T=1.75** (MASSIVE) and T=20 (XNLI, ceiling) — `ECE cal`
above is what those fits buy, read it next to `Conf` and `Brier` exactly as the caveat at the
top of this page says.

**Re-run under B-16 (2026-10-09, the v3 calibration key).** Accuracy is unchanged on all
three (**0.306 / 0.074 / 0.333**). Only MASSIVE's multilingual rows moved with their
temperature key: T **1.75 → 1.85**, `ECE raw` **0.042 → 0.046**, `Conf` 0.107 → 0.110 —
noise-level, in the same direction the B-16 key corrected (the probes read the selected mass
after the shipped temperature; the fitted confidence multilingual `choice`/`score` answers
now report is measured in [`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md)).

**Re-run under B-17 (2026-10-10, the retrained multilingual arm).** English-routed rows are
byte-unchanged (**typed-decisions 0.306, XNLI 0.333** — en ran no training). The multilingual
rows moved with the new weights: **MASSIVE 0.074 → 0.058** (still 3.4× its 0.017 chance)
while `ECE raw` goes **0.046 → 0.197** — the in-domain gains of the teacher-expanded banks
re-sharpened off-domain confidence, the L-007/L-015 trade this page has published in the
same direction every time capability was bought in-domain. Read it beside the internal
tables: the adapter got *sharper and better* at home and *sharper* away from it.

## Multi-domain experiment (B-5a / ADR-0016) — **released 2026-10-01**

Five domains trained into one adapter — `support`, `ecommerce`, `agent_tools`, `documents`,
`voice` — English only, 21,000 records, evaluated on `data/eval_en_domains.jsonl` (7,500 rows;
its `support` half is the same records as `eval_en.jsonl`, so **1.000** above is the comparable
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

### Released 2026-10-02: B-13 mixture retrain (JevBench preparation)

Same recipe and seed as the table above, one factor changed — the **training data**: the 21,000
five-domain records plus 11,000 pinned public records (MultiNLI/BoolQ/Banking77, licences in
`training/data/jev_sources.lock.json`) plus 3,540 rule-tree family records
(`training/data/jev/`), one 35,540-record mixture, then the identical frozen-trunk bank fit.
A fresh **control arm** (same recipe, old data) was trained beside it on the other GPU so the
delta could be attributed (L-006/L-012).

| Scope (n=7,500) | B-5 bank | **B-13 mixture — published** |
| --- | ---: | ---: |
| overall | 0.964 | **1.000** |
| `support` | 0.964 | **1.000** |
| `agent_tools` | 0.964 | **1.000** |
| `documents` | 0.963 | 0.999 |
| `ecommerce` | 0.964 | **1.000** |
| `voice` | 0.963 | **1.000** |
| `choice` (gate strict) | 0.9996 (1.000) | **1.000 (1.000)** |
| unseen text (n=791) | — | **0.9987** (ECE 0.010) |

**Gates: support 1.000 ≥ 0.85 ✓ · worst new domain 0.999 ≥ 0.70 ✓ · per-domain ECE
0.006–0.013 (all five ≤ 0.05) ✓ · gate strict 1.000, 0 fell to shared, 0 wrong domain ✓.**

**What the retrain bought outside this split (the number that matters).** On the 231 public
JevBench items — evaluation-only, never trained on — the published artifact scores
Intelligence **8.4 → 15.0** while the fresh control moves **+0.6**: the gain is the data's,
not run variance. Per tier: easy 0.500 → **0.625**, standard 0.361 → **0.417**, hard 0.288 →
**0.333** (still at its 0.336 chance — cc-score 0, unchanged).
The structural limit is unchanged and visible: `noul` still scores `sigmoid(cos(question,
state))` without reading the rubric, so policy/trap judgments stay at chance — that is the
open item, not the data. Full attribution table: `.specs/features/jevbench/tasks.md` JB-7;
public-probe deltas in the section above.

## Known limitations

- Per-language ECE (NFR-C06 / backlog **B-1**): on the v0.9.0 five-domain split five of six
  multilingual languages sit at ≤ 0.05 (`it` 0.0523 is the exception); on the **holdout** — the
  honest yardstick — the B-1 acceptance (per-language `choice`/`score` ≤ 0.05) is **NOT met**:
  `choice` `es` 0.0753 / `it` 0.0968 / `pt` 0.1014, `score` `de` 0.0540 / `es` 0.0951 /
  `it` 0.1495 / `nl` 0.0577 (English passes at 0.0155 / 0.0067). Recorded with the numbers,
  never re-fixed — `.specs/project/BACKLOG.md` **B-15**.
- **The English confidence fit is pooled by design (L-015) and its in-domain cost is real:**
  the pooled fit pegged `choice` at the grid's top (**T = 10.0**) to calibrate held-out rows,
  and the price is an in-domain `choice` ECE of **0.5243** (accuracy untouched). P3's
  in-domain ≤ 0.05 acceptance is recorded **NOT met** for the rebuilt asset; the follow-up is
  a fit-basis redesign, not another fit.
- **The routed numbers include the router.** ~13–15% of five-domain multilingual rows fall
  through to the English checkpoint by design (empty state or no Latin-language signal,
  **L-013**): the tables show the product as served, while the multilingual checkpoint itself
  answers **0.9885** when it answers everything.
- **`choice` was the weak primitive everywhere multilingual** (0.468 on the superseded B-12
  weights — the ladder lives in BACKLOG **B-12**): the corrected labels made `noul`/`score` easy
  and the shared trunk spent itself on them. B-5b fixed it the way the English side did — by
  **re-fitting the heads on a frozen trunk** (English `choice` 0.745 → 0.9996) — and multilingual
  `choice` now reads **0.992** on the five-domain split (0.258 for the adapter it replaces) and
  0.964 on the support routed split, with `noul`/`score` at 1.000/1.000 on the five-domain split.
  The English fix cost `noul`/`score` on support (0.992/0.978 → 0.946/0.946), which is why that
  published row shows both numbers instead of the best one.
- Calibrated ECE reflects the assets the runtime ships: the multilingual temperature is fitted
  on **pooled in-template + holdout** predictions (never in-sample), the English one on the
  never-trained P3 holdout. The public probes above are the opposite case — public
  distributions, evaluation-only, no shared states with training (re-measured on the v0.9.0
  artifacts, 2026-10-08).
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.288** while scoring **1.000** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
