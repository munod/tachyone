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

## English (ModernBERT-large + LoRA r=16 + choice head) — B-11 weights

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.972** | 0.020 | 23.094 | 36.793 |
| `choice` | 500 | 0.946 | 0.038 | 36.032 | 37.146 |
| `noul` | 500 | **0.992** | 0.025 | 14.678 | 19.332 |
| `score` | 500 | **0.978** | 0.016 | 22.841 | 26.442 |

Noisy view (noise_rate 0.15): overall accuracy **0.969**, ECE **0.022**.

`noul` label audit of the same 500 rows: **0 contradictory** (241 request / 232 neutral / 27 empty,
0 unreadable), positive rate 0.482. The whole history of this table in five rows — weights ×
labels, because neither half is comparable on its own:

| English checkpoint × labels | overall | `noul` | `choice` | `score` |
| --- | ---: | ---: | ---: | ---: |
| B-11 weights × pre-B-11 labels (what `v0.4.0` published) | 0.859 | 0.718 | 0.948 | 0.910 |
| B-11 weights × B-11 labels (ADR-0014) | 0.945 | 0.992 | 0.960 | 0.882 |
| **B-11 weights × B-12 labels — published** | **0.972** | 0.992 | **0.946** | 0.978 |
| B-12 retrain × B-12 labels | 0.962 | 1.000 | 0.888 | 0.998 |
| control — identical recipe, seed 2 | 0.841 | 1.000 | 0.732 | 0.792 |

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
graphs with bf16-resident weights. Capture is warmed before timing. Re-measured 2026-09-29 on the
B-12 adapters (`benchmarks/results/fast_path.json`).

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 8.972 | 10.082 | 98.8 | 148.4 | 60.3 |
| fast (CUDA graphs, bf16) | 3.385 | 4.466 | 261.7 | 353.8 | 141.8 |

**Parity:** max absolute answer-probability difference **0.000559** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **2.65× p50 speedup** for the fast path.

## Public probes (B-7)

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. Full tables (including per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | 0.330 | 0.20–0.50 | 0.543 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | 0.011 | 0.017 | 0.192 |
| XNLI (`en`) | CC BY-NC 4.0 | 5010 | 0.333 | 0.333 | 0.268 |

Evaluation only — no probe row has ever reached `training/`. Read each number against its
**chance** level and against the in-sample 0.972 above, not against the other rows: these are the
released adapters zero-shot on tasks they were never trained for. **Re-run on the B-12 adapters
(2026-09-29)**: typed-decisions 0.330 (best config 0.412, worst 0.250), XNLI exactly at chance
0.333, and MASSIVE **0.011 against its 0.017 chance** (per language 0.006–0.018 over 60 intents).
MASSIVE is a 60-way `choice` task and the multilingual `choice` head is the weakest cell of the
in-domain table too (0.468) — the external probe and the internal one tell the same story.
`ECE raw` is not comparable across these rows either: it moves with *confidence*, and the adapters
are now very unsure off-domain (`Conf` 0.021 MASSIVE / 0.354 XNLI / 0.399 typed-decisions;
`Brier` 0.983 / 0.670 / 0.699). The typed-decisions and XNLI fits still hit the grid ceiling
(T=20.0). This table is the measurement that `B-5` set out to improve.

## Multi-domain experiment (B-5a) — **both gates pass on the B-12 labels** (2026-09-29)

Five domains trained into one adapter — `support`, `ecommerce`, `agent_tools`, `documents`,
`voice` — English only, 21,000 records, evaluated on `data/eval_en_domains.jsonl` (7,500 rows;
its `support` half is the same records as `eval_en.jsonl`, so **0.972** above is the comparable
support number). **Nothing was released:** the adapters on the Hub are the support-only ones above.
The designated artifact is **run 5** (`choice_rank` 128, 6 epochs), measured here on the B-12
labels:

| Domain | released adapter (zero-shot) | run 5 (published) | Δ |
| --- | ---: | ---: | ---: |
| `support` | 0.972 | **0.886** | −0.086 |
| `agent_tools` | 0.328 | **0.935** | +0.607 |
| `documents` | 0.359 | **0.915** | +0.556 |
| `ecommerce` | 0.358 | **0.842** | +0.484 |
| `voice` | 0.539 | **0.815** | +0.276 |
| **overall** | 0.511 | **0.879** | +0.368 |

| Primitive | released | run 5 | ECE (run 5) |
| --- | ---: | ---: | ---: |
| `choice` | 0.452 | 0.745 | 0.055 ✅ |
| `noul` | 0.697 | 0.946 | 0.042 ✅ |
| `score` | 0.385 | 0.946 | 0.029 ✅ |

**Gates:** worst new domain ≥ 0.70 → **pass** (`voice` 0.815; `agent_tools` 0.935, `documents`
0.915, `ecommerce` 0.842). Support ≥ 0.85 → **pass** (**0.886**, better than the 0.861 it scored
on the B-11 labels). Per-domain ECE is 0.036–0.080: `agent_tools` 0.036 and `documents` 0.039 meet
the 0.05 target, `support` 0.058, `voice` 0.066 and `ecommerce` 0.080 stay declared exceptions —
one scalar temperature per primitive fitted across five domains at once.

**Why the artifact keeps its B-11 weights.** The B-12 cycle retrained run 5's recipe on the
corrected labels and it *failed* the support gate (0.763): `noul` and `score` both hit **1.000**
while `choice` collapsed to **0.303** (train loss 0.027, eval ~chance = the head memorised the
training rows). The same trade showed up milder in English (control 0.841) and not at all in the
multilingual run — it is the shared-trunk interference of B-5 seen from the other side, now that
the two tone tasks are trivially learnable. Retraining on the corrected data is therefore **no
longer** the cheap untried option; it was tried, measured, and lost to the B-11 weights by 0.111.

What the six original runs established still stands: the binding constraint on `choice` is neither
capacity nor time — `choice_rank` 32 → 128 → 256 with 4 → 6 → 8 epochs moved support 0.655 →
0.785 → 0.795 on the pre-B-11 labels while the fourth-option leak simply **moved between runs**.
The full six-run curve, the per-(domain, primitive) cross tables and the three structural options
live in `.specs/project/BACKLOG.md` **B-5**.

## Known limitations

- Per-language ECE meets the `0.05` target for **one of six** multilingual languages (`es` 0.045);
  `pt` 0.062, `fr` 0.101, `de` 0.118, `nl` 0.192 and `it` 0.197 are above it (NFR-C06). Multilingual
  `choice` ECE 0.099, `noul` ECE 0.108 — all declared.
- **`choice` is the weak primitive everywhere multilingual** (0.468): the corrected labels made
  `noul`/`score` easy and the shared trunk spends itself on them — the published English and
  five-domain adapters are the *B-11* weights precisely because retraining on the corrected labels
  cost `choice` (0.946 → 0.888, and 0.745 → 0.303 for five domains). The structural options are in
  `BACKLOG.md` **B-5**; the multilingual `choice` head is backlog `B-1`.
- Calibrated ECE is measured in-sample on the held-out synthetic split. The public probes above
  are the opposite case — public distributions, evaluation-only, no shared states with training.
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.236** while scoring **0.972** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
