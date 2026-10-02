# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

For the **head-to-head** against the open System One scorer and two local LLMs (accuracy, ECE,
Brier, latency, throughput, memory and contract compliance on two evaluation sets), see
[`compare.md`](compare.md).

**Setup:** single RTX 3060 12GB (the **B-5b** multilingual training and head fit ran on a single
NVIDIA L4 23GB) · 9,000 English / 21,000 English five-domain / 30,000 multilingual five-domain
deterministic synthetic train records (multilingual keeps support's 18,000 and adds 3,000 for each
of the four new domains; fully localized per language, per-record RNG) · held-out evals of
1,500 / 1,500 support rows (en / multi) and 7,500 / 7,500 five-domain rows (en / multi) · LoRA
(r=16 English, **r=64 multilingual**) plus a low-rank `choice` head (r=32; r=128 for both
five-domain runs) · 4 epochs English, **8 multilingual**, 6 English five-domain · the published
multilingual artifact = that 8-epoch trunk + `choice` heads re-fitted on the **frozen** trunk
(rank 128, 8 epochs, lr 1e-4; recipe in `choice_bank_fit.json`) · bf16 + gradient checkpointing.
Calibrated ECE is after per-`(primitive, language)`
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
English adapter publishes the **B-5 bank over run 5's trunk** (a trunk trained before B-11/B-12 —
the retrain-on-corrected-labels option was tried and lost, BACKLOG B-12) and the multilingual
adapter now publishes the **B-5b fitted bank** — the 30k five-domain joint run plus `choice`
heads re-fitted on its frozen trunk — superseding the B-12 retrain (its weights × labels ladder
lives in `.specs/project/BACKLOG.md` **B-12**). Provenance is stated with every table.

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

## Multilingual (mmBERT-base + LoRA r=64 + choice head, 8 epochs) — B-5b fitted bank, five domains

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

## Fast path (CUDA graphs)

mmBERT + `checkpoints/multi` (the adapter as it stood on 2026-10-01 — the B-12 retrain, replaced
by the B-5b bank on 2026-10-02), 189 held-out states, batch=1; `TACHYONE_FAST=1` uses per-shape CUDA
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

**The B-5b multilingual artifact** was measured on its own five-domain split
(`benchmarks/results/fast_path_multi.json`, mmBERT + the fitted bank, `eval_multi_domains`,
max_len 1024, n=284): **3.659×** p50 (13.97 → 3.82 ms), **0 top-label flips** (16 questions), max
answer-probability difference **0.000173** — the gate and the keyed heads still execute *after*
the encode, so the graphed path sees neither, exactly as ADR-0016 predicted for the keyed format.
The earlier runs stay where they were: the support-split run in the table above (`fast_path.json`,
the weights B-5b replaced) at **3.762× / 0 flips**, and the English run (`fast_path_en.json`) at
**2.59× / 0 flips**.

## Public probes (B-7)

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. Full tables (including per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | **0.269** | 0.20–0.50 | 0.542 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | **0.039** | 0.017 | 0.518 |
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

- Per-language ECE (NFR-C06): on the five-domain split **all six** multilingual languages meet the
  `0.05` target for the first time (ECE 0.001–0.004); on the support-only **routed** split three
  are still above it — `de` 0.083, `it` 0.056 and worst `nl` **0.167** (open, backlog **B-1**).
  That split's aggregate ECE is declared alongside (`choice` 0.032, `noul` 0.099, `score` 0.056,
  overall 0.062).
- **`choice` was the weak primitive everywhere multilingual** (0.468 on the superseded B-12
  weights — the ladder lives in BACKLOG **B-12**): the corrected labels made `noul`/`score` easy
  and the shared trunk spent itself on them. B-5b fixed it the way the English side did — by
  **re-fitting the heads on a frozen trunk** (English `choice` 0.745 → 0.9996) — and multilingual
  `choice` now reads **0.992** on the five-domain split (0.258 for the adapter it replaces) and
  0.964 on the support routed split, with `noul`/`score` at 1.000/1.000 on the five-domain split.
  The English fix cost `noul`/`score` on support (0.992/0.978 → 0.946/0.946), which is why that
  published row shows both numbers instead of the best one.
- Calibrated ECE is measured in-sample on the held-out synthetic split. The public probes above
  are the opposite case — public distributions, evaluation-only, no shared states with training.
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.236** while scoring **0.972** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
