# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

For the **head-to-head** against the open System One scorer and two local LLMs (accuracy, ECE,
Brier, latency, throughput, memory and contract compliance on two evaluation sets), see
[`compare.md`](compare.md).

**Setup:** single RTX 3060 12GB · 9,000 English / 18,000 multilingual deterministic synthetic
train records (fully localized per language, per-record RNG) · 1,500 held-out eval · LoRA (r=16
English, **r=64 multilingual**) plus a low-rank `choice` head (r=32) · 4 epochs · bf16 + gradient
checkpointing. Calibrated ECE is after per-`(primitive, language)` temperature fitting on the
held-out split (in-sample). The **noisy view** applies one surface edit (typo/accents/casing) to
15% of states (B-4).

**Labels (B-11 / [ADR-0014](adr/ADR-0014-noul-label-from-text.md)):** every `noul` target is now
derived from the text it accompanies — `request` → 1, `neutral`/empty → 0. The pre-B-11 rule took
the label from the loop index, which left **every** `noul` label in `es`/`de`/`nl` at 0 and
contradicted 121 of the 241 request-toned rows in `eval_en` (50%). All datasets were regenerated
— only `noul.target` bytes moved — and both adapters were retrained, so the tables below are one
consistent set; the pre-B-11 numbers are kept in `.specs/project/BACKLOG.md` **B-11** and are **not**
comparable line-by-line with these.

## English (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.945** | 0.034 | 23.424 | 37.989 |
| `choice` | 500 | 0.960 | 0.024 | 36.986 | 38.353 |
| `noul` | 500 | **0.992** | 0.025 | 14.838 | 23.309 |
| `score` | 500 | 0.882 | 0.067 | 23.201 | 25.212 |

Noisy view (noise_rate 0.15): overall accuracy **0.942**, ECE **0.035**.

`noul` label audit of the same 500 rows: **0 contradictory** (241 request / 232 neutral / 27 empty,
0 unreadable), positive rate 0.482. The three numbers that separate measurement from capability:

| English `noul` | accuracy |
| --- | ---: |
| pre-B-11 weights, pre-B-11 labels (what was published) | 0.718 |
| pre-B-11 weights, corrected labels | 0.946 |
| B-11 weights, corrected labels (published) | **0.992** |

The old weights were already reading the text — the label was what capped them. `score` is the one
primitive that moved down (0.910 → 0.882, ECE 0.039 → 0.067): its labels did not change, so that
is training variance on a different `noul` task sharing the trunk (L-006 territory), not a label
effect.

## Multilingual (mmBERT-base + LoRA r=64 + choice head, 8 epochs)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | **0.718** | 0.042 | 15.866 | 49.029 |
| `choice` | 500 | 0.468 | 0.022 | 20.246 | 55.600 |
| `noul` | 500 | 0.832 | 0.120 | 12.511 | 20.748 |
| `score` | 500 | 0.854 | 0.059 | 13.723 | 28.951 |
| lang `es` | 252 | 0.810 | 0.040 | 13.996 | 48.167 |
| lang `pt` | 252 | 0.774 | 0.025 | 14.019 | 54.784 |
| lang `it` | 249 | 0.751 | 0.044 | 16.025 | 47.652 |
| lang `fr` | 249 | 0.699 | 0.050 | 15.259 | 47.077 |
| lang `de` | 249 | 0.687 | 0.101 | 16.443 | 55.795 |
| lang `nl` | 249 | 0.586 | 0.110 | 19.746 | 46.429 |

Noisy view (noise_rate 0.15): overall accuracy **0.713**, ECE **0.047**.

**Why this table is not comparable with the 0.853 it replaces.** The pre-B-11 labels left *every*
`noul` row in `es`/`de`/`nl` at 0 (L-010), so 0.960 `noul` and 0.853 overall were partly a
language→label shortcut. Measured against the corrected labels, the **pre-B-11 weights score
0.714** (`noul` 0.614) — that is the honest before side, and the retrain above is the after:

| multilingual sweep (B-11) | overall | ECE | `noul` | `choice` | `score` |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre-B-11 weights on corrected labels | 0.714 | 0.076 | 0.614 | **0.674** | 0.854 |
| retrain, 4 epochs | 0.657 | 0.051 | 0.832 | 0.490 | 0.648 |
| control (identical config) | 0.663 | 0.059 | 0.832 | 0.496 | 0.660 |
| **8 epochs — published** | **0.718** | **0.042** | **0.832** | 0.468 | **0.854** |
| `choice_rank` 128, 8 epochs | 0.671 | 0.044 | 0.832 | 0.490 | 0.690 |

The control (Δ0.006 against the first retrain) shows the 4-epoch drop is systematic, not L-006
variance; 8 epochs recover `score` and take overall above the old weights with the best ECE of
the table; and `choice_rank` 32 → 128 collapsed the *training* loss (0.408 → 0.085) while eval
stayed at 0.490 — pure overfit, the same pattern as B-5a run 6. The residual `choice` cost
(0.468 against 0.674, with `choice` labels never having moved) is the balanced `noul` task
sharing the trunk; it is published rather than hidden.

**Per language:** `noul` accuracy now reads de 0.747, es 0.941, fr 0.855, it 0.843, nl 0.663,
pt 0.941 against **0 contradictory labels** in all six (the audit's request/neutral/empty split is
33–46 / 33–46 / 4–5 per language). Three of six languages meet **ECE ≤ 0.05** (`pt` 0.025,
`es` 0.040, `it` 0.044); `fr` 0.0502 sits on the line and `de` 0.101 / `nl` 0.110 remain open
(NFR-C06), as does `noul`'s own ECE of 0.120 — a balanced, text-derived label is a harder thing
to calibrate than a constant one.

## Fast path (CUDA graphs)

mmBERT + `checkpoints/multi`, 189 held-out states, batch=1; `TACHYONE_FAST=1` uses per-shape CUDA
graphs with bf16-resident weights. Capture is warmed before timing. Re-measured 2026-09-28 on the
B-11 adapters (`benchmarks/results/fast_path.json`).

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 9.186 | 10.205 | 101.5 | 149.8 | 60.5 |
| fast (CUDA graphs, bf16) | 3.373 | 3.840 | 261.0 | 355.0 | 141.7 |

**Parity:** max absolute answer-probability difference **0.000771** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **2.72× p50 speedup** for the fast path.

## Public probes (B-7)

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. Full tables (including per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | 0.330 | 0.20–0.50 | 0.498 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | 0.013 | 0.017 | 0.136 |
| XNLI (`en`) | CC BY-NC 4.0 | 5010 | 0.333 | 0.333 | 0.269 |

Evaluation only — no probe row has ever reached `training/`. Read each number against its
**chance** level and against the in-sample 0.945 above, not against the other rows: these are the
released adapters zero-shot on tasks they were never trained for. **Re-run on the B-11 adapters
(2026-09-28)**, which moves them in different directions and is worth reading carefully:
typed-decisions 0.323 → 0.330 (best config 0.412, worst 0.250), XNLI flat at exactly chance
(0.334 → 0.333), and MASSIVE **0.033 → 0.013, i.e. below its 0.017 chance** (per language
0.006–0.023 over 60 intents). MASSIVE is a 60-way `choice` task, so its drop tracks the
multilingual `choice` regression the sweep above records — the external probe sees the same
trade-off the in-domain table does. `ECE raw` moved the other way on two of three (MASSIVE
0.214 → 0.136, XNLI 0.452 → 0.269) because the adapters are now *less* confident off-domain
(`Conf` 0.020 / 0.354; `Brier` 0.983 / 0.670) — read the four columns together, and note that
the typed-decisions and XNLI fits still hit the grid ceiling (T=20.0). This table is the
measurement that `B-5` set out to improve.

## Multi-domain experiment (B-5a) — **both gates pass on the B-11 labels** (2026-09-28)

Five domains trained into one adapter — `support`, `ecommerce`, `agent_tools`, `documents`,
`voice` — English only, 21,000 records, evaluated on `data/eval_en_domains.jsonl` (7,500 rows;
its `support` half is the same records as `eval_en.jsonl`, so **0.945** above is the comparable
support number). **Nothing was released:** the adapters on the Hub are the support-only ones above.

The gate verdict moved with B-11. On the pre-B-11 labels run 5 scored 0.785 on `support` and
**failed** that gate; only the `noul` labels changed (`choice` and `score` labels are untouched),
so the *same weights* re-measure to:

| Domain | released adapter (zero-shot) | best 5-domain run | Δ |
| --- | ---: | ---: | ---: |
| `support` | 0.945 | **0.861** | −0.084 |
| `agent_tools` | 0.335 | **0.929** | +0.594 |
| `documents` | 0.366 | **0.887** | +0.521 |
| `ecommerce` | 0.397 | **0.915** | +0.518 |
| `voice` | 0.523 | **0.807** | +0.284 |
| **overall** | 0.513 | **0.880** | +0.367 |

| Primitive | released | best run | ECE (best run) |
| --- | ---: | ---: | ---: |
| `choice` | 0.476 | 0.789 | 0.046 ✅ |
| `noul` | 0.697 | 0.946 | 0.042 ✅ |
| `score` | 0.365 | 0.905 | 0.048 ✅ |

**Gates:** worst new domain ≥ 0.70 → **pass** (`voice` 0.807; `agent_tools` 0.929, `documents`
0.887, `ecommerce` 0.915). Support ≥ 0.85 → **pass** (**0.861**, was 0.785 on the pre-B-11
labels). Per-domain ECE is 0.045–0.067 (was 0.082–0.116): `agent_tools` 0.045 and `documents`
0.046 meet the 0.05 target, the other three stay declared exceptions — one scalar temperature per
primitive fitted across five domains at once.

Two honest caveats. **Run 5 was trained on pre-B-11 labels**, so a retrain on corrected data is
the cheap untried option; and the baseline moved with it — the released support-only adapter now
scores **0.945** on `support`, so the five-domain adapter clears the *absolute* gate while sitting
**0.084 below** the released one. Both numbers are here so the structural options below are read
against the right yardstick.

What the six runs established: the binding constraint is neither capacity nor time.
`choice_rank` 32 → 128 → 256 with 4 → 6 → 8 epochs moved support 0.655 → 0.785 → 0.795 (pre-B-11
labels), while the *shared* low-rank `choice` head under-fits whichever domain it under-fits last —
the fourth-option leak simply **moves between runs** (`billing → other` in run 5; `billing → sales`
in run 6 while `ecommerce` grew `returns → catalog` 117). `choice`/`score` are unchanged by B-11
(0.789 / 0.905 before and after), so that diagnosis stands; the full six-run curve, the
per-(domain, primitive) cross tables and the three structural options live in
`.specs/project/BACKLOG.md` **B-5**.

## Known limitations

- Per-language ECE is above the `0.05` target for three of six multilingual languages (`nl` 0.110,
  `de` 0.101, and `fr` at 0.0502); `es` (0.040), `it` (0.044) and `pt` (0.025) meet it (NFR-C06).
  Multilingual `noul` ECE is 0.120 and English `score` ECE 0.067 — both declared.
- **`choice` costs `noul` in the multilingual checkpoint.** The corrected labels are balanced and
  text-derived, and the shared trunk trades ~0.20 of `choice` for ~0.22 of `noul` against the
  pre-B-11 weights (0.468 vs 0.674, 0.832 vs 0.614) — overall washes out at 0.718 vs 0.714. The
  sweep in the multilingual section above shows neither epochs nor head capacity recover it; the
  structural options are in `BACKLOG.md` **B-5**.
- Calibrated ECE is measured in-sample on the held-out synthetic split. The public probes above
  are the opposite case — public distributions, evaluation-only, no shared states with training.
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.241** while scoring **0.945** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
