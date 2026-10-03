# JevBench Preparation (B-13) Specification

**Phase:** Post-M6 · **Spec accepted:** 2026-10-02
**Status:** **JB-1 … JB-14 (P3) executed 2026-10-02 — the calibration assets are adopted;
P3's acceptance (`calibration(ece)` ≥ 60) is recorded as NOT met at 52.6**; JB-1 … JB-8
done (2026-10-02) — the B-13 artifact is adopted and published
([`munod/tachyone-en` `f28103bf`](https://huggingface.co/munod/tachyone-en/commit/f28103bf4a85bacd10df90c21125c84522c82993),
six files sha256-verified). Final record: Intelligence **8.4 → 15.0** on the 231 public items
(control +0.6), all B-5 gates PASS, probes XNLI 0.566 / typed-decisions 0.367 / MASSIVE 0.051,
head-to-head 0.288 (theirs) / 1.000 (ours). **P3 executed and adopted the same day — see
*P3 results* below: Calibration 0.0 → 52.6 and composite 1.59 → 4.28 with Intelligence
invariant at 15.0, but the recorded gate (≥ 60) stands NOT met.** P4 (the `[bench request]`
issue) is the only open phase; the cycle's own I ≥ 50 target stands recorded as **not met**.

**Context:** `.specs/project/BACKLOG.md` B-13 · external harness:
<https://github.com/fstandhartinger/jevbench> (MIT — cloned to a local download for
measurement, **never vendored**, per AD-011 rule e)

## Why

JevBench ranks Jev-class decision models on four equally-weighted axes — Intelligence
(chance-corrected accuracy per tier), Calibration (ECE + fidelity to gold distributions),
Speed and Cost — with a `(I/50)²` multiplier below 50 Intelligence. It speaks exactly the
wire Tachyone already freezes (`POST /v1/systemone`, adapter `typesafe`), so the project's
quality work gets an external, public scoreboard.

## Locked decisions (2026-10-02)

1. **Measure first (P0)** — no training before the baseline exists.
2. **Submission as an offline artifact** (the issue #158 pattern): pinned Hub weights,
   evaluator runs the sealed set on their own hardware; no API flag, no endpoint exposure.
3. **Scope = `checkpoints/en` only** — JevBench is English-only.
4. **Training data follows the family shape** (P2), with the public items
   **evaluation-only for ever**: never trained on, never calibrated on (a public-to-sealed
   gap > 25 pp is penalised; the tev1 recipe makes the same choice — "No JevBench items
   were used").
5. **B-6 (contrastive) stays an optional experiment inside P3**, with the JevBench public
   items as its measurement surface instead of MASSIVE.

## What the harness is

534 decisions — **231 public** (48 `easy` + 72 `standard` + 111 `hard`; the `judge` tier
146 items has **no** public items) + 308 sealed. Public tier mapping comes from
`results/v1.2/jevbench-v1.2-per-task.json`. Tier chances: easy 0.284, standard 0.317,
judge 0.292, hard 0.336; tier weights 0.14 / 0.28 / 0.28 / 0.30.

Scoring (`jevbench/composite_v13.py`): Intelligence = weighted chance-corrected tier
accuracy (missing tiers renormalised); Calibration = `100·(1 − ECE/0.5)` (ECE-only
variant used below — gold TVD distributions are sealed); Speed = `100 − 20·log10(s/0.1)`
on p50/p95 with self-hosted `×2 + 0.15 s`; Cost = `100 − 30·log10($/1000 ÷ $0.001)`;
composite = equal-weight geometric mean × `(I/50)²` when `I < 50`.

## P0 baseline — Tachyone `0.6.0`, `checkpoints/en` (2026-10-02)

Wire contract: **231/231 strict-valid, 0 renormalised, 0 failed** — the adapter works
unchanged (the P1(a) audit passed on the first run).

| axis | Tachyone | Laya (same trunk class, board #33) |
| --- | ---: | ---: |
| Intelligence (public-only, judge renormalised) | **8.4** | 45.8 |
| Calibration (ECE-only) | **0.0** (ECE 0.543) | 62.5 |
| Speed (adjusted p50 0.403 s / p95 0.816 s) | **84.8** | 71.1 |
| Cost ($0.00510 / 1000 decisions @ $0.01/M) | **78.8** | 86.2 |
| **Composite (public diagnostic)** | **0.43** (multiplier 0.028) | 54.4 |

Per tier (chance in parentheses): easy **0.500** (0.284) · standard **0.361** (0.317) ·
hard **0.288** (0.336 — *below* chance, cc clipped to 0) · judge no public items.

Per primitive: `noul` **0.473** (binary ≈ chance 0.5) · `choice` **0.309** ·
`score` **0.222** (ordinal MAE 0.813).

Worst families: `hard/trap` 0.000 (n=8), `hard/multi_hop` 0.056 (n=18),
`hard/routing_hard` 0.200, `standard/ordinal` 0.250, `hard/long_policy` 0.263,
`hard/temporal_numeric` 0.267; `easy/extraction` 0.333 and `standard/extraction` 0.333
(note: extraction is a 4-way choice, chance 0.25).

Same-item comparison (231 public items): vs **Laya** we are right where it is wrong on 24
and wrong where it is right on **77**; vs **Jev 1.13.0**: 8 vs **126**.

**Reading:** speed and cost are already board-viable; the wire needs nothing. Everything
else — Intelligence at 8.4 (multiplier 0.028) and Calibration zeroed by ECE 0.543 — is
the whole gap. Both were predicted by the probes (XNLI ≈ chance, `ECE raw` 0.35–0.65,
"sharp off-domain"). Two structural facts behind the numbers:

- **`noul` ignores the rubric**: the runtime scores `sigmoid(cos(question, state)/T)` —
  no reading of `criteria` — so policy/adequacy judgements sit at binary chance. The lever
  is training data in those families (P2), not the wire.
- **English inference truncates at 512 tokens** (`DEFAULT_CHECKPOINTS[tachyone-en].context`)
  while hard states average **1,079** tokens (max 3,746) — the deciding facts are cut
  before the encoder sees them (P1b).

Reproduce (server must be up):

```bash
TACHYONE_BACKEND=encoder TACHYONE_PORT=8756 \
TACHYONE_ADAPTERS="tachyone-en=$PWD/checkpoints/en" TACHYONE_PRELOAD=tachyone-en \
uv run tachyone-serve                       # detached; health on /health; NO TACHYONE_FAST

cd /tmp/opencode/jevbench && for f in original easy hard; do
  python3 -m jevbench.cli run --tasks datasets/public/$f.jsonl \
    --adapter typesafe --endpoint http://127.0.0.1:8756 --key-env '' \
    --model tachyone-latest --price-in-per-m 0.01 --price-out-per-m 0 \
    --cost-basis derived_usage_times_tariff_encoder_size_class_0.01usd_per_Minput \
    --cap-usd 5 --delay-s 0.05 \
    --results RUN/$f/results.jsonl --raw-dir RUN/$f/raw \
    --ledger RUN/ledger.jsonl --manifest RUN/$f/manifest.json
done
python3 /tmp/opencode/analyze_p0.py         # axes via the harness's composite_v13
```

Artifacts (outside the repo): `/tmp/opencode/jevbench_runs/` (results, manifests, ledger,
`p0_baseline.json`), `/tmp/opencode/analyze_p0.py`, local clone `/tmp/opencode/jevbench`.

## P1 results (2026-10-02)

- **(a) Wire audit — green twice:** 231/231 `strict_valid`, 0 renormalised, 0 failed on
  both the P0 and the P1 run. The `typesafe` adapter needs nothing from us (hard rule 1
  not triggered).
- **(b) Long context — measured and REVERTED (lesson L-014).** `context` 512 → 4096
  (`15792d4`), full re-run of the 231 items: **1 item flipped right, 1 flipped wrong —
  82/231 both runs, Intelligence 8.4 → 8.4** — while raw p95 went **333 ms → 2,700 ms**
  and the Speed axis **84.8 → 77.6**. Seeing the deciding facts is not the binding
  constraint for a similarity encoder; knowing what to do with them is (that is P2).
  Reverted in `4bb5a6a` with the A/B recorded in the code comment and in
  `tests/test_encoder.py::test_english_context_matches_the_trained_length`. Revisit only
  together with a longer *training* `max_len`.
- **(c) Speed / fast path — serve with `TACHYONE_FAST` off (the default).** End-to-end at
  the short shapes this suite mostly sends, fast-on measured *slower* (p50 126.7 ms on vs
  82.0 ms off — the micro-benchmark's 2.68× is forward-only, not HTTP-serving), and at the
  raised context fast-on **OOMed**: 18.8 GiB in CUDA-graph private pools on a 22 GiB GPU
  (per-shape capture was built for fixed short shapes). Also: **never set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on this stack** — the control run
  held 20.5 GiB at warm-up (vs 1.9 GiB without) and leaked 24.5 GiB of GPU memory across
  both devices when the process was killed (unrecoverable without a reboot; GPU0 keeps
  12.0 GiB free — the historical training envelope).
- **(d) Cost basis — measured and ready for submission:** mean **510 input tokens per
  decision** (max 3,746), `$0.00510 / 1000 decisions` at the encoder size-class price
  `$0.01/M in, $0 out` (the same basis they used for Laya) → Cost axis **78.8**.

## P2 results (2026-10-02)

Two arms one factor apart (the data), same recipe/seed, both trained and fit, one
harness, explicit adapter (L-013), calibration refit per arm:

| arm | in-domain overall | gates B-5 | bench score | Intelligence | Calibration | Speed | Cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| published (P0 reference) | 0.9637 | all PASS | 0.43 | 8.4 | 0.0 | 85.0 | 78.8 |
| control (fresh run-5) | 0.9999 | all PASS | 0.51 | 9.0 | 0.0 | 84.6 | 78.8 |
| **treatment (mixture)** | **0.9999** | **all PASS** | **1.59** | **15.0** | 0.0 | 84.3 | 78.8 |

Attribution: control +0.6 vs treatment **+6.0** Intelligence — the gain rides on the
data, not run variance (L-006/L-012). **Target I ≥ 50 missed (15.0).** Calibration is 0
for every arm as shipped (L-015: the in-domain refit sharpens to T=0.05 and zeroes the
axis off-domain — measured consistently, asset present, for all three). Remaining
distance is concentrated where the architecture is thin: `noul` never reads the rubric
(`hard/trap` 0.000, `standard/policy` ≈ chance), and `hard/temporal_numeric` regressed
(0.267 → 0.133). Biggest gains: `standard/ordinal` 0.25 → 0.667, `score` 0.222 → 0.611,
`hard/multi_hop` 0.056 → 0.333, `easy/intent` 0.583 → 0.750. Full record:
`.specs/features/jevbench/tasks.md` JB-7; artifacts
`benchmarks/results/en_jev_{published,ctrl,treat}.json`.

## P3 design (pre-registered 2026-10-02, before any P3 asset was fitted)

**Diagnosis (measured, not assumed).** The shipped asset sharpens to `T=0.05/0.05/0.1`
(L-015). Reconstructing the pre-temperature scores from the published run and sweeping `T`
over the three surfaces gives, per primitive (ECE, top-label, 10 bins):

| `T` | `choice` public / in-domain / source-remainder | `noul` public / in-domain / source-remainder | `score` public / in-domain |
| --- | --- | --- | --- |
| 0.20 (shipped-ish) | 0.507 / 0.0002 / 0.150 | 0.460 / 0.029 / 0.168 | 0.285 / 0.471 |
| 0.50 | 0.277 / 0.0005 / 0.138 | 0.321 / 0.183 / **0.026** | 0.415 / 0.295 |
| 0.75 | 0.130 / 0.0013 / 0.125 | 0.212 / 0.267 / 0.065 | 0.429 / 0.132 |
| 1.00 (natural) | 0.069 / 0.0045 / 0.107 | 0.167 / 0.318 / 0.100 | 0.391 / **0.028** |
| 1.50 | **0.028** / 0.026 / 0.059 | 0.118 / 0.375 / 0.152 | 0.361 / 0.081 |

**No global temperature serves all three surfaces** — `noul` alone wants `0.2` at home,
`0.5` on the source remainder and `≥1.5` on the public items. Two structural facts decide
the design:

1. **`noul`'s runtime signal carries no difficulty information.** `cos(question, state)`
   reads 0.68 (in-domain) vs 0.65 (public) — and within the public items it does *not*
   separate right from wrong (AUC 0.477). Its temperature can only trade one surface for
   another.
2. **An in-domain-ness signal does exist and is exact.** `strength = max_k cos(state,
   prototype_k)` over k-means centroids (K=32) of the checkpoint's own training states:
   in-domain **0.991** (p10 0.979) vs public **0.702** (p90 0.907), **AUC 1.000** — holding
   when the prototypes are built from the real 35,540-record mixture.

**Legal calibration surfaces** (never trained on, never a JevBench item, overlap asserted):

| slice | n | accuracy | strength |
| --- | ---: | ---: | ---: |
| fresh in-domain eval (new seed) | 1,500 | ~1.00 | 0.99 |
| source remainder (MultiNLI + BoolQ train rows not in `train_jev_sources`) | 1,200 | 0.806 | 0.45–0.75 (p50 0.72) |
| fresh families, seed 7 (zero public overlap) | 1,000 | 0.843 | 0.743 |
| **truncation slice** — fresh families behind a document preamble, decisive clause past the 512-token window | 1,000 | **0.379** (`noul` **0.487**) | 0.715 |

The truncation slice is the one that covers the bench's difficulty: P1 already recorded
that hard states average 1,079 tokens against a 512-token window (L-014), so ">512-token
inputs" is a *known deployment condition*, not a benchmark artefact — and its accuracy
lands where the public items are (0.379 vs 0.420).

**Design (locked for this cycle):**

- **`choice` — one global temperature** fitted on the pooled legal holdout with the existing
  `fit_temperature` grid. Its own confidence is already informative in every regime
  (public 0.41 at `T=1` against 0.36 accuracy).
- **`noul` — instance-dependent confidence.** `p = g(strength)` when the natural answer is
  "yes", `1 − g(strength)` when it is "no": the **direction comes from the answer, the
  magnitude from the evidence**. `g` is a piecewise-linear map over strength bins fitted to
  empirical accuracy on the same holdout. Argmax is preserved by construction.
- **`score` — temperature pinned at the shipped 0.1.** Its answer is the distribution's
  *expected value*, so temperature moves the answer (in-domain rounded-EV accuracy is 0.53
  at `T=1` vs 0.9996 shipped); it also has no off-domain holdout rows. Changing it would
  move Intelligence, which this phase must not do.
- **Pre-registered fit protocol:** `fit_temperature` on the pooled holdout rows, `DEFAULT_GRID`,
  per primitive, `score` excluded (pinned); `g` = mean accuracy per strength bin, monotone
  enforced. No public item is a fit input at any point; the public run is measured **once**
  after the assets are frozen.
- **Mechanism note (L-014 → P3):** only confidence moves. Every answer — `choice` argmax,
  `noul` direction, `score` expected value — is provably untouched, so Intelligence and all
  B-5 gates are invariant by construction, not by luck.

**Acceptance (P3):** public run ECE ≤ 0.15 → `calibration(ece)` ≥ 60; in-domain ECE per
primitive ≤ 0.05 with accuracy byte-identical to the published set; `eval_en_domains`,
probes and head-to-head accuracies unchanged; wire untouched (contract test unchanged).

## P3 results (2026-10-02)

Assets frozen first, public run measured **once**, in-domain gate checked against the
published report:

| surface | published | **P3** | gate |
| --- | ---: | ---: | --- |
| in-domain accuracy (n=7,500, every primitive and domain) | 0.999867 | **0.999867** | must not move ✓ |
| in-domain ECE choice / noul / score | 0.000007 / 0.000012 / 0.000366 | **0.025886 / 0.000004 / 0.000366** | each ≤ 0.05 ✓ |
| public ECE (231 items) | 0.5393 | **0.2369** | ≤ 0.15 ✗ |
| **Calibration axis** | **0.0** | **52.6** | **≥ 60 ✗** |
| Intelligence | 15.0 | **15.0** | invariant ✓ |
| Brier | 1.109 | **0.745** | — |
| **composite** | 1.59 | **4.28** | — |

Wire 231/231, 0 failed. Fitted values: **choice T 0.05 → 1.5**, **noul map** (10 knots,
strength 0.410 → 0.9966 ⇒ confidence 0.567 → 1.0) with T 0.25 kept only as the no-map
fallback, **score pinned at 0.1** (its answer is the expected value). Contribution to the
0.2369: `choice` 0.1421 · `noul` 0.0953 · `score` 0.0201.

**Why the gate was missed — two facts, both recorded rather than smoothed:**

- **The pre-registration's estimate was wrong.** "T=1 gives public ECE 0.086" came from
  reconstructing the pre-temperature distribution out of *saturated* `T=0.05` output, where
  precision is already gone; the simulated natural distribution was nearly uniform.
  Measured properly: natural **0.2376** vs fitted **0.2369** — the assets are as good as
  natural, not 3× better (**L-016**).
- **No legal fit set can see bench difficulty.** At equal evidence the model scores 0.84 on
  never-trained MultiNLI and 0.36 on the bench, so `choice`'s pooled-optimum T and the
  `noul` plateau (0.777) are *correct for the data we may fit on*. The one legal slice in
  the bench's regime (L-014 truncation slice: 0.379 accuracy, 0.715 strength) was removed by
  the 2026-10-02 direction to build everything on `train_en_domains`. Passing ≤ 0.15 needs
  bench-difficulty legal data or the P2 capability fix — not another fit.

**Adopted** because Intelligence is invariant by construction, home moves from an
in-sample 0.000116 to an honest 0.008508 at byte-identical accuracy, Brier halves and the
board composite goes 1.59 → 4.28. The acceptance box stays unticked.

## Plan (P1–P4)

| Phase | Work | Gate |
| --- | --- | --- |
| **P1** hardening | **DONE (2026-10-02)** — see *P1 results* above: wire audit green ×2, context A/B measured → reverted (L-014), serve fast-off, cost basis measured | 231/231 valid; A/B recorded |
| **P2** Intelligence | family-shaped training data in two layers: **(a)** real public sources (MultiNLI/BoolQ/Banking77, + SST-5/AG News only after licence review — tev1 `DATA_SOURCES.md`) converted to our record shape with pinned provenance; **(b)** synthetic **executable rule trees** for `long_policy`/`multi_hop`/`temporal_numeric`/`trap` + the six original families | public items **evaluation-only**; targets: easy ≥ 0.95, standard ≥ 0.73 → **I ≥ 50**; ablation with control (L-006), one harness (L-013) |
| **P3** Calibration | **DONE (2026-10-02), adopted — see *P3 results*** (evidence-conditioned `noul` confidence from a training-prototype bank, `choice` T 1.5, `score` pinned; fitted on legal slices only) | **NOT met**: `calibration(ece)` 52.6 < 60 (ECE 0.2369 > 0.15). In-domain half met exactly: ECE ≤ 0.05, answers byte-identical ✓ |
| **P4** submission | issue `[bench request]`: pinned `munod/tachyone-en` + base `answerdotai/ModernBERT-large`, licences, inference command, `temperature.json`, this diagnostic, cost basis; `docs/jevbench.md` + CHANGELOG | docs gate + issue filed |

## Out of scope

- Training or calibrating on public or sealed JevBench items (gap penalty; tev1 and every
  careful entrant draw the same line).
- Vendoring the harness or its scoring code into this repository (AD-011).
- Non-English checkpoints (the suite is English-only).

**Requirement traceability:** no public API change ⇒ hard rule 1 not triggered; tests are
co-located per hard rule 6 when P1 lands.
