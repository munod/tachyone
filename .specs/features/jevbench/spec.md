# JevBench Preparation (B-13) Specification

**Phase:** Post-M6 · **Spec accepted:** 2026-10-02
**Status:** **P0 + P1 + P2 measured (2026-10-02)** — JB-1…JB-7 done. **P2 verdict:
Intelligence 8.4 → 15.0 (control 9.0), all B-5 gates PASS, cycle target I ≥ 50 NOT met.**
**Open: JB-8** (adopt the treatment artifact vs hold for a second P2 iteration), then
P3/P4.

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
| control (fresh run-5) | 0.9999 | all PASS | 1.44 | 9.0 | 65.3 | 84.5 | 78.8 |
| **treatment (mixture)** | **0.9999** | **all PASS** | **4.19** | **15.0** | 48.4 | 84.7 | 78.8 |

Attribution: control +0.6 vs treatment **+6.0** Intelligence — the gain rides on the
data, not run variance (L-006/L-012). **Target I ≥ 50 missed (15.0).** Remaining
distance is concentrated where the architecture is thin: `noul` never reads the rubric
(`hard/trap` 0.000, `standard/policy` ≈ chance), and `hard/temporal_numeric` regressed
(0.267 → 0.133). Biggest gains: `standard/ordinal` 0.25 → 0.667, `score` 0.222 → 0.611,
`hard/multi_hop` 0.056 → 0.333, `easy/intent` 0.583 → 0.750. Full record:
`.specs/features/jevbench/tasks.md` JB-7; artifacts
`benchmarks/results/en_jev_{published,ctrl,treat}.json`.

## Plan (P1–P4)

| Phase | Work | Gate |
| --- | --- | --- |
| **P1** hardening | **DONE (2026-10-02)** — see *P1 results* above: wire audit green ×2, context A/B measured → reverted (L-014), serve fast-off, cost basis measured | 231/231 valid; A/B recorded |
| **P2** Intelligence | family-shaped training data in two layers: **(a)** real public sources (MultiNLI/BoolQ/Banking77, + SST-5/AG News only after licence review — tev1 `DATA_SOURCES.md`) converted to our record shape with pinned provenance; **(b)** synthetic **executable rule trees** for `long_policy`/`multi_hop`/`temporal_numeric`/`trap` + the six original families | public items **evaluation-only**; targets: easy ≥ 0.95, standard ≥ 0.73 → **I ≥ 50**; ablation with control (L-006), one harness (L-013) |
| **P3** Calibration | ECE 0.543 → ≤ 0.15: flatten confidence when the best similarity is weak (argmax untouched → Intelligence independent); diverse holdout fit; optional B-6 contrastive | `calibration(ece)` ≥ 60 on the public run |
| **P4** submission | issue `[bench request]`: pinned `munod/tachyone-en` + base `answerdotai/ModernBERT-large`, licences, inference command, `temperature.json`, this diagnostic, cost basis; `docs/jevbench.md` + CHANGELOG | docs gate + issue filed |

## Out of scope

- Training or calibrating on public or sealed JevBench items (gap penalty; tev1 and every
  careful entrant draw the same line).
- Vendoring the harness or its scoring code into this repository (AD-011).
- Non-English checkpoints (the suite is English-only).

**Requirement traceability:** no public API change ⇒ hard rule 1 not triggered; tests are
co-located per hard rule 6 when P1 lands.
