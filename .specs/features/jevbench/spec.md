# JevBench Preparation (B-13) Specification

**Phase:** Post-M6 · **Spec accepted:** 2026-10-02
**Status:** **P0 done (2026-10-02)** — baseline measured on the 231 public items;
P1 (hardening) next.

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
uv run tachyone-serve                       # detached; health on /health

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

## Plan (P1–P4)

| Phase | Work | Gate |
| --- | --- | --- |
| **P1** hardening | (a) wire audit — **done in P0** (231/231); (b) English `context` 512 → long enough for the hard tier, **A/B measured, not assumed** (mean-pool shift risk); (c) fast path + p50/p95; (d) cost basis documented | public run: accuracy up, latency accounted |
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
