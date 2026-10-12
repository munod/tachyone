# JevBench

> **Status (2026-10-08):** JevBench was **set aside by decision** — its maintainer
> changed the submission methodology upstream and Tachyone development continued
> (2026-10-10/12: a **measurement-only dry-run** re-read the public set on the `v0.13.0`
> artifact and the **B-20** data cycle answered its verdict — three recorded attempts, the
> mixture layer **not adopted** (a trade, not an improvement); see *Dry-run re-measurement*
> below and `BACKLOG.md` **B-20**).
> (`.specs/project/STATE.md`). Every JevBench number on this page belongs to the
> **v0.8.0 artifact** and is kept as a measurement of that revision; it is **not**
> re-measured for v0.9.0. The current release's numbers are in
> [`benchmarks.md`](benchmarks.md) and [`benchmarks/report.md`](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

**[JevBench](https://github.com/fstandhartinger/jevbench)** (MIT) ranks Jev-class decision
models on four equally-weighted axes — Intelligence (chance-corrected accuracy per tier),
Calibration, Speed and Cost — with a `(I/50)²` multiplier below 50 Intelligence. It speaks
exactly the frozen `/v1/systemone` wire Tachyone ships: its `typesafe` adapter needed
**nothing** from us (231/231 strict-valid on the first run), which is why it works as an
external scoreboard for a project whose whole design is "drop-in Jev contract".

> **Everything on this page is a public-item diagnostic, not an official score.** The
> official number is computed by the maintainer over the sealed set; we never see it, never
> trained on it and never calibrated on it. The 231 public items are evaluation-only **for
> ever** — our builders refuse to write a record that collides with one of them.

## The one set (English artifact, 2026-10-02)

Measured on the 231 published items (48 easy / 72 standard / 111 hard — the frozen v1.2 set
shipped in `datasets/public/`) against **`munod/tachyone-en` revision
[`1c88ebef`](https://huggingface.co/munod/tachyone-en/commit/1c88ebef8f15f68e8a6583c564d636221f22c29f)**
through the harness's own adapter:

| Axis (public-only) | Value | Note |
| --- | ---: | --- |
| Intelligence | **15.0** | judge tier has no public items, so the weights renormalise over three tiers |
| Calibration | **52.6** | ECE-only variant (`100·(1 − ECE/0.5)`); gold distributions are sealed |
| Speed | **84.9** | raw p50 70.0 ms / p95 487.9 ms, self-hosted `×2 + 0.15 s` → 0.290 s / 1.126 s |
| Cost | **78.8** | 510 input tokens/decision → $0.00510 per 1,000 decisions |
| **Composite** | **4.28** | `(I/50)²` multiplier 0.089 — the board's own formula, on public items only |

| Tier | n | Accuracy | Chance | Chance-corrected |
| --- | ---: | ---: | ---: | ---: |
| easy | 48 | 0.625 | 0.284 | 47.6 |
| standard | 72 | 0.417 | 0.317 | 14.6 |
| hard | 111 | 0.333 | 0.336 | 0.0 (below chance, clipped) |
| judge | 0 | — | 0.292 | no public items |

| Primitive | n | Accuracy |
| --- | ---: | ---: |
| `noul` | 74 | 0.487 |
| `choice` | 139 | 0.360 |
| `score` | 18 | 0.611 (ordinal MAE 0.575) |

Wire: **231/231 strict-valid, 0 renormalised, 0 failed.** ECE (top-label, 10 bins)
**0.2369**, Brier **0.7451**.

**Two recorded misses, both ours and both kept visible:** the cycle's own target was
**Intelligence ≥ 50** and it landed at **15.0** (`noul` never reads the rubric, so `hard/trap`
stays 0.000); the P3 calibration target was **≥ 60** and it landed at **52.6**. The full
record — including why no legal fit set can see bench difficulty — is in
`.specs/features/jevbench/spec.md`.

## Dry-run re-measurement (2026-10-10, `v0.13.0` — measured only, no submission)

Owner decision: answer "would the board read better today?" without re-engaging the
set-aside benchmark. **Same protocol** (231 public items at the pinned rev `bb05a335`, the
harness's own `typesafe` adapter against an explicit-adapter serve; 231/231 strict-valid,
0 failed):

| Axis (public-only) | 2026-10-02 (`v0.8.0`) | 2026-10-10 (`v0.13.0`) | Δ |
| --- | ---: | ---: | --- |
| Intelligence | 15.0 | **10.5** | −4.5 |
| Calibration | 52.6 | **85.9** | **+33.3** |
| Speed | 84.9 | 85.0 | = |
| Cost | 78.8 | 78.8 | = |
| **Composite** | **4.28** | **2.17** | pulled by the `(I/50)²` multiplier |
| public ECE | 0.2369 | **0.0704** | **P3's ≥ 60 / ECE ≤ 0.15 target is MET now** |

| Tier | acc then | acc now | chance-corr then → now |
| --- | ---: | ---: | --- |
| easy | 0.625 | 0.417 | 47.6 → 18.5 |
| standard | 0.417 | 0.431 | 14.6 → 16.6 |
| hard | 0.333 | 0.342 | 0.0 → 1.0 |

| Primitive | then | now |
| --- | ---: | ---: |
| `noul` | 0.487 | **0.554** |
| `choice` | 0.360 | 0.288 |
| `score` | 0.611 | 0.444 |

**Both directions decompose cleanly.** Calibration is much better because B-16/B-18
replaced the temperature-only chain with strength-keyed fitted maps — exactly the axis P3
failed. Intelligence is worse because the October artifact *was* the mixture retrain
(MultiNLI + BoolQ + Banking77 — the "+6.0 treatment") and the B-15 recompose is
**synthetic-only**: `easy/intent` 0.750 → 0.250 and `hard/adversarial` 0.667 → 0.167 (the
natural-text families), the same loss the XNLI probe recorded (0.566 → 0.333). Against
that, `noul` 0.487 → **0.554** and **`hard/trap` 0.000 → 0.500** — the B-11/B-12 label fixes
closed half the rubric-blindness that capped Intelligence — and `hard/temporal_numeric`
0.133 → 0.267. Routed and explicit-`en` are identical: every bench item is English and
routes to `tachyone-en`.

**B-20 answered this verdict directly (2026-10-10/12) — and the answer is a trade.** The
mixture layer came back beside the synthetic recipe in three recorded attempts (a trainer
gradient-clipping fix along the way): Intelligence **10.5 → 12.2** with `easy` 0.417 → 0.479
and `hard` 0.342 → 0.405 and typed-decisions 0.306 → 0.409 — but holdout accuracy
0.9825 → 0.9419, Calibration 85.9 → **49.0**, XNLI 0.333 → 0.285, `standard` 0.431 → 0.361
and `hard/trap` 0.50 → 0.25. **Not adopted (owner decision) — `v0.13.0` stays published.**
The full evidence (three attempts, loss shapes, fit bases) is `BACKLOG.md` **B-20**; the
queued follow-up is a redesign of the layer as a touch-up from the published trunk at a
lower ratio.

### The LLM-backend reference row (same protocol, 2026-10-11)

What a general LLM reads on the same 231 items — not a submission, a reference point for the
two axes a decision engine trades away: `gemma4:26b` through the `llm` backend (local
Ollama on a dedicated GPU; the B-20 training ran untouched on the other).

| Axis (public-only) | `v0.8.0` encoder | `v0.13.0` encoder | **LLM · gemma4:26b** |
| --- | ---: | ---: | ---: |
| Intelligence | 15.0 | 10.5 | **97.7** |
| Calibration | 52.6 | 85.9 | **94.6** |
| Speed | 84.9 | 85.0 | **44.2** |
| Cost | 78.8 | 78.8 | *unmetered (see note)* |
| **Composite** | **4.28** | **2.17** | **75.3** (see notes) |

Accuracy **227/231 (98.3%)** — easy 1.000, standard 1.000, hard 0.964 (`choice` 0.986,
`noul` 0.986, `score` 0.944); ECE **0.0272**. The four misses are genuinely hard: a multi-hop
credit calculation, a vulnerability-triage ladder, a trap item and a numeric `judge` item.

**Notes (read before quoting the composite).**
- **Speed is the trade**: raw p50 **11.0 s** / p95 84.7 s serial local inference (the board's
  `×2 + 0.15 s` adjustment → 22.2 s / 169.6 s) against the encoder's ~70 ms. That gap *is*
  the product thesis.
- **Cost is unmetered**: the LLM ran locally with no metered price, and the October cost
  basis ($0.00510 per 1,000 decisions) was priced for the encoder self-host — it is **not**
  applied to a 26B row. At five times the per-token basis the composite would read ≈ 65
  instead of 75.3; at ten times, ≈ 55.
- The `(I/50)²` multiplier applies **below 50 only** (the board's formula) — above 50 the
  composite is the geometric mean alone.
- Protocol: `qwen3.6:35b` was unusable as-is (the B-17 thinking quirk — empty content while
  reasoning burns the budget), so the row is `gemma4:26b`; the harness client timeout was
  raised 120 → 600 s in the local clone for slow local inference (plumbing only — no scoring
  logic touched); models and runs kept in `/tmp/opencode/jev_dryrun/`.

The honest frame for the three rows: **the decision engine does not compete with an LLM
judge on Intelligence — it competes on Speed, Cost and deployability**, and its calibration
is now in the same class. What the encoder rows must keep fixing is the coverage the
synthetic recipe never sees (that is B-20's job, running since 2026-10-10).

## What is pinned

| | |
| --- | --- |
| Method | `docs/METHOD-v1.5.md` at upstream commit [`bb05a335`](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc), sha256 `c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07` |
| Harness | same commit, cloned (MIT) — **never vendored** (AD-011 rule e) |
| Weights | `munod/tachyone-en` @ `1c88ebef8f15f68e8a6583c564d636221f22c29f` |
| Base | `answerdotai/ModernBERT-large` (Apache-2.0), LoRA r=16 + a low-rank `choice` head |
| Licences | our code Apache-2.0; base Apache-2.0; training sources: MultiNLI (mixed CC-BY-3.0/CC-BY-SA-3.0/MIT), BoolQ (CC-BY-SA-3.0), Banking77 (CC-BY-4.0) — pinned with sha256 in `training/data/jev_sources.lock.json` |
| Calibration | `temperature_calibration.json` (`choice` T=1.5, `score` **pinned** at 0.1) + `confidence_calibration.json` (the `noul` evidence map). **Neither changes a decision** — argmax and expected value are preserved by construction, which is why Intelligence reads the same before and after P3 |
| Cost basis | encoder size class, $0.01/M input, $0 output, measured 510 input tokens/decision |

## Reproduce

```bash
# 1. the harness (pinned; MIT; do not vendor it)
git clone https://github.com/fstandhartinger/jevbench /tmp/jevbench
git -C /tmp/jevbench checkout bb05a335bc809e61b20c0f745d25499a82b326fc

# 2. Tachyone, served with an explicit adapter (L-013) and the fast path off
TACHYONE_BACKEND=encoder TACHYONE_PORT=8756 \
  TACHYONE_ADAPTERS="tachyone-en=$PWD/checkpoints/en" TACHYONE_PRELOAD=tachyone-en \
  uv run tachyone-serve

# 3. the 231 public items through the harness's own typesafe adapter
cd /tmp/jevbench && for f in original easy hard; do
  python3 -m jevbench.cli run --tasks datasets/public/$f.jsonl \
    --adapter typesafe --endpoint http://127.0.0.1:8756 --key-env '' \
    --model tachyone-latest --price-in-per-m 0.01 --price-out-per-m 0 \
    --cost-basis derived_usage_times_tariff_encoder_size_class_0.01usd_per_Minput \
    --cap-usd 5 --delay-s 0.05 \
    --results RUN/$f/results.jsonl --raw-dir RUN/$f/raw \
    --ledger RUN/ledger.jsonl --manifest RUN/$f/manifest.json
done
```

The four axes are then `jevbench/composite_v13.py` over those results — the public-item view
of the frozen method (v1.4/v1.5 fold in the sealed accuracy, which only the maintainer has).

## Submission

Filed **2026-10-02** as **[`fstandhartinger/jevbench#182`](https://github.com/fstandhartinger/jevbench/issues/182)** — an **offline artifact**
submission: pinned Hub weights, licences, the inference command (smoke-tested before filing),
the calibration assets with their sha256, this diagnostic and the cost basis; no API endpoint,
no credential, and no sealed item in the issue. The body also discloses the two things a
submitter should disclose — that **no JevBench item (public or sealed) reached training or
calibration** (asserted by every builder against `datasets/public/`), and that our training
data *was* authored to the benchmark's published family taxonomy. The verbatim body is
recorded in [`.specs/features/jevbench/submission-issue.md`](https://github.com/munod/tachyone/blob/main/.specs/features/jevbench/submission-issue.md).
