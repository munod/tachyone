# JevBench

> **Status (2026-10-08):** JevBench was **set aside by decision** — its maintainer
> changed the submission methodology upstream and Tachyone development continued
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
