# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/jeba/blob/main/benchmarks/report.md).

**Setup:** single RTX 3060 12GB · 9,000 English / 18,000 multilingual deterministic synthetic
train records (fully localized per language, per-record RNG) · 1,500 held-out eval · LoRA (r=16
English, **r=64 multilingual**) plus a low-rank `choice` head (r=32) · 4 epochs · bf16 + gradient
checkpointing. Calibrated ECE is after per-`(primitive, language)` temperature fitting on the
held-out split (in-sample). The **noisy view** applies one surface edit (typo/accents/casing) to
15% of states (B-4).

## English (ModernBERT-large + LoRA r=16 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.859 | 0.023 | 23.307 | 39.010 |
| `choice` | 500 | 0.948 | 0.020 | 36.849 | 40.910 |
| `noul` | 500 | 0.718 | 0.020 | 14.841 | 22.032 |
| `score` | 500 | 0.910 | 0.039 | 22.956 | 25.211 |

Noisy view (noise_rate 0.15): overall accuracy **0.854**, ECE **0.026**.

## Multilingual (mmBERT-base + LoRA r=64 + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.853 | 0.038 | 13.254 | 23.189 |
| `choice` | 500 | 0.684 | 0.083 | 19.627 | 24.108 |
| `noul` | 500 | 0.960 | 0.038 | 12.003 | 17.053 |
| `score` | 500 | 0.916 | 0.032 | 12.699 | 18.302 |
| lang `es` | 252 | 0.956 | 0.038 | 13.169 | 21.753 |
| lang `pt` | 252 | 0.885 | 0.024 | 13.063 | 21.918 |
| lang `it` | 249 | 0.880 | 0.059 | 13.219 | 47.450 |
| lang `fr` | 249 | 0.871 | 0.051 | 13.397 | 23.058 |
| lang `de` | 249 | 0.863 | 0.063 | 13.493 | 31.312 |
| lang `nl` | 249 | 0.663 | 0.104 | 13.216 | 22.701 |

Noisy view (noise_rate 0.15): overall accuracy **0.847**, ECE **0.042**.

Raising the multilingual LoRA rank from 16 to 64 lifted overall accuracy 0.702 → **0.853** and `es`
ECE 0.170 → **0.038**; two of six languages now meet ECE ≤ 0.05 (`es` 0.038 and `pt` 0.024).

## Fast path (CUDA graphs)

mmBERT + `checkpoints/multi`, 56 held-out states, batch=1; `JEBA_FAST=1` uses per-shape CUDA
graphs with bf16-resident weights. Capture is warmed before timing.

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 10.267 | 11.047 | 93.8 | 161.7 | 71.9 |
| fast (CUDA graphs, bf16) | 3.834 | 4.119 | 238.8 | 404.8 | 163.8 |

**Parity:** max absolute answer-probability difference **0.0044** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **2.68× p50 speedup** for the fast path.

## Known limitations

- Per-language ECE is above the `0.05` target for four of six languages (`nl` 0.104, `de` 0.063,
  `it` 0.059, `fr` 0.051); only `es` (0.038) and `pt` (0.024) meet it.
- Calibrated ECE is measured in-sample on the held-out synthetic split.
- Numbers are from deterministic synthetic data; public-probe comparisons (MASSIVE, XNLI,
  typed-decisions) are evaluation-only and not yet published.
