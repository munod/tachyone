# Benchmarks

Reproducible accuracy/ECE/latency results. The full artifact, environment capture, and exact
reproduction commands live in
[`benchmarks/report.md`](https://github.com/munod/jeba/blob/main/benchmarks/report.md).

**Setup:** single RTX 3060 12GB · 9,000 English / 18,000 multilingual deterministic synthetic
train records (fully localized per language, per-record RNG) · 1,500 held-out eval · LoRA (r=16)
plus a low-rank `choice` head (r=32) · 4 epochs · bf16 + gradient checkpointing. Calibrated ECE is
after per-`(primitive, language)` temperature fitting on the held-out split (in-sample).

## English (ModernBERT-large + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.781 | 0.077 | 22.947 | 37.973 |
| `choice` | 500 | 0.708 | 0.141 | 36.743 | 38.657 |
| `noul` | 500 | 0.744 | 0.012 | 14.508 | 22.257 |
| `score` | 500 | 0.892 | 0.090 | 22.591 | 24.917 |

## Multilingual (mmBERT-base + LoRA + choice head)

| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| overall | 1500 | 0.711 | 0.073 | 12.403 | 20.261 |
| `choice` | 500 | 0.734 | 0.100 | 18.843 | 21.084 |
| `noul` | 500 | 0.716 | 0.082 | 11.157 | 15.169 |
| `score` | 500 | 0.682 | 0.067 | 12.160 | 16.022 |
| lang `pt` | 252 | 0.865 | 0.045 | 12.395 | 20.041 |
| lang `fr` | 249 | 0.855 | 0.039 | 12.557 | 20.278 |
| lang `it` | 249 | 0.827 | 0.068 | 12.249 | 44.317 |
| lang `nl` | 249 | 0.606 | 0.178 | 12.389 | 19.985 |
| lang `de` | 249 | 0.598 | 0.119 | 12.469 | 19.940 |
| lang `es` | 252 | 0.512 | 0.182 | 12.464 | 19.441 |

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

- Per-language ECE is still above the `0.05` target for `es`/`nl`/`de` and multilingual `score`.
- Calibrated ECE is measured in-sample on the held-out synthetic split.
- Numbers are from deterministic synthetic data; public-probe comparisons (MASSIVE, XNLI,
  typed-decisions) are evaluation-only and not yet published.
