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

mmBERT + `checkpoints/multi`, 56 held-out states, batch=1; `TACHYONE_FAST=1` uses per-shape CUDA
graphs with bf16-resident weights. Capture is warmed before timing.

| Path | p50 (ms) | p95 (ms) | Throughput b1 (items/s) | b4 | b16 |
| --- | --- | --- | --- | --- | --- |
| stock (fp32) | 10.267 | 11.047 | 93.8 | 161.7 | 71.9 |
| fast (CUDA graphs, bf16) | 3.834 | 4.119 | 238.8 | 404.8 | 163.8 |

**Parity:** max absolute answer-probability difference **0.0044** with **0 top-label flips** (16
questions); response shape unchanged. **NFR-P01** (p50 ≤ 20 ms, p95 ≤ 50 ms) is met by both paths,
with a **2.68× p50 speedup** for the fast path.

## Public probes (B-7)

Three public datasets Tachyone did **not** train on, scored through the same metric code as
everything else. Full tables (including per-language and per-config accuracy *and* ECE), licences,
citations and the exact reproduction commands live in
[`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md).

| Probe | Licence | n | Accuracy | Chance | ECE raw |
| --- | --- | ---: | ---: | ---: | ---: |
| typed-decisions (`LocalLLaMA`, 4 configs) | Apache-2.0 | 2000 | 0.323 | 0.20–0.50 | 0.408 |
| MASSIVE intents (7 languages) | CC-BY-4.0 | 3584 | 0.033 | 0.017 | 0.214 |
| XNLI (`en`) | CC BY-NC 4.0 | 5010 | 0.334 | 0.333 | 0.452 |

Evaluation only — no probe row has ever reached `training/`. Read each number against its
**chance** level and against the in-sample 0.859 above, not against the other rows: these are the
released adapters zero-shot on tasks they were never trained for. MASSIVE lands at chance in
every language (0.016–0.047 over 60 intents), XNLI is exactly at chance (three labels), and
typed-decisions — the one dataset that already speaks our wire — reaches 0.323 (best config
0.402, worst 0.218). All three temperature fits hit the grid ceiling (T=20.0), so a low
calibrated ECE is bought by flattening; `Conf` and `Brier` in `probes.md` expose it. This table is
the measurement that `B-5` set out to improve.

## Multi-domain experiment (B-5a) — 1 of 2 gates met (2026-09-27)

Five domains trained into one adapter — `support`, `ecommerce`, `agent_tools`, `documents`,
`voice` — English only, 21,000 records, evaluated on `data/eval_en_domains.jsonl` (7,500 rows;
its `support` half is byte-identical to `eval_en.jsonl`, so the 0.859 above is the comparable
support number). **Nothing was released:** the adapters on the Hub are still the support-only ones
above, because the B-5 gate *no regression on support* did not clear. The full six-run curve, the
per-(domain, primitive) cross tables and the diagnosis live in `.specs/project/BACKLOG.md` **B-5**.

| Domain | released adapter (zero-shot) | best 5-domain run | Δ |
| --- | ---: | ---: | ---: |
| `support` | 0.859 | **0.785** | −0.074 |
| `agent_tools` | 0.478 | **0.849** | +0.371 |
| `documents` | 0.501 | **0.810** | +0.309 |
| `ecommerce` | 0.537 | **0.838** | +0.301 |
| `voice` | 0.513 | **0.739** | +0.226 |
| **overall** | 0.578 | **0.804** | +0.226 |

| Primitive | released | best run | ECE (best run) |
| --- | ---: | ---: | ---: |
| `choice` | 0.502 | 0.789 | 0.046 ✅ |
| `noul` | 0.707 | 0.720 | 0.171 ❌ |
| `score` | 0.524 | 0.905 | 0.048 ✅ |

**Gates:** worst new domain ≥ 0.70 → **pass** (`voice` 0.739; `agent_tools` 0.849, `documents`
0.810, `ecommerce` 0.838). Support ≥ 0.85 → **fail** (0.785, −0.065). Per-domain ECE is
published: 0.082–0.116, **all above the 0.05 target** and declared as exceptions — one scalar
temperature per primitive is fitted across five domains at once.

What the six runs established: the binding constraint is neither capacity nor time.
`choice_rank` 32 → 128 → 256 with 4 → 6 → 8 epochs moved support 0.655 → 0.785 → 0.795, while the
*shared* low-rank `choice` head under-fits whichever domain it under-fits last — the
fourth-option leak simply **moves between runs** (`billing → other` in run 5; `billing → sales` in
run 6 while `ecommerce` grew `returns → catalog` 117). Closing the last 0.065 needs a structural
decision (per-domain adapters, or two-stage training), which is an ADR rather than a
hyper-parameter.

## Known limitations

- Per-language ECE is above the `0.05` target for four of six languages (`nl` 0.104, `de` 0.063,
  `it` 0.059, `fr` 0.051); only `es` (0.038) and `pt` (0.024) meet it.
- Calibrated ECE is measured in-sample on the held-out synthetic split. The public probes above
  are the opposite case — public distributions, evaluation-only, no shared states with training.
- **Domain coverage is narrow.** Run against an external public probe (the peer scorer's own nine
  families — news, banking intents, emotions, MMLU, reviews, tickets), the released English
  adapter scores **0.229** while scoring **0.854** on its own support records. That is the
  adapter's training distribution, not a ceiling for the engine; broadening it is backlog `B-5`.
  Both tables, the method and the caveats (in-sample synthetic, different `n` per engine, ECE read
  together with `Conf`/`Brier`) are published in [`compare.md`](compare.md).
