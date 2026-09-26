# Non-Functional Requirements

**Status:** Baseline. Targets marked **[open]** still need confirmation after benchmarks.

---

## Performance (NFR-P)

| ID | Requirement | Target | Phase | How verified |
| --- | --- | --- | --- | --- |
| NFR-P01 | Encoder backend latency, GPU (RTX 3060, batch=1) | p50 ≤ 20 ms, p95 ≤ 50 ms (met: stock 9.1/10.8 ms; fast 3.7/4.1 ms) | M3 | `benchmarks/` latency harness |
| NFR-P02 | Encoder backend latency, CPU (12 cores) | p50 ≤ 200 ms **[open]** | M3 | latency harness |
| NFR-P03 | Router decision overhead | < 0.5 ms | M3 | micro-benchmark |
| NFR-P04 | LLM backend overhead beyond provider call | < 50 ms | M2 | timing measurement |
| NFR-P05 | Batch throughput | near-linear scaling up to memory limit | M3 | batch sweep |
| NFR-P06 | Server cold start (encoder preloaded) | ≤ 30 s **[open]** | M3 | timed startup |
| NFR-P07 | Fast-path (TileLang/CUDA graphs) latency on supported CUDA | p50 improves vs stock forward; no shape change (met: 2.47× p50, 0 top-label flips) | Backlog B-2 | micro-benchmark |

## Resource footprint (NFR-R)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-R01 | Base install importable without torch/transformers | required | M0 |
| NFR-R02 | Encoder peak RSS, single English checkpoint | ≤ 4 GB **[open]** | M3 |
| NFR-R03 | Multilingual checkpoint load + inference within 12 GB VRAM | required | M3 |
| NFR-R04 | Two checkpoints loaded (preload) fit together **[open]** | required | M3 |
| NFR-R05 | Training fits RTX 3060 12 GB (LoRA/QLoRA) | required | M4 |

## Reliability & correctness (NFR-C)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-C01 | Jev wire parity | 100% golden fixtures | M1 |
| NFR-C02 | Probability normalization | sums to 1.0 ± 1e-6 | M1 |
| NFR-C03 | Deterministic inference on CPU given fixed weights + seed | required | M3 |
| NFR-C04 | Reproducible training given seed/config | within documented tolerance | M4 |
| NFR-C05 | Graceful fallback when acceleration/extra unavailable | no crash | M5 |
| NFR-C06 | Calibration quality | ECE ≤ 0.05 **[near: overall multi 0.038, en 0.061; `nl` 0.104 open]** | M4 |
| NFR-C07 | Per-language calibration quality | ECE ≤ 0.05 per language (choice/score) **[5/6 languages met; `nl` 0.104 open]** | Backlog B-1 |

## Compatibility (NFR-X)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-X01 | Existing Jev clients work unchanged | verified with real client | M2 |
| NFR-X02 | Extensions are additive; canonical shape never changes | contract test | M2–M5 |
| NFR-X03 | Python version support | `>=3.12,<3.13` | M0 |
| NFR-X04 | Platforms | Linux (primary), macOS CPU, Windows CPU **[open]** | M3+ |
| NFR-X05 | Devices | CUDA + CPU | M3 |

## Security & privacy (NFR-S)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-S01 | No secrets committed to the repo | enforced | M0 |
| NFR-S02 | API key read from env, never logged | enforced | M2 |
| NFR-S03 | Base install performs no network egress | enforced | M3 |
| NFR-S04 | Offline operation with no key | required | M3 |
| NFR-S05 | Telemetry opt-out (`DO_NOT_TRACK=1`) if present | enforced | M5 |
| NFR-S06 | Constant-time/naive key comparison documented (dev vs prod) | documented | M2 |

## Maintainability & quality (NFR-M)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-M01 | Lint clean with ruff | 0 errors | M0 |
| NFR-M02 | Type clean with pyright | 0 errors (strict-ish config) | M0 |
| NFR-M03 | Contract test updated with any public API change | enforced by review/CI | M0+ |
| NFR-M04 | Conventional commits; one logical change per commit | enforced by review | M0+ |
| NFR-M05 | Each backend passes the shared contract suite | 100% | M2+ |
| NFR-M06 | Extras isolate optional deps (import guards) | enforced | M5 |

## Observability (NFR-O)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-O01 | `/health` endpoint reports backend/device/checkpoints | required | M2 |
| NFR-O02 | Structured logging of requests (no payload secrets) | required | M2 |
| NFR-O03 | Hook-based metrics (latency, routing, errors) | required | M3 |

## Documentation & licensing (NFR-D)

| ID | Requirement | Target | Phase |
| --- | --- | --- | --- |
| NFR-D01 | README with vision, quickstart, comparison | required | M0 |
| NFR-D02 | Architecture, protocol, testing, training docs maintained | required | M0+ |
| NFR-D03 | Apache-2.0 license | required | M0 |
| NFR-D04 | Reproducible benchmark report and model card | required | M6 |

## Accessibility (NFR-A)

Not applicable — jeba has no GUI (CLI + HTTP + SDK). A future web console would introduce
accessibility requirements (see the `Accessibility Testing` skill) but is out of scope.

---

**Counts:** 41 non-functional requirements (39 baseline + NFR-P07/NFR-C07 tracked in
`.specs/project/BACKLOG.md`). Targets marked **[open]** are tracked in
`.specs/project/STATE.md` and resolved after M3/M4 measurements.
