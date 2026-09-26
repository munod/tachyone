# Functional Requirements

**Status:** Baseline. IDs are stable and traceable.
**Priority:** P1 = MVP, P2 = should-have, P3 = nice-to-have.
**Traceability:** See `docs/requirements/traceability.md`. Task IDs refer to `docs/tasks.md`.

Conventions: `WIRE` protocol · `PRIM` primitives · `BACK` backends · `ROUTE` routing ·
`CAL` calibration · `EXT` extensions · `SERVE` serving/SDK/CLI · `TRAIN` training · `OPS` ops.

---

## Protocol & wire (WIRE)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| WIRE-01 | Accept `POST /v1/systemone` with `Authorization: Bearer <key>` and `Content-Type: application/json`. | P1 | M2 | Server routes the canonical path; auth enforced when a key is configured. |
| WIRE-02 | Accept request `{ state, model, questions }` where `state` is string/object/array and `questions` is `map<string, Question>`. | P1 | M1 | pydantic validates; invalid body → 422. |
| WIRE-03 | Return `{ model, answers, usage{input_tokens, output_tokens} }`. | P1 | M1 | Response serializes with exact field names/nesting. |
| WIRE-04 | Return documented error statuses 401, 422, 429, 529 with faithful semantics. | P1 | M2 | Fault injection yields each status. |
| WIRE-05 | Client SDK retries 429/529 with exponential backoff + jitter. | P2 | M2 | Retry count/backoff verified. |
| WIRE-06 | Maintain exact Jev field-name/type parity verified against golden fixtures. | P1 | M1 | Fixture diff is empty. |
| WIRE-07 | Report token `usage` on every response. | P1 | M1 | `usage` present and integer-typed. |

## Primitives (PRIM)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| PRIM-01 | Support `noul` with optional `criteria.{true,false}`; answer `{type:"noul", noul: 0..1}`. | P1 | M1 | Valid/invalid payloads; answer shape. |
| PRIM-02 | Support `choice` with `criteria: map<option, desc\|null>`, max **255** options; answer `{type, choice, probabilities, confidence}`. | P1 | M1 | 255 ok; 256 → 422. |
| PRIM-03 | Support `score` with ordered `criteria` array of **2..10** levels; answer `{type, score, legend, probabilities, confidence}`. | P1 | M1 | 2 and 10 ok; 1/11 → 422. |
| PRIM-04 | `probabilities` keys exactly equal declared options/levels; values in [0,1]. | P1 | M1 | Key-coverage validator. |
| PRIM-05 | Derive `confidence` from the distribution for `choice`/`score`; `noul` has no separate confidence. | P1 | M3 | Confidence ∈ [0,1]; noul shape has no confidence. |
| PRIM-06 | Evaluate multiple questions per request independently and key answers by question id. | P1 | M1 | N in → N out, ids echoed. |

## Backends (BACK)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| BACK-01 | Provide a stable `Backend` interface decoupled from the wire. | P1 | M1 | Protocol defined; wire imports only the seam. |
| BACK-02 | Provide an LLM backend using structured outputs of existing LLMs. | P1 | M2 | Answers all primitives via a configured provider. |
| BACK-03 | Provide a local encoder backend: single forward pass, three heads. | P1 | M3 | All primitives answered in one pass. |
| BACK-04 | Provide an ONNX runtime backend. | P2 | M5 | Contract parity with encoder backend. |
| BACK-05 | Select backend via configuration/env without changing the wire. | P1 | M2 | Same fixture, same shape across backends. |
| BACK-06 | Default local backend works fully offline with no API key. | P1 | M3 | Network-disabled run succeeds. |
| BACK-07 | Support batch prediction (`predict_batch`) with length-sorted batching. | P2 | M3 | Aligned results for a list of states. |

## Routing & multilingual (ROUTE)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| ROUTE-01 | Detect input script/language without a model forward pass. | P1 | M3 | Correct detection on multi-script samples. |
| ROUTE-02 | Select English vs multilingual checkpoint automatically. | P1 | M3 | Expected checkpoint chosen per sample. |
| ROUTE-03 | Routing overhead < ~0.5 ms per decision. | P2 | M3 | Measured benchmark. |
| ROUTE-04 | Manage checkpoint lifecycle: `preload`, `max_loaded`, LRU evict, `unload`, `attach`. | P2 | M3 | No leak; correct eviction under `max_loaded=1`. |
| ROUTE-05 | Support 100+ languages via the multilingual checkpoint. | P1 | M3 | Held-out multilingual suite passes. |

## Calibration & confidence (CAL)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| CAL-01 | Train probabilities with RLCD under a strictly proper scoring rule. | P1 | M4 | Loss is proper scoring; training completes. |
| CAL-02 | Fit temperature to minimize ECE on a held-out calibration split. | P1 | M4 | ECE reduced vs uncalibrated baseline. |
| CAL-03 | Report `confidence` for `choice`/`score` derived from probabilities. | P1 | M3 | Value in [0,1], monotone with concentration. |
| CAL-04 | Probabilities are calibrated to the documented ECE target. | P1 | M4 | ECE ≤ target on held-out set. |
| CAL-05 | Full distributions are always present in the canonical answer; `return_details` is accepted by the backend/schema surface for API stability (additive, currently a no-op). | P2 | M3 | Details returned without changing canonical fields. |
| CAL-06 | Expose client-side uncertainty (entropy, margin) and a thresholded abstain/handoff signal over the returned probabilities. | P2 | Post-M6 | Helpers in `handoff.py`; no wire change; covered by unit tests. |

## Extensions (EXT)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| EXT-01 | Provide hooks: `on_predict_start/end`, `on_route`, `on_load`, `on_evict`, `on_error`. | P2 | M3 | Each hook fires on its event. |
| EXT-02 | Hooks never alter the canonical `/v1/systemone` response shape. | P1 | M3 | Contract test passes with hooks registered. |
| EXT-03 | Allow router override (force checkpoint/language) additively. | P2 | M5 | Override honored; canonical shape unchanged. |
| EXT-04 | Provide `predict_batch` extension endpoint. | P2 | M5 | Batch request/response works. |
| EXT-05 | Python SDK mirrors the wire and exposes extension ergonomics. | P1 | M2 | SDK round-trip matches wire. |

## Serving, SDK, CLI (SERVE)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| SERVE-01 | FastAPI server implementing `/v1/systemone`. | P1 | M2 | Endpoint live and contract-compliant. |
| SERVE-02 | Extension endpoints `/predict`, `/predict/batch`, `/health`. | P2 | M2 | Each responds correctly. |
| SERVE-03 | CLI `jeba "text" --preset triage --predict`. | P1 | M2 | Prints primitive answer. |
| SERVE-04 | Entry points `jeba`, `jeba-serve`, `jeba-mcp-server`. | P1 | M2/M5 | Console scripts resolve. |
| SERVE-05 | Python SDK matching the wire. | P1 | M2 | Client call round-trips. |
| SERVE-06 | Configuration via env vars (`HOST/PORT/DEVICE/PRELOAD/MODELS/THREADS/API_KEY/BACKEND`). | P1 | M2 | Env changes take effect at startup. |
| SERVE-07 | Presets `router`, `guard`, `moderation`, `triage`, `email`. | P2 | M2 | Each expands to canonical questions. |
| SERVE-08 | `decide(schema=...)` from JSON Schema or pydantic → decision primitives. | P2 | M2 | Schema produces valid questions. |

## Training (TRAIN)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| TRAIN-01 | Deterministic synthetic data generation to JSONL. | P1 | M4 | Same seed → identical bytes. |
| TRAIN-02 | LoRA/QLoRA fine-tuning within 12GB VRAM. | P1 | M4 | Training completes without OOM. |
| TRAIN-03 | RLCD proper-scoring training loop. | P1 | M4 | Proper-scoring loss implemented. |
| TRAIN-04 | Temperature/calibration fitting script. | P1 | M4 | ECE reduced on calibration split. |
| TRAIN-05 | Evaluation harness reporting accuracy, ECE, latency per primitive/language. | P1 | M4 | Artifacts saved and reproducible. |
| TRAIN-06 | Reproducible scripts + configs (seed, hyperparams) for every stage. | P1 | M4 | Re-run within documented tolerance. |
| TRAIN-07 | Stream large datasets to disk during generation. | P1 | M4 | Memory bounded. |
| TRAIN-08 | Seeded, opt-in input-noise augmentation (typos/accents/casing) with a clean vs noisy evaluation split. | P2 | Post-M6 | Same seed → identical bytes; noisy view reported separately. |

## Ecosystem & ops (OPS)

| ID | Requirement | Prio | Phase | Acceptance summary |
| --- | --- | --- | --- | --- |
| OPS-01 | pip extras `serve`, `fast`, `onnx`, `langchain`, `mcp`, `train` isolate dependencies. | P1 | M5 | Installing an extra only adds its deps. |
| OPS-02 | MCP stdio server (`jeba-mcp-server`). | P2 | M5 | Tools callable from an MCP host. |
| OPS-03 | LangChain `Runnable` adapter. | P2 | M5 | Returns canonical primitives. |
| OPS-04 | Docker + compose deployment. | P2 | M5 | Server reachable and contract-compliant. |
| OPS-05 | CI gates: ruff, pyright, pytest on push/PR. | P1 | M0 | CI green. |
| OPS-06 | Reproducible benchmarks (MASSIVE/XNLI/typed-decisions). | P2 | M6 | Report reproducible from commands. |
| OPS-07 | Hugging Face release + model card + GitHub release. | P2 | M6 | Artifacts published. |
| OPS-08 | Telemetry (if any) is opt-out and never required. | P2 | M5 | `DO_NOT_TRACK=1` disables; offline works. |

---

**Counts:** 60 functional requirements (P1: 40 · P2: 20 · P3: 0).
Every requirement maps to a task in `docs/tasks.md` (0 unmapped).
