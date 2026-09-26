# LLM Backend & Serving Specification

**Phase:** M2 (Phase 2)
**Status:** Implemented (M2 complete)
**Related docs:** `docs/architecture.md`, `docs/protocol.md`, `docs/testing.md`

## Problem Statement

We need an end-to-end jeba system quickly, before any encoder training. Existing LLMs can
already answer structured questions. By wrapping them behind the `Backend` seam and serving
the Jev contract, we get a usable, verifiable product that existing Jev clients can hit.

## Goals

- [x] Ship a `POST /v1/systemone` server backed by structured outputs of existing LLMs.
- [x] Provide a Python SDK and CLI that speak the same contract.
- [x] Prove a real Jev client can be repointed at jeba unchanged.
- [x] Keep the LLM backend optional so the base install stays offline/no-key.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Local encoder accuracy | `encoder-backend` (M3) |
| Model training | `training-calibration` (M4) |
| ONNX/MCP/LangChain | `ecosystem` (M5) |

---

## User Stories

### P1: Drop-in server ⭐ MVP

**User Story:** As an integrator, I want to point my existing Jev client at jeba's base URL
and have it work, so I can migrate without a rewrite.

**Acceptance Criteria:**

1. WHEN `POST /v1/systemone` is called with a valid Bearer token and body THEN the server
   SHALL return the Jev response shape defined in `docs/protocol.md`.
2. WHEN no/invalid token is supplied THEN the server SHALL return 401.
3. WHEN the body is invalid THEN the server SHALL return 422 with a validation payload.
4. WHEN the backend is rate-limited or overloaded THEN the server SHALL surface 429/529 and
   the SDK SHALL retry with exponential backoff.

**Independent Test:** Repoint a captured Jev client fixture at the local server; assert equality
of response shape and primitive values.

### P1: Stable pluggable backend ⭐ MVP

**User Story:** As a developer, I want to swap LLM providers without touching the wire, so I
can use my preferred model or a local one.

**Acceptance Criteria:**

1. WHEN the backend is selected via config/env THEN the wire response SHALL be identical in shape.
2. WHEN structured output fails to parse THEN the backend SHALL retry (bounded) and, if still
   failing, raise a backend error mapped to a non-contract 500.
3. WHEN a request has N questions THEN the backend SHALL evaluate them independently and return
   N answers keyed by id.

**Independent Test:** Run the same fixture against two configured providers; assert shape parity.

### P2: Python SDK & CLI

**User Story:** As a Python user, I want `jeba` and a small client so I can answer questions
locally and in scripts.

**Acceptance Criteria:**

1. WHEN `jeba "text" --preset triage --predict` runs THEN the CLI SHALL print the primitive answer.
2. WHEN the SDK client method is called THEN it SHALL POST the exact wire request and parse the response.
3. WHEN `jeba-serve` runs THEN it SHALL start the HTTP server with env-configurable host/port/device.

**Independent Test:** CLI invocation against a local server returns a well-formed answer.

### P2: Presets and `decide()`

**User Story:** As an app builder, I want reusable presets and schema-driven questions.

**Acceptance Criteria:**

1. WHEN a preset is requested (`router`/`guard`/`moderation`/`triage`/`email`) THEN the system
   SHALL expand it into canonical questions.
2. WHEN a JSON Schema or pydantic model is passed to `decide(schema=...)` THEN the system SHALL
   produce decision primitives (`return_details` optional).

---

## Edge Cases

- WHEN an LLM returns extra prose around JSON THEN the backend SHALL extract and validate JSON.
- WHEN probabilities do not sum to 1 THEN the backend SHALL normalize before returning.
- WHEN `state` is an object/array THEN it SHALL be serialized into the prompt without lossy coercion.
- WHEN the API key env var is absent THEN the server SHALL start but reject protected calls with 401.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| BACK-01 | P1 stable backend | Design | Done |
| BACK-02 | P1 drop-in server | Design | Done |
| BACK-05 | P1 stable backend | Design | Done |
| SERVE-01 | P1 drop-in server | Design | Done |
| SERVE-03 | P2 SDK/CLI | Design | Done |
| SERVE-04 | P2 SDK/CLI | Design | Done |
| SERVE-05 | P2 SDK/CLI | Design | Done |
| SERVE-06 | P2 SDK/CLI | Design | Done |
| SERVE-07 | P2 presets | Design | Done |
| SERVE-08 | P2 presets | Design | Done |
| WIRE-05 | P1 drop-in server | Design | Done |

**Coverage:** 11 requirements mapped to `docs/tasks.md` M2. All implemented (M2 complete; confidence derivation continues in M3).

## Success Criteria

- [x] Repointed Jev client returns a correct answer for all three primitives.
- [x] 429/529 retry behavior observed with a fault-injected provider.
- [x] Base install (no `serve`/LLM extra) imports and runs offline.
