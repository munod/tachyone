# Local Encoder Backend Specification

**Phase:** M3 (Phase 3)
**Status:** Implemented (M3 complete; trained/calibrated heads land in M4)
**Related docs:** `docs/architecture.md`, `docs/training.md`, `docs/adr/ADR-0005-encoder-rlcd.md`

## Problem Statement

The LLM backend proves the contract but depends on an external model and inference speed.
jeba's differentiator is a local, non-autoregressive encoder that answers all three
primitives in a single forward pass, offline, in 100+ languages, with calibrated confidence.

## Goals

- [x] Answer all three primitives in one forward pass with ModernBERT/mmBERT + 3 heads.
- [x] Automatically route inputs to the right checkpoint by script/language.
- [x] Emit calibrated `confidence`/probabilities (no network, no API key).
- [x] Support batching and lifecycle management (preload/evict/unload, max_loaded).

## Out of Scope

| Feature | Reason |
| --- | --- |
| Training the encoder | `training-calibration` (M4) |
| ONNX / fast path | `ecosystem` (M5) |
| HTTP layer | Reused from `llm-backend` (M2) |

---

## User Stories

### P1: Single-pass local inference ⭐ MVP

**User Story:** As an edge developer, I want local answers with no network so I can run in
air-gapped environments.

**Acceptance Criteria:**

1. WHEN a request is processed by the encoder backend THEN it SHALL perform a single forward
   pass per batch and return answers for all three primitive types.
2. WHEN no API key is present and no network is available THEN the encoder backend SHALL still
   answer successfully.
3. WHEN `predict_batch` is called with a list of states THEN the backend SHALL batch, sort by
   length, and return aligned results.

**Independent Test:** Run offline with network disabled; assert answers and shape match the contract.

### P1: Language/script router ⭐ MVP

**User Story:** As a multilingual user, I want the right checkpoint chosen automatically.

**Acceptance Criteria:**

1. WHEN input script/language is detected THEN the router SHALL select English vs multilingual checkpoint.
2. WHEN routing THEN overhead SHALL be under ~0.5 ms per decision (no model inference).
3. WHEN a requested checkpoint is not loaded and `max_loaded` is reached THEN the router SHALL
   evict the least-recently-used checkpoint.
4. WHEN `preload` is configured THEN listed checkpoints SHALL be loaded at startup.

**Independent Test:** Feed samples in Latin/Cyrillic/CJK/Arabic scripts; assert checkpoint IDs chosen.

### P1: Calibrated confidence ⭐ MVP

**User Story:** As a decision consumer, I want confidence I can trust.

**Acceptance Criteria:**

1. WHEN a `choice`/`score` answer is produced THEN `confidence` SHALL be derived from the
   probability distribution and lies in [0,1].
2. WHEN `return_details=True` THEN the response SHALL include full probabilities (extension, additive).
3. WHEN calibration is fitted THEN reported ECE on a held-out set SHALL meet the target in
   `docs/training.md`.

**Independent Test:** Compare confidence vs empirical accuracy bins; compute ECE.

### P2: Lifecycle & observability hooks

**User Story:** As an operator, I want hooks around predictions and model lifecycle.

**Acceptance Criteria:**

1. WHEN hooks are registered (`on_predict_start`, `on_predict_end`, `on_route`, `on_load`,
   `on_evict`, `on_error`) THEN they SHALL fire at the corresponding events.
2. WHEN a hook raises THEN the system SHALL invoke `on_error` and continue serving (hooks must
   not corrupt the canonical response).
3. WHEN a hook mutates payloads THEN the canonical `/v1/systemone` response shape SHALL remain unchanged.

---

## Edge Cases

- WHEN input exceeds checkpoint context THEN the backend SHALL truncate per documented policy.
- WHEN `max_loaded=1` and a second language arrives THEN eviction SHALL not leak memory.
- WHEN only one option exists for a choice THEN confidence SHALL be well-defined (distribution degenerate).
- WHEN probabilities are near-ties THEN confidence SHALL reflect low certainty.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| BACK-03 | P1 single-pass | Design | Done |
| BACK-06 | P1 single-pass | Design | Done |
| BACK-07 | P1 single-pass | Design | Done |
| ROUTE-01 | P1 router | Design | Done |
| ROUTE-02 | P1 router | Design | Done |
| ROUTE-03 | P1 router | Design | Done |
| ROUTE-04 | P1 router | Design | Done |
| ROUTE-05 | P1 router | Design | Done |
| CAL-03 | P1 confidence | Design | Done |
| CAL-04 | P1 confidence | Design | Pending (M4 ECE target) |
| CAL-05 | P1 confidence | Design | Done |
| EXT-01 | P2 hooks | Design | Done |
| EXT-02 | P2 hooks | Design | Done |

**Coverage:** 13 requirements mapped to `docs/tasks.md` M3.

## Success Criteria

- [x] Offline single-pass answers for all primitives across multiple languages.
- [x] Router selects correct checkpoint with documented overhead.
- [ ] ECE target met on held-out calibration set (M4).
