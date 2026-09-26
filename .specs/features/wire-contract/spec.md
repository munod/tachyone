# Wire Contract & Primitives Specification

**Phase:** M0–M1 (Phase 0–1)
**Status:** Implemented (M1 complete; contract frozen)
**Related docs:** `docs/protocol.md`, `docs/architecture.md`, `docs/adr/ADR-0001-jev-drop-in-protocol.md`

## Problem Statement

Tachyone must be a drop-in replacement for the hosted TypeSafe Jev `/v1/systemone` API. Before
any model or server exists, we need a frozen, testable data contract — primitives, request,
response, and error shapes — so that backends, SDK, and server can be built against one
authoritative definition. Getting this wrong causes cascading failures downstream.

## Goals

- [x] Freeze `primitives.py` and `wire.py` to exact Jev field parity.
- [x] Prove parity with golden contract tests (request → response → error shapes).
- [x] Provide a deterministic fake backend so the full cycle is testable offline.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Real model inference | That is `llm-backend` / `encoder-backend` |
| HTTP transport | `serve.py` lands in M2; M1 validates Python-level wire only |
| Auth infrastructure | Only the 401 shape is contractually validated here |

---

## User Stories

### P1: Canonical primitives ⭐ MVP

**User Story:** As a Tachyone developer, I want `choice`/`score`/`noul` as validated pydantic
models so that every backend shares one definition of a question and its answer.

**Why P1:** Everything (wire, server, training) depends on these types.

**Acceptance Criteria:**

1. WHEN a `noul` question is defined with optional `{true, false}` criteria THEN the system
   SHALL accept it and produce `{type: "noul", noul: <float 0..1>}`.
2. WHEN a `choice` question is defined THEN the system SHALL accept `criteria` as a map of
   `option -> description|null` and SHALL reject more than 255 options.
3. WHEN a `score` question is defined THEN the system SHALL accept an ordered array of 2 to
   10 levels and SHALL reject fewer than 2 or more than 10.
4. WHEN an answer is produced for `choice`/`score` THEN `probabilities` keys SHALL exactly
   match the declared options/levels.
5. WHEN a `choice`/`score` answer is produced THEN it SHALL include a `confidence` derived
   from its probability distribution; `noul` SHALL NOT carry a separate confidence field.

**Independent Test:** Instantiate each primitive with valid and invalid payloads and assert
pydantic validation + serialized shape.

### P1: Jev-exact request/response ⭐ MVP

**User Story:** As an integrator with an existing Jev client, I want the same request and
response JSON so I can repoint my client without code changes.

**Why P1:** This is the product's core promise.

**Acceptance Criteria:**

1. WHEN a valid request is received THEN the system SHALL accept
   `{ "state": string|object|array, "model": string, "questions": map<string, Question> }`.
2. WHEN a valid request is answered THEN the system SHALL return
   `{ "model": string, "answers": map<string, Answer>, "usage": {"input_tokens": int, "output_tokens": int} }`.
3. WHEN responses are compared to golden Jev fixtures THEN field names, types, and nesting
   SHALL match exactly (no extra required fields, no renamed fields).
4. WHEN multiple questions are sent in one request THEN each SHALL be evaluated
   independently and keyed by its original question id.

**Independent Test:** Load golden fixtures, run through the fake backend, diff serialized JSON.

### P2: Error contract

**User Story:** As an operator, I want the documented error statuses so my client's retry
logic behaves as it does against Jev.

**Why P2:** Needed for production robustness but not for the first happy-path demo.

**Acceptance Criteria:**

1. WHEN the API key is missing/invalid THEN the system SHALL return `401 Unauthorized`.
2. WHEN the request body is malformed THEN the system SHALL return `422 Unprocessable Entity`.
3. WHEN rate-limited THEN the system SHALL return `429 Too Many Requests`.
4. WHEN overloaded THEN the system SHALL return `529` and clients SHALL retry with
   exponential backoff.

**Independent Test:** Fake backend emits each error; assert status + body shape.

### P2: Bootstrap & quality gates

**User Story:** As a maintainer, I want a pinned, linted, typed project so changes are safe.

**Acceptance Criteria:**

1. WHEN `uv sync` runs THEN it SHALL install Python 3.12 and dev tools from a locked graph.
2. WHEN `uv run ruff check . && uv run pyright && uv run pytest` runs THEN all SHALL pass.
3. WHEN a commit is pushed THEN CI SHALL run the same three gates.

---

## Edge Cases

- WHEN an empty `questions` map is sent THEN the system SHALL return an empty `answers` map
  (or 422 — decision tracked as OD-5/open; default to empty answers).
- WHEN `state` is a large object/array THEN it SHALL be accepted (no size contract beyond transport limits).
- WHEN two question ids collide in a map THEN pydantic SHALL reject the payload (JSON maps cannot collide).
- WHEN `choice` has exactly 1 or 256 options THEN the system SHALL return 422.
- WHEN `score` has 1 or 11 levels THEN the system SHALL return 422.
- WHEN `noul` criteria omits one or both branches THEN the system SHALL accept it.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| PRIM-01 | P1 primitives | Design | Done |
| PRIM-02 | P1 primitives | Design | Done |
| PRIM-03 | P1 primitives | Design | Done |
| PRIM-04 | P1 primitives | Design | Done |
| PRIM-05 | P1 primitives | Design | Pending (M3 confidence derivation) |
| PRIM-06 | P1 primitives | Design | Done |
| WIRE-01 | P1 request/response | Design | Pending (M2 transport) |
| WIRE-02 | P1 request/response | Design | Done |
| WIRE-03 | P1 request/response | Design | Done |
| WIRE-04 | P2 errors | Design | Done (status shapes; transport in M2) |
| WIRE-05 | P2 errors | - | Pending (M2 SDK retry) |
| WIRE-06 | P1 request/response | Design | Done |
| WIRE-07 | P1 request/response | Design | Done |

**Coverage:** 13 requirements, all mapped to `docs/tasks.md` M0–M1.

## Success Criteria

- [ ] Golden contract test suite passes for happy paths and all documented errors.
- [ ] No field drift between Tachyone serialization and captured Jev fixtures.
- [ ] `curl` (or Python) round-trip against the fake backend returns a well-formed response.
