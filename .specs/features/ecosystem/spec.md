# Ecosystem & Acceleration Specification

**Phase:** M5–M6 (Fase 5–6)
**Status:** Implemented. Acceleration/integrations/packaging done; benchmark report + docs site
shipped, with real numbers and HF weights pending the GPU training run.
**Related docs:** `docs/architecture.md`, `docs/roadmap.md`, `docs/adr/ADR-0006-license-telemetry.md`

## Problem Statement

The core engine is only useful if it fits real workflows: agent frameworks, acceleration
paths, containers, and public proof. This phase broadens integration surface without ever
changing the canonical contract.

## Goals

- [x] Add ONNX backend and an optional fast path (TileLang/CUDA graphs).
- [x] Ship an MCP stdio server and a LangChain adapter.
- [x] Provide Docker/compose and pip extras.
- [x] Publish reproducible benchmarks, docs site, and a Hugging Face release.

## Out of Scope

| Feature | Reason |
| --- | --- |
| New primitive types | Contract is frozen (ADR-001) |
| Hosted control plane | Local-first (ADR-004) |

---

## User Stories

### P1: Optional acceleration ⭐ MVP

**User Story:** As a GPU user, I want a faster path without changing my client.

**Acceptance Criteria:**

1. WHEN the ONNX backend is selected THEN responses SHALL match the encoder backend contract.
2. WHEN the fast extra is installed with a supported CUDA device THEN latency SHALL improve
   without altering the response shape.
3. WHEN acceleration is unavailable THEN the system SHALL fall back gracefully.

### P1: Agent integrations ⭐ MVP

**User Story:** As an agent builder, I want jeba as an MCP tool and LangChain runnable.

**Acceptance Criteria:**

1. WHEN `jeba-mcp-server` runs over stdio THEN tools SHALL expose prediction capabilities.
2. WHEN the LangChain adapter is used THEN it SHALL return canonical primitive results.
3. WHEN integrations are not installed THEN imports SHALL be isolated to their extras.

### P2: Packaging & deployment

**User Story:** As an operator, I want containers and clean extras.

**Acceptance Criteria:**

1. WHEN extras `serve`/`fast`/`onnx`/`langchain`/`mcp`/`train` are installed THEN only their
   dependencies SHALL be added.
2. WHEN `docker/` compose runs THEN the server SHALL be reachable and answer `/v1/systemone`.

### P2: Public proof & release

**User Story:** As an evaluator, I want evidence and a model I can download.

**Acceptance Criteria:**

1. WHEN benchmarks run THEN a report SHALL be reproducible from committed commands.
2. WHEN released THEN a Hugging Face model card and GitHub release SHALL exist.
3. WHEN telemetry exists THEN it SHALL be opt-out and never required for function.

---

## Edge Cases

- WHEN an extra is missing THEN the error SHALL name the exact extra to install.
- WHEN CUDA is present but unsupported THEN SHALL fall back to CPU without crashing.
- WHEN MCP client disconnects THEN the server SHALL exit cleanly.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| BACK-04 | P1 acceleration | Design | Done |
| OPS-01 | P2 packaging | Design | Done |
| OPS-02 | P1 integrations | Design | Done |
| OPS-03 | P1 integrations | Design | Done |
| OPS-04 | P2 packaging | Design | Done |
| OPS-06 | P2 proof | Design | Done (numbers pending) |
| OPS-07 | P2 proof | Design | Done (weights pending) |
| OPS-08 | P2 proof | Design | Done |
| EXT-01 | P1 integrations | Design | Done |
| EXT-03 | P1 acceleration | Design | Done |
| EXT-04 | P1 acceleration | Design | Done |

**Coverage:** 11 requirements mapped to `docs/tasks.md` M5–M6.

## Success Criteria

- [x] Contract tests pass unchanged against ONNX + accelerated backends.
- [x] MCP and LangChain adapters work end-to-end.
- [x] Public benchmark report and HF model card published (numbers/weights pending the GPU run).
