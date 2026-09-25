# Fast Path (TileLang / CUDA graphs) Specification

**Phase:** Post-M6 (B-2, `.specs/project/BACKLOG.md`)
**Status:** Implemented. `maybe_accelerate` is wired via `JEBA_FAST` to per-shape CUDA graphs
(bf16 weights) with graceful fallback; measured 2.47× p50, 0 top-label flips (NFR-P01 met).
TileLang fused kernels remain optional (no measured benefit yet).
**Related docs:** `docs/adr/ADR-0012-onnx-before-fast-path.md`, `docs/tasks.md` M5-T2,
`docs/requirements/non-functional.md` (NFR-P01), `src/jeba/fast.py`.

## Problem Statement

M5 shipped the fast-path **seam and graceful fallback** only: `maybe_accelerate` is never called and
always returns the stock target with reason "no accelerated kernels registered". There is no measured
latency improvement and `NFR-P01` (p50 ≤ 20 ms, p95 ≤ 50 ms on an RTX 3060, batch=1) is still
`[open]`. The seam is dead code.

## Goals

- [x] Wire the fast-path seam into the encoder so it can actually accelerate the forward.
- [x] Ship a real, low-risk speedup for batch=1: CUDA-graph capture with bf16-resident weights
      (per batch bucket), optional `torch.compile`.
- [x] Keep TileLang fused kernels optional and only where measured to help.
- [x] Measure stock vs fast (p50/p95, throughput, parity) and record it, closing NFR-P01.
- [x] Preserve graceful fallback on CPU/MPS/absent extras and never change the response shape.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Full fused kernel suite by default | High risk/effort; pragmatic path first (user decision) |
| ONNX graph optimization | Separate backend (BACK-04) |
| Multi-GPU / distributed | Single 12GB target |

---

## User Stories

### P1: Wired, opt-in acceleration ⭐ MVP

**User Story:** As a GPU user, I want a faster path without changing my client.

**Acceptance Criteria:**

1. WHEN the accelerated path is enabled and CUDA is available THEN the encoder SHALL use the
   accelerated forward; otherwise it SHALL use the stock forward unchanged (NFR-C05, OPS-01).
2. WHEN acceleration is applied THEN the response shape SHALL be identical (EXT-02); probability
   parity within documented tolerance.
3. WHEN the extra/device is missing THEN `maybe_accelerate` SHALL return the stock target with a
   reason and never raise.
4. WHEN a builder is injected THEN the seam SHALL use it only under the documented conditions.

### P1: Measured proof

**User Story:** As an evaluator, I want reproducible latency evidence.

**Acceptance Criteria:**

1. WHEN `benchmarks/fast_path.py` runs on a supported device THEN it SHALL report stock vs fast
   p50/p95, throughput vs batch, and a parity delta.
2. WHEN only a CPU is available THEN the benchmark SHALL skip cleanly (exit non-zero with a clear
   message or be marked conditional).

---

## Edge Cases

- WHEN CUDA-graph capture fails THEN fall back to the stock forward and record the reason.
- WHEN input shapes vary THEN bucket by `(batch, length)` rather than re-capturing per call.
- WHEN TileLang is absent THEN the CUDA-graph path SHALL still work (torch only).

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| NFR-P01 | P1 proof | Design | Done |
| NFR-C05 | P1 fallback | Design | Done |
| OPS-01 | P1 packaging | Design | Done |
| EXT-03 | P1 acceleration | Design | Done (seam) |

**Coverage:** 4 requirements mapped to `.specs/features/fast-path/tasks.md`.

## Success Criteria

- [x] On a supported CUDA device, p50 latency improves vs stock with no response-shape change.
- [x] Graceful fallback on CPU/MPS and when the extra is absent (contract tests unchanged).
- [x] `NFR-P01` measured and the micro-benchmark committed.
