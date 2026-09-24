# ADR-0012: Ship the ONNX backend before the TileLang fast path

**Status:** Accepted
**Date:** 2026-09-24

## Context

M5 adds an ONNX runtime backend and an optional TileLang/CUDA-graph fast path. Both touch the
local encoder but with different value and risk: ONNX is portable (CPU and CUDA via
`onnxruntime`), ships smaller payloads, and lets the same contract tests run against a second
runtime; the TileLang path is a CUDA-only latency optimization with kernel/CUDA-graph
compatibility risk. OD-4 asked for sequencing.

## Decision

Implement `backends/onnx.py` first and gate it with the same contract suite as the encoder. Add
the TileLang fast path afterwards behind the `fast` extra, with a graceful fallback to the
stock forward when the extra or a supported device is unavailable. Neither path may change the
canonical response shape (EXT-02); a router override (force checkpoint/language) is an additive
extension (EXT-03).

## Consequences

**Positive:** Portability and a second contract-verified runtime early; acceleration is opt-in
and cannot regress the default path.
**Negative:** Latency gains land later.
**Neutral / follow-ups:** Benchmarks (`NFR-P01`) measure the fast path once it exists.

## Alternatives Considered

- **Fast path first** — quicker latency wins but CUDA-only, fragile, and no portability.
- **Both in parallel** — larger simultaneous test/CI surface for little schedule benefit.

## Related

- `docs/architecture.md`, ADR-0002 (pluggable backend), BACK-04, OPS-01, EXT-03
