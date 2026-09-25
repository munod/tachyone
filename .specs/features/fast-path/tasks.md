# Fast Path (TileLang / CUDA graphs) — Tasks

**Spec:** `.specs/features/fast-path/spec.md`
**Status:** In progress

---

## B1: Wire the seam + CUDA-graph accelerated encode

**What:** Generalize `maybe_accelerate(target, *, device, builder=None)` to invoke an injected
accelerator builder when CUDA is present and otherwise return the stock target with a reason. Build a
CUDA-graph-captured encode per `(batch, length)` bucket with bf16-resident weights behind a
`JEBA_FAST` config flag in the encoder loader. TileLang fused kernels stay optional.
**Where:** `src/jeba/fast.py`, `src/jeba/backends/encoder.py`, `src/jeba/config.py`.
**Depends on:** — · **Requirement:** NFR-P01, NFR-C05, OPS-01, EXT-02.
**Done when:** acceleration activates only under documented conditions, falls back cleanly, and never
changes the response shape; existing `tests/test_fast.py` still pass.
**Tests:** `tests/test_fast.py`, `tests/test_config.py`, `tests/test_encoder.py` · **Gate:** full.
**Commit:** `perf(fast): wire acceleration seam and CUDA-graph encode`.

## B2: Fast-path micro-benchmark + parity

**What:** Micro-benchmark stock vs fast (p50/p95, throughput vs batch) with a probability parity
check; record results and close NFR-P01.
**Where:** `benchmarks/fast_path.py`, `benchmarks/report.md`, `benchmarks/README.md`.
**Depends on:** B1 · **Requirement:** NFR-P01, OPS-06.
**Done when:** the benchmark reproduces from committed commands and reports parity within tolerance.
**Tests:** conditional-by-extra (skip without CUDA) · **Gate:** build.
**Commit:** `feat(benchmarks): add fast-path micro-benchmark`.
