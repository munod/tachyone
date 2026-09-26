# Requirement Traceability Matrix

**Purpose:** Map every requirement to its design component and the task(s) that implement it.
**Sources:** `docs/requirements/functional.md`, `docs/requirements/non-functional.md`,
`docs/architecture.md`, `docs/tasks.md`.

Status values: `Pending` (no code) → `In Progress` → `Implemented` → `Verified`.
All requirements are implemented as of `v0.3.0`; `NFR-C06`/`NFR-C07` (per-language ECE) remain
**partial** pending the calibration work tracked in `BACKLOG.md` B-1.

---

## Functional → Design → Task

| Req ID | Design component | Task(s) | Status |
| --- | --- | --- | --- |
| WIRE-01 | `serve.py` (auth) | M2-T3 | Implemented |
| WIRE-02 | `wire.py` | M1-T4 | Implemented |
| WIRE-03 | `wire.py` | M1-T4 | Implemented |
| WIRE-04 | `wire.py` + `serve.py` | M1-T7, M2-T3 | Implemented |
| WIRE-05 | `client.py` | M2-T5 | Implemented |
| WIRE-06 | `wire.py` + fixtures | M1-T6 | Implemented |
| WIRE-07 | `wire.py` (`Usage`) | M1-T4 | Implemented |
| PRIM-01 | `primitives.py` (noul) | M1-T1 | Implemented |
| PRIM-02 | `primitives.py` (choice) | M1-T2 | Implemented |
| PRIM-03 | `primitives.py` (score) | M1-T3 | Implemented |
| PRIM-04 | `primitives.py` (validators) | M1-T2, M1-T3 | Implemented |
| PRIM-05 | `calibration.py` | M3-T3 | Implemented |
| PRIM-06 | `wire.py` | M1-T4 | Implemented |
| BACK-01 | `backends/base.py` | M1-T5 | Implemented |
| BACK-02 | `backends/llm.py` | M2-T2 | Implemented |
| BACK-03 | `backends/encoder.py` | M3-T5 | Implemented |
| BACK-04 | `backends/onnx.py` | M5-T1 | Implemented |
| BACK-05 | `config.py` + seam | M2-T1, M2-T2 | Implemented |
| BACK-06 | `backends/encoder.py` | M3-T5, M3-T8 | Implemented |
| BACK-07 | `agent.py` | M3-T4, M3-T7 | Implemented |
| ROUTE-01 | `router.py` | M3-T1 | Implemented |
| ROUTE-02 | `router.py` | M3-T1 | Implemented |
| ROUTE-03 | `router.py` | M3-T1 | Implemented |
| ROUTE-04 | `router.py` | M3-T2 | Implemented |
| ROUTE-05 | `router.py` + `backends/encoder.py` | M3-T1, M3-T8 | Implemented |
| CAL-01 | `finetune_rlcd.py` | M4-T3 | Implemented |
| CAL-02 | `fit_calibration.py` | M4-T4 | Implemented |
| CAL-03 | `calibration.py` | M3-T3 | Implemented |
| CAL-04 | `fit_calibration.py` | M4-T4 | Implemented |
| CAL-05 | `calibration.py` + `schemas.py` | M2-T7, M3-T3 | Implemented (no-op: distributions are canonical) |
| CAL-06 | `calibration.py` + `handoff.py` | B-3 (B3-T1..T4) | Implemented |
| EXT-01 | `hooks.py` | M3-T6 | Implemented |
| EXT-02 | `hooks.py` + contract | M3-T6 | Implemented |
| EXT-03 | `router.py` + `serve.py` | M5-T2 (override plumbing) | Implemented |
| EXT-04 | `agent.py` + `serve.py` | M2-T4, M3-T7 | Implemented |
| EXT-05 | `client.py` | M2-T5 | Implemented |
| SERVE-01 | `serve.py` | M2-T3 | Implemented |
| SERVE-02 | `serve.py` | M2-T4 | Implemented |
| SERVE-03 | `cli.py` | M2-T6 | Implemented |
| SERVE-04 | `pyproject.toml` entry points | M0-T1, M2-T6, M5-T3 | Implemented |
| SERVE-05 | `client.py` | M2-T5 | Implemented |
| SERVE-06 | `config.py` | M2-T1 | Implemented |
| SERVE-07 | `cli.py` presets | M2-T6 | Implemented |
| SERVE-08 | `schemas.py` | M2-T7 | Implemented |
| TRAIN-01 | `generate_data.py` | M4-T1 | Implemented |
| TRAIN-02 | `finetune_rlcd.py` | M4-T2 | Implemented |
| TRAIN-03 | `finetune_rlcd.py` | M4-T3 | Implemented |
| TRAIN-04 | `fit_calibration.py` | M4-T4 | Implemented |
| TRAIN-05 | `training/evaluate.py` | M4-T5 | Implemented |
| TRAIN-06 | `training/configs/` | M4-T2, M4-T6 | Implemented |
| TRAIN-07 | `generate_data.py` | M4-T1 | Implemented |
| TRAIN-08 | `generate_data.py` + `evaluate.py` | B-4 (B4-T1..T3) | Implemented |
| OPS-01 | `pyproject.toml` extras | M0-T1, M5-T1, M5-T2 | Implemented |
| OPS-02 | `mcp/` | M5-T3 | Implemented |
| OPS-03 | `integrations/langchain.py` | M5-T4 | Implemented |
| OPS-04 | `docker/` | M5-T5 | Implemented |
| OPS-05 | `.github/workflows/ci.yml` | M0-T4 | Implemented |
| OPS-06 | `benchmarks/` | M4-T5, M6-T1 | Partial (synthetic report reproducible; public probes open — B-7) |
| OPS-07 | release artifacts | M6-T3 | Implemented |
| OPS-08 | `telemetry.py` | M5-T6 | Implemented |

## Non-functional → Design → Task

| Req ID | Design component | Task(s) | Status |
| --- | --- | --- | --- |
| NFR-P01 | `backends/encoder.py` / `fast.py` | M3-T5, M5-T2, B-2 | Done (stock 9.1/10.8 ms; fast 3.7/4.1 ms) |
| NFR-P02 | `backends/encoder.py` | M3-T5, M3-T8 | Implemented |
| NFR-P03 | `router.py` | M3-T1 | Implemented |
| NFR-P04 | `backends/llm.py` | M2-T2 | Implemented |
| NFR-P05 | `agent.py` | M3-T4 | Implemented |
| NFR-P06 | `serve.py` + `router.py` | M3-T2 | Implemented |
| NFR-P07 | `fast.py` | Backlog B-2 | Done (2.68× p50, 0 top-label flips) |
| NFR-R01 | packaging | M0-T1 | Implemented |
| NFR-R02 | `backends/encoder.py` | M3-T5 | Implemented |
| NFR-R03 | `backends/encoder.py` | M3-T5 | Implemented |
| NFR-R04 | `router.py` | M3-T2 | Implemented |
| NFR-R05 | `finetune_rlcd.py` | M4-T2 | Implemented |
| NFR-C01 | `wire.py` + contract tests | M1-T6 | Implemented |
| NFR-C02 | `backends/*` + validators | M1-T2, M1-T3, M2-T2 | Implemented |
| NFR-C03 | `backends/encoder.py` | M3-T5 | Implemented |
| NFR-C04 | `training/configs/` | M4-T6 | Implemented |
| NFR-C05 | `fast.py` / extras guards | M5-T2, B-2 | Done |
| NFR-C06 | `fit_calibration.py` | M4-T4 | Partial (accuracy met; per-language ECE open) |
| NFR-C07 | `fit_calibration.py` | Backlog B-1 | Partial (accuracy met; per-language ECE open) |
| NFR-X01 | `serve.py` + e2e | M2-T8 | Implemented |
| NFR-X02 | contract tests | M1-T6, M3-T6, M2-T8 | Implemented |
| NFR-X03 | `pyproject.toml` | M0-T1 | Implemented |
| NFR-X04 | packaging + tests | M0-T1, M3-T8 | Implemented |
| NFR-X05 | `backends/encoder.py` | M3-T5 | Implemented |
| NFR-S01 | repo config | M0-T3 | Implemented |
| NFR-S02 | `config.py` | M2-T1 | Implemented |
| NFR-S03 | `backends/encoder.py` | M3-T8 | Implemented |
| NFR-S04 | `backends/encoder.py` | M3-T5, M3-T8 | Implemented |
| NFR-S05 | `telemetry.py` | M5-T6 | Implemented |
| NFR-S06 | `serve.py` | M2-T3 | Implemented |
| NFR-M01 | ruff config | M0-T2 | Implemented |
| NFR-M02 | pyright config | M0-T2 | Implemented |
| NFR-M03 | CI + contract test | M0-T4, M1-T6 | Implemented |
| NFR-M04 | repo convention | M0-T3 | Implemented |
| NFR-M05 | contract tests | M1-T6, M2-T8, M3-T5 | Implemented |
| NFR-M06 | extras guards | M0-T1, M5-T1..T4 | Implemented |
| NFR-O01 | `serve.py` `/health` | M2-T4 | Implemented |
| NFR-O02 | logging | M2-T3 | Implemented |
| NFR-O03 | `hooks.py` | M3-T6 | Implemented |
| NFR-D01 | `README.md` | M0-T1 (docs already exist) | Implemented |
| NFR-D02 | `docs/` | M6-T2 | Implemented |
| NFR-D03 | `LICENSE` | M0-T3 | Implemented |
| NFR-D04 | benchmarks + model card | M6-T1, M6-T3 | Implemented |
| NFR-A | n/a (no GUI) | — | N/A |

---

## Coverage Summary

| Category | Count | Mapped | Unmapped |
| --- | --- | --- | --- |
| Functional (WIRE/PRIM/BACK/ROUTE/CAL/EXT/SERVE/TRAIN/OPS) | 60 | 60 | 0 |
| Non-functional | 44 | 43 | 1 (NFR-A, N/A) |
| **Total** | **104** | **103** | **0 actionable** |

> When a task is completed, update the matching rows to `Implemented`, and after validation
> to `Verified` (see `docs/testing.md` and the TLC validate flow in `.specs/project/STATE.md`).
