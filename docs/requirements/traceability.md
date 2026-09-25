# Requirement Traceability Matrix

**Purpose:** Map every requirement to its design component and the task(s) that implement it.
**Sources:** `docs/requirements/functional.md`, `docs/requirements/non-functional.md`,
`docs/architecture.md`, `docs/tasks.md`.

Status values: `Pending` (no code) → `In Progress` → `Implemented` → `Verified`.
All rows are **Pending** at baseline (pre-implementation).

---

## Functional → Design → Task

| Req ID | Design component | Task(s) | Status |
| --- | --- | --- | --- |
| WIRE-01 | `serve.py` (auth) | M2-T3 | Pending |
| WIRE-02 | `wire.py` | M1-T4 | Pending |
| WIRE-03 | `wire.py` | M1-T4 | Pending |
| WIRE-04 | `wire.py` + `serve.py` | M1-T7, M2-T3 | Pending |
| WIRE-05 | `client.py` | M2-T5 | Pending |
| WIRE-06 | `wire.py` + fixtures | M1-T6 | Pending |
| WIRE-07 | `wire.py` (`Usage`) | M1-T4 | Pending |
| PRIM-01 | `primitives.py` (noul) | M1-T1 | Pending |
| PRIM-02 | `primitives.py` (choice) | M1-T2 | Pending |
| PRIM-03 | `primitives.py` (score) | M1-T3 | Pending |
| PRIM-04 | `primitives.py` (validators) | M1-T2, M1-T3 | Pending |
| PRIM-05 | `calibration.py` | M3-T3 | Pending |
| PRIM-06 | `wire.py` | M1-T4 | Pending |
| BACK-01 | `backends/base.py` | M1-T5 | Pending |
| BACK-02 | `backends/llm.py` | M2-T2 | Pending |
| BACK-03 | `backends/encoder.py` | M3-T5 | Pending |
| BACK-04 | `backends/onnx.py` | M5-T1 | Pending |
| BACK-05 | `config.py` + seam | M2-T1, M2-T2 | Pending |
| BACK-06 | `backends/encoder.py` | M3-T5, M3-T8 | Pending |
| BACK-07 | `agent.py` | M3-T4, M3-T7 | Pending |
| ROUTE-01 | `router.py` | M3-T1 | Pending |
| ROUTE-02 | `router.py` | M3-T1 | Pending |
| ROUTE-03 | `router.py` | M3-T1 | Pending |
| ROUTE-04 | `router.py` | M3-T2 | Pending |
| ROUTE-05 | `router.py` + `backends/encoder.py` | M3-T1, M3-T8 | Pending |
| CAL-01 | `finetune_rlcd.py` | M4-T3 | Pending |
| CAL-02 | `fit_calibration.py` | M4-T4 | Pending |
| CAL-03 | `calibration.py` | M3-T3 | Pending |
| CAL-04 | `fit_calibration.py` | M4-T4 | Pending |
| CAL-05 | `calibration.py` + `schemas.py` | M2-T7, M3-T3 | Pending |
| EXT-01 | `hooks.py` | M3-T6 | Pending |
| EXT-02 | `hooks.py` + contract | M3-T6 | Pending |
| EXT-03 | `router.py` + `serve.py` | M5-T2 (override plumbing) | Pending |
| EXT-04 | `agent.py` + `serve.py` | M2-T4, M3-T7 | Pending |
| EXT-05 | `client.py` | M2-T5 | Pending |
| SERVE-01 | `serve.py` | M2-T3 | Pending |
| SERVE-02 | `serve.py` | M2-T4 | Pending |
| SERVE-03 | `cli.py` | M2-T6 | Pending |
| SERVE-04 | `pyproject.toml` entry points | M0-T1, M2-T6, M5-T3 | Pending |
| SERVE-05 | `client.py` | M2-T5 | Pending |
| SERVE-06 | `config.py` | M2-T1 | Pending |
| SERVE-07 | `cli.py` presets | M2-T6 | Pending |
| SERVE-08 | `schemas.py` | M2-T7 | Pending |
| TRAIN-01 | `generate_data.py` | M4-T1 | Pending |
| TRAIN-02 | `finetune_rlcd.py` | M4-T2 | Pending |
| TRAIN-03 | `finetune_rlcd.py` | M4-T3 | Pending |
| TRAIN-04 | `fit_calibration.py` | M4-T4 | Pending |
| TRAIN-05 | `benchmarks/evaluate.py` | M4-T5 | Pending |
| TRAIN-06 | `training/configs/` | M4-T2, M4-T6 | Pending |
| TRAIN-07 | `generate_data.py` | M4-T1 | Pending |
| OPS-01 | `pyproject.toml` extras | M0-T1, M5-T1, M5-T2 | Pending |
| OPS-02 | `mcp/` | M5-T3 | Pending |
| OPS-03 | `integrations/langchain.py` | M5-T4 | Pending |
| OPS-04 | `docker/` | M5-T5 | Pending |
| OPS-05 | `.github/workflows/ci.yml` | M0-T4 | Pending |
| OPS-06 | `benchmarks/` | M4-T5, M6-T1 | Pending |
| OPS-07 | release artifacts | M6-T3 | Pending |
| OPS-08 | `telemetry.py` | M5-T6 | Pending |

## Non-functional → Design → Task

| Req ID | Design component | Task(s) | Status |
| --- | --- | --- | --- |
| NFR-P01 | `backends/encoder.py` / `fast.py` | M3-T5, M5-T2, B-2 | Done (stock 9.1/10.8 ms; fast 3.7/4.1 ms) |
| NFR-P02 | `backends/encoder.py` | M3-T5, M3-T8 | Pending |
| NFR-P03 | `router.py` | M3-T1 | Pending |
| NFR-P04 | `backends/llm.py` | M2-T2 | Pending |
| NFR-P05 | `agent.py` | M3-T4 | Pending |
| NFR-P06 | `serve.py` + `router.py` | M3-T2 | Pending |
| NFR-P07 | `fast.py` | Backlog B-2 | Done (2.47× p50, 0 top-label flips) |
| NFR-R01 | packaging | M0-T1 | Pending |
| NFR-R02 | `backends/encoder.py` | M3-T5 | Pending |
| NFR-R03 | `backends/encoder.py` | M3-T5 | Pending |
| NFR-R04 | `router.py` | M3-T2 | Pending |
| NFR-R05 | `finetune_rlcd.py` | M4-T2 | Pending |
| NFR-C01 | `wire.py` + contract tests | M1-T6 | Pending |
| NFR-C02 | `backends/*` + validators | M1-T2, M1-T3, M2-T2 | Pending |
| NFR-C03 | `backends/encoder.py` | M3-T5 | Pending |
| NFR-C04 | `training/configs/` | M4-T6 | Pending |
| NFR-C05 | `fast.py` / extras guards | M5-T2, B-2 | Done |
| NFR-C06 | `fit_calibration.py` | M4-T4 | Pending |
| NFR-C07 | `fit_calibration.py` | Backlog B-1 | Pending |
| NFR-X01 | `serve.py` + e2e | M2-T8 | Pending |
| NFR-X02 | contract tests | M1-T6, M3-T6, M2-T8 | Pending |
| NFR-X03 | `pyproject.toml` | M0-T1 | Pending |
| NFR-X04 | packaging + tests | M0-T1, M3-T8 | Pending |
| NFR-X05 | `backends/encoder.py` | M3-T5 | Pending |
| NFR-S01 | repo config | M0-T3 | Pending |
| NFR-S02 | `config.py` | M2-T1 | Pending |
| NFR-S03 | `backends/encoder.py` | M3-T8 | Pending |
| NFR-S04 | `backends/encoder.py` | M3-T5, M3-T8 | Pending |
| NFR-S05 | `telemetry.py` | M5-T6 | Pending |
| NFR-S06 | `serve.py` | M2-T3 | Pending |
| NFR-M01 | ruff config | M0-T2 | Pending |
| NFR-M02 | pyright config | M0-T2 | Pending |
| NFR-M03 | CI + contract test | M0-T4, M1-T6 | Pending |
| NFR-M04 | repo convention | M0-T3 | Pending |
| NFR-M05 | contract tests | M1-T6, M2-T8, M3-T5 | Pending |
| NFR-M06 | extras guards | M0-T1, M5-T1..T4 | Pending |
| NFR-O01 | `serve.py` `/health` | M2-T4 | Pending |
| NFR-O02 | logging | M2-T3 | Pending |
| NFR-O03 | `hooks.py` | M3-T6 | Pending |
| NFR-D01 | `README.md` | M0-T1 (docs already exist) | Pending |
| NFR-D02 | `docs/` | M6-T2 | Pending |
| NFR-D03 | `LICENSE` | M0-T3 | Pending |
| NFR-D04 | benchmarks + model card | M6-T1, M6-T3 | Pending |
| NFR-A | n/a (no GUI) | — | N/A |

---

## Coverage Summary

| Category | Count | Mapped | Unmapped |
| --- | --- | --- | --- |
| Functional (WIRE/PRIM/BACK/ROUTE/CAL/EXT/SERVE/TRAIN/OPS) | 51 | 51 | 0 |
| Non-functional | 39 | 38 | 1 (NFR-A, N/A) |
| **Total** | **90** | **89** | **0 actionable** |

> When a task is completed, update the matching rows to `Implemented`, and after validation
> to `Verified` (see `docs/testing.md` and the TLC validate flow in `.specs/project/STATE.md`).
