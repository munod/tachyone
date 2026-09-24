# State

**Last Updated:** 2026-09-24
**Current Work:** Phase M2 (LLM Backend + Serving) complete. Next: M3 — Local Encoder.

## Milestone Status

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Bootstrap | ✅ Complete | uv + py3.12, ruff/pyright/pytest, CI |
| M1 Wire Contract | ✅ Complete | primitives + wire.py, contract suite |
| M2 LLM Backend + Serve | ✅ Complete | llm/fake backends, FastAPI, SDK, CLI, presets, decide, e2e |
| M3 Local Encoder | ⏳ Next | encoder backend, router, calibration, hooks |
| M4 Training + Calibration | ⬜ Not started | |
| M5 Ecosystem + Accel | ⬜ Not started | |
| M6 Proof + Release | ⬜ Not started | |

> This is the persistent memory for the jeba project across sessions. Decisions here are
> authoritative. Anything marked **DECIDED** must not be reopened without a new ADR.

---

## Recent Decisions (Last 60 days)

### AD-001: Drop-in Jev `/v1/systemone` with additive extensions (2026-09-24)

**Decision:** jeba speaks the exact TypeSafe Jev wire contract and accepts existing Jev
clients unchanged. Extensions (router control, hooks, `predict_batch`, integrations) are
additive and never mutate the canonical request/response shape.
**Reason:** Existing clients are the fastest path to adoption and to a verifiable contract;
a stable wire is a stronger asset than a bespoke API.
**Trade-off:** We inherit Jev's primitive semantics and error model; some ideas must be
expressed as extensions rather than first-class fields.
**Impact:** `docs/protocol.md` is the source of truth; contract tests gate every backend.
See `docs/adr/ADR-0001-jev-drop-in-protocol.md`.

### AD-002: Plugable backend with phased LLM → encoder (2026-09-24)

**Decision:** A stable `Backend` interface sits behind `wire.py`. Phase 2 ships an LLM
(structured outputs) backend; Phase 3 adds a local encoder backend; ONNX is a later
backend. The wire never depends on which backend is active.
**Reason:** Delivers an end-to-end system immediately while de-risking the harder encoder
training work.
**Trade-off:** An extra abstraction layer and two code paths to maintain.
**Impact:** `backends/base.py` is the seam; contract tests run against every backend.
See `docs/adr/ADR-0002-pluggable-backend-phasing.md`.

### AD-003: Python 3.12 + uv (2026-09-24)

**Decision:** Pin `requires-python = ">=3.12,<3.13"` and use `uv` for env + lockfile.
**Reason:** System Python is 3.14 but torch/transformers do not support it yet; uv is fast
and gives a committed `uv.lock` for reproducibility.
**Trade-off:** Contributors must use 3.12; slightly narrower dependency ceiling.
**Impact:** `.python-version`, `pyproject.toml`, CI all pin 3.12.
See `docs/adr/ADR-0003-python312-uv.md`.

### AD-004: Local-first, offline-capable core (2026-09-24)

**Decision:** The base install must run fully offline with no API key or hosted service.
LLM/remote backends are optional extras. Server HTTP mode is one mode, not the only one.
**Reason:** Privacy, latency, cost, and edge/on-prem requirements are the core differentiator.
**Trade-off:** We cannot assume a network or cloud for defaults.
**Impact:** No hosted dependency in core; extras isolate heavy/optional deps.
See `docs/adr/ADR-0004-local-first.md`.

### AD-005: Encoder backend + RLCD proper-scoring calibration (2026-09-24)

**Decision:** The local backend is a single-forward-pass encoder (ModernBERT / mmBERT) with
three task heads, trained with RLCD against strictly proper scoring rules, plus temperature
fitting to minimize ECE.
**Reason:** Non-autoregressive single-pass gives low latency; proper scoring yields
calibrated probabilities that make `confidence` meaningful.
**Trade-off:** Requires data generation and calibration infrastructure; bounded by 12GB VRAM.
**Impact:** `docs/training.md` defines the pipeline; `calibration.py` owns confidence.
See `docs/adr/ADR-0005-encoder-rlcd.md`.

### AD-006: Apache-2.0 and opt-out telemetry (2026-09-24)

**Decision:** License the project Apache-2.0. If telemetry is ever added it must be
opt-out (`DO_NOT_TRACK=1` / equivalent) and never required for function.
**Reason:** Matches the open-source ecosystem we build on (Laya) and keeps trust local-first.
**Trade-off:** No permissive-license leverage over downstream forks; telemetry is weak by design.
**Impact:** No secrets in repo; no mandatory phone-home.
See `docs/adr/ADR-0006-license-telemetry.md`.

### AD-007: Multilingual from Phase 3 via mmBERT-base (2026-09-24)

**Decision:** Ship a multilingual checkpoint (mmBERT-base, 100+ languages) alongside the
English checkpoint, selected automatically by a script/language router.
**Reason:** Multilingual capability is a primary differentiator and is not retrofittable cheaply.
**Trade-off:** Two checkpoints to load/manage; router adds a small amount of complexity.
**Impact:** `router.py`; benchmark suite must include multilingual probes.

---

## Active Blockers

### B-001: Python 3.14 vs torch/transformers

**Discovered:** 2026-09-24
**Impact:** System interpreter cannot run the model layer; any accidental use of 3.14 breaks deps.
**Workaround:** Pin 3.12 via `.python-version` and `requires-python`.
**Resolution:** Revisit when torch/transformers publish 3.14 wheels; tracked as AD-003 consequence.

### B-002: RTX 3060 12GB training budget

**Discovered:** 2026-09-24
**Impact:** Full fine-tuning of large encoders is infeasible; naive LoRA/QLoRA may OOM.
**Workaround:** LoRA/QLoRA + gradient checkpointing + small effective batch + grad accumulation.
**Resolution:** Validate with the Phase 4 training run; document VRAM ceilings in `docs/training.md`.

---

## Lessons Learned

### L-001: Contract-first prevents backend churn

**Context:** Borrowed from Laya's convention that public API changes require updating the
contract test in the same PR.
**Problem:** Without a frozen contract, swapping LLM → encoder silently changes responses.
**Solution:** Freeze `wire.py` and `docs/protocol.md` in Phase 1; gate every backend on the
same contract tests.
**Prevents:** Divergence between backends and broken existing clients.

---

## Quick Tasks Completed

| #   | Description | Date | Commit | Status |
| --- | ----------- | ---- | ------ | ------ |
| M0-T1 | uv project skeleton (py3.12, extras, entry points, uv.lock) | 2026-09-24 | `build: bootstrap uv project (python 3.12)` | ✅ |
| M0-T2 | ruff + pyright + pytest config and dev deps | 2026-09-24 | `chore: configure ruff, pyright, pytest` | ✅ |
| M0-T3 | repo hygiene (.gitignore, .editorconfig, LICENSE, git init) | 2026-09-24 | `chore: initialize repository with project specifications` | ✅ |
| M0-T4 | CI workflow (ruff + pyright + pytest on py3.12) | 2026-09-24 | `ci: add lint, type, and test gates` | ✅ |
| M1-T1 | `noul` primitive | 2026-09-24 | `feat(primitives): add noul question and answer` | ✅ |
| M1-T2 | `choice` primitive (1..255 options) | 2026-09-24 | `feat(primitives): add choice question and answer` | ✅ |
| M1-T3 | `score` primitive (2..10 levels) | 2026-09-24 | `feat(primitives): add score question and answer` | ✅ |
| M1-T4 | wire envelope + errors + `answer()` | 2026-09-24 | `feat(wire): add systemone request and response` | ✅ |
| M1-T5 | backend seam + fake backend | 2026-09-24 | `feat(backends): add backend protocol and fake backend` | ✅ |
| M1-T6 | golden contract tests | 2026-09-24 | `test(contract): freeze systemone wire parity` | ✅ |
| M1-T7 | error-shape tests | 2026-09-24 | `test(contract): cover systemone error shapes` | ✅ |
| M2-T1 | env configuration | 2026-09-24 | `feat(config): add environment configuration` | ✅ |
| M2-T2 | structured-output LLM backend | 2026-09-24 | `feat(backends): add structured-output LLM backend` | ✅ |
| M2-T3 | `/v1/systemone` HTTP endpoint | 2026-09-24 | `feat(serve): add systemone HTTP endpoint` | ✅ |
| M2-T4 | predict/batch/health endpoints | 2026-09-24 | `feat(serve): add predict, batch, and health endpoints` | ✅ |
| M2-T5 | Python SDK client + backoff | 2026-09-24 | `feat(sdk): add python client with backoff` | ✅ |
| M2-T6 | CLI + presets | 2026-09-24 | `feat(cli): add predict CLI and presets` | ✅ |
| M2-T7 | `decide()` from JSON Schema | 2026-09-24 | `feat(schemas): add decide() from json schema` | ✅ |
| M2-T8 | repointed-Jev-client e2e | 2026-09-24 | `test(e2e): verify repointed jev client` | ✅ |

---

## Deferred Ideas

- [ ] Provider registry + OpenAI-compatible LLM backend — captured during: design
- [ ] Streaming multi-question responses — captured during: design
- [ ] Optional opt-out telemetry — captured during: AD-006
- [ ] TileLang/CUDA-graphs fast path micro-benchmark — captured during: roadmap M5
- [ ] Web console for interactive triage — captured during: scope discussion

---

## Open Decisions (need resolution before the blocking phase)

- [x] **OD-1:** RESOLVED (2026-09-24, M2) — Provider-agnostic OpenAI-compatible surface with
      an injectable transport; no new core dependency. See `docs/adr/ADR-0008`.
- [ ] **OD-2:** Telemetry default (on vs off) and exact env var name. Blocking: M5.
- [x] **OD-3:** RESOLVED (2026-09-24) — Weights fetched from the Hugging Face Hub on demand and
      cached locally (`JEBA_MODELS_DIR` or `~/.cache/jeba/models`), with `jeba download` prefetch
      and cache-only offline mode. M3 uses public base encoders + untrained heads. See ADR-0010.
- [ ] **OD-4:** ONNX vs TileLang sequencing within M5. Blocking: M5.
- [x] **OD-5:** RESOLVED (2026-09-24, M2) — ``/predict`` and ``/predict/batch`` mirror the
      canonical response shape and are additive; ``/v1/systemone`` is untouched. See `docs/adr/ADR-0009`.

---

## Todos

- [x] Create `.specs/features/*` specs for each phase (done in this baseline).
- [ ] Convert `docs/tasks.md` into `.specs/features/<phase>/tasks.md` at execution time.
- [ ] Add `mermaid-studio` skill (recommended) for rendered architecture diagrams.
- [ ] Add `.specs/features/*/design.md` and `tasks.md` for the remaining phases at execution time
      (only `wire-contract` has a design.md so far; the rest are covered by `docs/architecture.md`).

## Documentation Baseline (2026-09-24)

The full spec/design/planning documentation set was produced with **zero production code**:

- Root: `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `LICENSE` (Apache-2.0).
- `docs/`: `overview.md`, `roadmap.md`, `protocol.md`, `architecture.md`, `tasks.md`,
  `testing.md`, `training.md`.
- `docs/requirements/`: `functional.md` (51 reqs), `non-functional.md` (39 reqs), `traceability.md`.
- `docs/adr/`: `README.md` + ADR-0001..ADR-0007.
- `.specs/project/`: `PROJECT.md`, `ROADMAP.md`, `STATE.md`.
- `.specs/features/`: `wire-contract/` (spec + design), `llm-backend/`, `encoder-backend/`,
  `training-calibration/`, `ecosystem/` (specs).

**Verified:** 51/51 functional requirements traced to tasks; 7/7 ADRs indexed and present;
all relative doc links resolve; 0 `.py` files created.

---

## Preferences

**Model Guidance Shown:** never
