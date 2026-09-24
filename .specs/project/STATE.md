# State

**Last Updated:** 2026-09-24
**Current Work:** Documentation baseline (pre-implementation). No code exists yet.

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
| —   | (none yet)  | —    | —      | —      |

---

## Deferred Ideas

- [ ] Provider registry + OpenAI-compatible LLM backend — captured during: design
- [ ] Streaming multi-question responses — captured during: design
- [ ] Optional opt-out telemetry — captured during: AD-006
- [ ] TileLang/CUDA-graphs fast path micro-benchmark — captured during: roadmap M5
- [ ] Web console for interactive triage — captured during: scope discussion

---

## Open Decisions (need resolution before the blocking phase)

- [ ] **OD-1:** Exact LLM provider abstraction surface (single `complete_structured` call vs
      provider plugin interface). Blocking: M2. See `docs/architecture.md`.
- [ ] **OD-2:** Telemetry default (on vs off) and exact env var name. Blocking: M5.
- [ ] **OD-3:** How model weights are distributed (Hugging Face hub vs bundled vs download-on-first-use).
      Blocking: M3. Note: offline-first implies weights must be cached/fetchable without a key.
- [ ] **OD-4:** ONNX vs TileLang sequencing within M5. Blocking: M5.
- [ ] **OD-5:** Whether to expose a `/predict` extension shape distinct from `/v1/systemone`,
      and its schema. Blocking: M2.

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
