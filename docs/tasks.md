# Task Plan

**Status:** Phase M0 complete (M0-T1..T4). M1 not started.
**Design:** `docs/architecture.md` · **Requirements:** `docs/requirements/`
**Gates:** `uv run ruff check .` (lint) · `uv run pyright` (types) · `uv run pytest` (tests), per `docs/testing.md`.

> Every task is atomic (one deliverable), names its requirement IDs, lists dependencies, and
> has a binary "Done when" check. Commit messages use conventional commits.
> **[P]** = parallelizable within its phase. **Test co-location:** tests are written in the
> task that creates the code, never deferred.

---

## Execution Plan

### Phase M0 — Bootstrap (Sequential)

```
M0-T1 → M0-T2 → M0-T3 → M0-T4
```

### Phase M1 — Wire Contract (T1 first, then parallel)

```
                 ┌→ M1-T3 ─┐
M1-T1 → M0-T1,2 ─┼→ M1-T4 ─┼→ M1-T6 → M1-T7
                 └→ M1-T5 ─┘
        M1-T2 ─────────────┘
```

### Phase M2 — LLM Backend + Serving (T1 first, then parallel)

```
M1-T6 → M2-T1 ─┬→ M2-T3 ─→ M2-T4 ─┐
               ├→ M2-T2 ───────────┤
               ├→ M2-T5 ───────────┼→ M2-T8
               ├→ M2-T6 ───────────┤
               └→ M2-T7 ───────────┘
```

### Phase M3 — Local Encoder (T3,T4 first, then parallel)

```
M2 → M3-T3 ─┬→ M3-T4 ─→ M3-T5 ─┬→ M3-T8
            │                   ├→ M3-T6 ─┘
M3-T1 ──────┘→ M3-T2 ───────────┘
M3-T4 → M3-T7
```

### Phase M4 — Training (parallel after T1)

```
M3-T5 → M4-T1 ─┬→ M4-T2 → M4-T3 → M4-T4 ─┐
               ├→ M4-T5 ─────────────────┼→ M4-T6
               └──────────────────────────┘
```

### Phase M5 — Ecosystem (parallel)

```
M3, M4 → M5-T1 ─┐
        → M5-T2 ─┤
        → M5-T3 ─┼→ M5-T6
        → M5-T4 ─┤
        → M5-T5 ─┘
```

### Phase M6 — Proof & Release (Sequential)

```
M5 → M6-T1 → M6-T2 → M6-T3
```

---

## Task Breakdown

### M0 — Bootstrap

#### M0-T1: uv project skeleton

**What:** `pyproject.toml` (py>=3.12,<3.13), extras (`serve`/`fast`/`onnx`/`langchain`/`mcp`/`train`), console entry points, `src/jeba/__init__.py`, `tests/__init__.py`.
**Where:** `pyproject.toml`, `.python-version`, `uv.lock`, `src/jeba/`, `tests/`.
**Depends on:** None · **Reuses:** — · **Requirement:** NFR-X03, NFR-M06, OPS-01.
**Done when:** `uv sync` succeeds on 3.12; `uv run python -c "import jeba"` works; entry points resolve to importable stubs.
**Tests:** none (config) · **Gate:** quick · **Commit:** `build: bootstrap uv project (python 3.12)`.

#### M0-T2: quality tooling config

**What:** ruff, pyright, pytest configuration and dev dependencies.
**Where:** `pyproject.toml` (modify).
**Depends on:** M0-T1 · **Requirement:** NFR-M01, NFR-M02, OPS-05.
**Done when:** `uv run ruff check .`, `uv run pyright`, `uv run pytest` all exit 0 on the skeleton.
**Tests:** none (tooling) · **Gate:** build · **Commit:** `chore: configure ruff, pyright, pytest`.

#### M0-T3: repo hygiene

**What:** `.gitignore`, `.editorconfig`, `LICENSE` (Apache-2.0), `git init`.
**Where:** root.
**Depends on:** M0-T1 · **Requirement:** NFR-D03, NFR-S01.
**Done when:** no secrets/virtualenvs tracked; license present; first commit made.
**Tests:** none · **Gate:** quick · **Commit:** `chore: add license and repo hygiene`.

#### M0-T4: CI workflow

**What:** GitHub Actions running ruff + pyright + pytest on push/PR with Python 3.12.
**Where:** `.github/workflows/ci.yml`.
**Depends on:** M0-T2 · **Requirement:** OPS-05, NFR-M03.
**Done when:** CI green on the skeleton; future public-API changes are gated by the contract test step.
**Tests:** none (CI config) · **Gate:** build · **Commit:** `ci: add lint, type, and test gates`.

---

### M1 — Wire Contract

#### M1-T1: `noul` primitive

**What:** `NoulQuestion` + `NoulAnswer` pydantic models and validators.
**Where:** `src/jeba/primitives.py`.
**Depends on:** M0-T1 · **Requirement:** PRIM-01.
**Done when:** valid `noul` validates; answer serializes `{type:"noul", noul: float}`.
**Tests:** unit (`tests/test_primitives_noul.py`) · **Gate:** quick · **Commit:** `feat(primitives): add noul question and answer`.

#### M1-T2: `choice` primitive

**What:** `ChoiceQuestion` + `ChoiceAnswer`; enforce ≤255 options; probability key coverage.
**Where:** `src/jeba/primitives.py` (modify).
**Depends on:** M1-T1 · **Requirement:** PRIM-02, PRIM-04.
**Done when:** 255 options accepted, 256 rejected; probabilities keys checked.
**Tests:** unit (`tests/test_primitives_choice.py`) · **Gate:** quick · **Commit:** `feat(primitives): add choice question and answer`.

#### M1-T3: `score` primitive [P]

**What:** `ScoreQuestion` + `ScoreAnswer`; enforce 2–10 levels; `legend` mapping.
**Where:** `src/jeba/primitives.py` (modify).
**Depends on:** M1-T1 · **Requirement:** PRIM-03, PRIM-04.
**Done when:** 2 and 10 levels accepted, 1/11 rejected; `legend` generated from criteria.
**Tests:** unit (`tests/test_primitives_score.py`) · **Gate:** quick · **Commit:** `feat(primitives): add score question and answer`.

#### M1-T4: wire envelope [P]

**What:** `SystemOneRequest`, `SystemOneResponse`, `Usage`; `answer()` delegating to a backend.
**Where:** `src/jeba/wire.py`.
**Depends on:** M1-T1 · **Requirement:** WIRE-02, WIRE-03, WIRE-07, PRIM-06.
**Done when:** N questions → N answers keyed by id; response has `model/answers/usage`.
**Tests:** unit (`tests/test_wire.py`) · **Gate:** quick · **Commit:** `feat(wire): add systemone request and response`.

#### M1-T5: backend seam + fake backend [P]

**What:** `Backend` Protocol, `PredictionResult`, deterministic `FakeBackend` for tests.
**Where:** `src/jeba/backends/base.py`, `tests/fakes.py`.
**Depends on:** M1-T1 · **Requirement:** BACK-01.
**Done when:** `wire.answer` works with `FakeBackend`; no backend import leaks into wire.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(backends): add backend protocol and fake backend`.

#### M1-T6: golden contract tests [P]

**What:** Golden Jev fixtures + parity diff for all primitives and multi-question requests.
**Where:** `tests/fixtures/`, `tests/test_contract_wire.py`.
**Depends on:** M1-T2, M1-T3, M1-T4, M1-T5 · **Requirement:** WIRE-06, WIRE-02, WIRE-03, PRIM-01, PRIM-02, PRIM-03, PRIM-06.
**Done when:** serialized responses match fixtures byte-for-byte (modulo float tolerance); contract test is the CI gate.
**Tests:** contract · **Gate:** full · **Commit:** `test(contract): freeze systemone wire parity`.

#### M1-T7: error-shape tests

**What:** Tests for 401/422/429/529 and primitive limit violations.
**Where:** `tests/test_contract_errors.py`.
**Depends on:** M1-T6 · **Requirement:** WIRE-04, PRIM-02, PRIM-03.
**Done when:** each status produced with expected body; limit violations yield 422.
**Tests:** contract · **Gate:** full · **Commit:** `test(contract): cover systemone error shapes`.

---

### M2 — LLM Backend + Serving

#### M2-T1: configuration module

**What:** `config.py` reading `JEBA_*` env vars with defaults + validation.
**Where:** `src/jeba/config.py`.
**Depends on:** M1-T6 · **Requirement:** SERVE-06, NFR-S02.
**Done when:** defaults applied; secrets never logged; invalid config fails fast.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(config): add environment configuration`.

#### M2-T2: LLM backend

**What:** `LLMBackend` using structured outputs; prompt assembly from questions; JSON extraction; normalization.
**Where:** `src/jeba/backends/llm.py`.
**Depends on:** M2-T1 · **Requirement:** BACK-02, BACK-05.
**Done when:** answers all primitives via a configured provider; probabilities normalized; bounded retry on parse failure.
**Tests:** unit (mocked provider) + integration (optional live) · **Gate:** full · **Commit:** `feat(backends): add structured-output LLM backend`.

#### M2-T3: server `/v1/systemone`

**What:** FastAPI app, Bearer auth dependency, request/response wiring, error mapping.
**Where:** `src/jeba/serve.py`.
**Depends on:** M2-T1, M2-T2 · **Requirement:** SERVE-01, WIRE-01, WIRE-04.
**Done when:** contract tests pass over HTTP; 401 without key; 422 on bad body.
**Tests:** integration · **Gate:** full · **Commit:** `feat(serve): add systemone HTTP endpoint`.

#### M2-T4: extension endpoints

**What:** `/predict`, `/predict/batch`, `/health`.
**Where:** `src/jeba/serve.py` (modify).
**Depends on:** M2-T3 · **Requirement:** SERVE-02, EXT-04.
**Done when:** each endpoint responds; `/health` reports backend/device.
**Tests:** integration · **Gate:** full · **Commit:** `feat(serve): add predict, batch, and health endpoints`.

#### M2-T5: Python SDK client + retry

**What:** SDK client posting wire requests, parsing responses, exponential backoff + jitter on 429/529.
**Where:** `src/jeba/client.py`, `src/jeba/__init__.py`.
**Depends on:** M2-T1 · **Requirement:** SERVE-05, WIRE-05, EXT-05.
**Done when:** round-trip matches wire; retry verified against a fault-injected server.
**Tests:** unit + integration · **Gate:** full · **Commit:** `feat(sdk): add python client with backoff`.

#### M2-T6: CLI + presets [P]

**What:** `jeba` CLI with `--preset`, `--predict`, `--serve`; presets `router/guard/moderation/triage/email`.
**Where:** `src/jeba/cli.py`.
**Depends on:** M2-T1 · **Requirement:** SERVE-03, SERVE-04, SERVE-07.
**Done when:** `jeba "text" --preset triage --predict` prints an answer; presets expand to questions.
**Tests:** unit (preset expansion) + smoke · **Gate:** full · **Commit:** `feat(cli): add predict CLI and presets`.

#### M2-T7: `schemas.py` decide() [P]

**What:** Convert JSON Schema / pydantic models into decision primitives; `return_details` passthrough.
**Where:** `src/jeba/schemas.py`.
**Depends on:** M2-T1 · **Requirement:** SERVE-08, CAL-05.
**Done when:** a sample schema yields valid questions; unknown schema fails clearly.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(schemas): add decide() from json schema`.

#### M2-T8: repointed-Jev-client e2e

**What:** End-to-end test where a captured Jev client payload is sent to a live jeba server; compare shape.
**Where:** `tests/e2e/test_jev_client.py`.
**Depends on:** M2-T3, M2-T4, M2-T5, M2-T6, M2-T7 · **Requirement:** NFR-X01, NFR-X02.
**Done when:** e2e passes; canonical shape unchanged by any extension.
**Tests:** e2e · **Gate:** full · **Commit:** `test(e2e): verify repointed jev client`.

---

### M3 — Local Encoder Backend

#### M3-T1: script/language detection + checkpoint selection

**What:** `Router.route(text) -> checkpoint_id`; language/script map; English vs multilingual.
**Where:** `src/jeba/router.py`.
**Depends on:** M2-T8 · **Requirement:** ROUTE-01, ROUTE-02, ROUTE-03, ROUTE-05.
**Done when:** correct checkpoint chosen across Latin/Cyrillic/CJK/Arabic samples; 100+ language list; routing overhead measured < 0.5 ms.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(router): add script and language routing`.

#### M3-T2: checkpoint lifecycle

**What:** `preload`, `max_loaded`, LRU `evict`, `unload`, `attach`.
**Where:** `src/jeba/router.py` (modify).
**Depends on:** M3-T1 · **Requirement:** ROUTE-04.
**Done when:** `max_loaded=1` evicts correctly; no leaked references.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(router): add checkpoint lifecycle management`.

#### M3-T3: confidence derivation

**What:** `confidence(probabilities)` + `return_details` support.
**Where:** `src/jeba/calibration.py`.
**Depends on:** M2-T8 · **Requirement:** CAL-03, CAL-05, PRIM-05, EXT-05.
**Done when:** confidence ∈ [0,1], monotone with concentration; details returned additively.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(calibration): derive confidence from distributions`.

#### M3-T4: agent forward pass + batching

**What:** Single forward pass path, `sort_by_length`, `predict_batch`.
**Where:** `src/jeba/agent.py`.
**Depends on:** M3-T3 · **Requirement:** BACK-07, NFR-P05.
**Done when:** batch results align to inputs; sorted batching improves throughput vs naive.
**Tests:** unit + benchmark · **Gate:** full · **Commit:** `feat(agent): add single-pass batching`.

#### M3-T5: encoder backend (3 heads)

**What:** `EncoderBackend` implementing `Backend` with three task heads (noul/choice/score).
**Where:** `src/jeba/backends/encoder.py`.
**Depends on:** M3-T1, M3-T4 · **Requirement:** BACK-03, BACK-06, PRIM-01, PRIM-02, PRIM-03.
**Done when:** all primitives answered offline; contract tests pass; no network/key.
**Tests:** contract + integration (offline) · **Gate:** full · **Commit:** `feat(backends): add encoder backend with three heads`.

#### M3-T6: hooks

**What:** Hook registry for `on_predict_start/end`, `on_route`, `on_load`, `on_evict`, `on_error`; `on_error` on raise.
**Where:** `src/jeba/hooks.py` (+ wiring in `agent.py`).
**Depends on:** M3-T4 · **Requirement:** EXT-01, EXT-02.
**Done when:** each hook fires; raising hook triggers `on_error` and serving continues; contract shape unchanged.
**Tests:** unit + hook-contract (`tests/test_hooks_api.py`) · **Gate:** full · **Commit:** `feat(hooks): add prediction and lifecycle hooks`.

#### M3-T7: predict_batch extension [P]

**What:** Expose `predict_batch` via agent API and extension endpoint.
**Where:** `src/jeba/agent.py`, `src/jeba/serve.py` (modify).
**Depends on:** M3-T4 · **Requirement:** BACK-07, EXT-04.
**Done when:** batch of states returns aligned answers over HTTP.
**Tests:** integration · **Gate:** full · **Commit:** `feat(agent): expose batch prediction`.

#### M3-T8: offline multilingual integration

**What:** Network-disabled run across languages; verify routing + answers + fallbacks.
**Where:** `tests/integration/test_offline_multilingual.py`.
**Depends on:** M3-T5, M3-T6 · **Requirement:** ROUTE-05, BACK-06, NFR-S03, NFR-S04, NFR-P02.
**Done when:** baseline (untrained) checkpoint returns valid shapes across 100+ language samples offline.
**Tests:** integration · **Gate:** full · **Commit:** `test(integration): offline multilingual smoke`.

---

### M4 — Training & Calibration

#### M4-T1: synthetic data generation

**What:** `generate_data.py` → deterministic JSONL per primitive, streamed to disk.
**Where:** `training/generate_data.py`.
**Depends on:** M3-T5 · **Requirement:** TRAIN-01, TRAIN-07.
**Done when:** same seed → identical file; records include type/input/target.
**Tests:** unit (determinism) · **Gate:** quick · **Commit:** `feat(training): add deterministic data generation`.

#### M4-T2: LoRA/QLoRA fine-tuning

**What:** `finetune_rlcd.py` training loop with quantized base, LoRA, grad checkpointing/accumulation.
**Where:** `training/finetune_rlcd.py`.
**Depends on:** M4-T1 · **Requirement:** TRAIN-02, TRAIN-06, NFR-R05.
**Done when:** completes on RTX 3060 12GB without OOM; saves checkpoint + hyperparams.
**Tests:** integration (short run) · **Gate:** full · **Commit:** `feat(training): add LoRA fine-tuning`.

#### M4-T3: RLCD proper-scoring loss

**What:** RLCD training against a strictly proper scoring rule over primitive distributions.
**Where:** `training/finetune_rlcd.py` (modify).
**Depends on:** M4-T2 · **Requirement:** CAL-01, TRAIN-03.
**Done when:** loss is proper-scoring; probabilities normalized.
**Tests:** unit (loss properties) · **Gate:** full · **Commit:** `feat(training): add RLCD proper-scoring objective`.

#### M4-T4: temperature/calibration fitting

**What:** `fit_calibration.py` minimizing ECE on a held-out split.
**Where:** `training/fit_calibration.py`.
**Depends on:** M4-T3 · **Requirement:** CAL-02, CAL-04, TRAIN-04, NFR-C06.
**Done when:** ECE reduces vs baseline; test split untouched during fit.
**Tests:** unit + evaluation · **Gate:** full · **Commit:** `feat(training): fit calibration temperature`.

#### M4-T5: evaluation harness [P]

**What:** Accuracy/ECE/latency per primitive + language; public probes (MASSIVE/XNLI/typed-decisions).
**Where:** `benchmarks/`, `training/evaluate.py`.
**Depends on:** M4-T1 · **Requirement:** TRAIN-05, OPS-06.
**Done when:** report artifacts reproducible from committed commands.
**Tests:** evaluation · **Gate:** full · **Commit:** `feat(benchmarks): add accuracy, ECE, latency harness`.

#### M4-T6: reproducibility configs

**What:** Seed + config capture for every stage; documented tolerances.
**Where:** `training/configs/`, docs.
**Depends on:** M4-T4 · **Requirement:** TRAIN-06, NFR-C04.
**Done when:** re-run within documented tolerance; configs committed.
**Tests:** none (process) · **Gate:** build · **Commit:** `chore(training): add reproducible configs`.

---

### M5 — Ecosystem & Acceleration

#### M5-T1: ONNX backend [P]

**What:** `OnnxBackend` implementing `Backend` via onnxruntime.
**Where:** `src/jeba/backends/onnx.py`.
**Depends on:** M3-T5 · **Requirement:** BACK-04, OPS-01.
**Done when:** contract tests pass against ONNX; extra import-guarded.
**Tests:** contract + integration · **Gate:** full · **Commit:** `feat(backends): add onnx runtime backend`.

#### M5-T2: fast path (TileLang/CUDA graphs) [P]

**What:** Optional accelerated execution path behind the `fast` extra; graceful fallback.
**Where:** `src/jeba/fast.py` (planned), packaging.
**Depends on:** M3-T5 · **Requirement:** NFR-P01, NFR-C05, OPS-01, EXT-03.
**Done when:** latency improves on supported CUDA; falls back otherwise; router override (force checkpoint/language) honored additively.
**Tests:** benchmark (conditional-by-extra) · **Gate:** full · **Commit:** `perf: add optional fast path`.

#### M5-T3: MCP stdio server [P]

**What:** `jeba-mcp-server` exposing prediction tools over stdio.
**Where:** `src/jeba/mcp/`.
**Depends on:** M3-T5 · **Requirement:** OPS-02, SERVE-04.
**Done when:** an MCP host can list and call the tool; clean shutdown on disconnect.
**Tests:** integration (conditional-by-extra) · **Gate:** full · **Commit:** `feat(mcp): add stdio server`.

#### M5-T4: LangChain adapter [P]

**What:** `Runnable` adapter returning canonical primitives.
**Where:** `src/jeba/integrations/langchain.py`.
**Depends on:** M3-T5 · **Requirement:** OPS-03, EXT-02.
**Done when:** adapter invocation returns contract-shaped results; extra import-guarded.
**Tests:** integration (conditional-by-extra) · **Gate:** full · **Commit:** `feat(langchain): add runnable adapter`.

#### M5-T5: Docker + compose [P]

**What:** `Dockerfile` + `docker-compose.yml` running `jeba-serve`.
**Where:** `docker/`.
**Depends on:** M3-T5 · **Requirement:** OPS-04.
**Done when:** container answers `/v1/systemone`; compose up works.
**Tests:** smoke (container) · **Gate:** full · **Commit:** `build(docker): add image and compose`.

#### M5-T6: telemetry opt-out

**What:** Opt-out telemetry guard (`DO_NOT_TRACK=1`), disabled by default.
**Where:** `src/jeba/telemetry.py` (planned).
**Depends on:** M5-T1..T5 · **Requirement:** OPS-08, NFR-S05.
**Done when:** telemetry never blocks offline use; opt-out honored.
**Tests:** unit · **Gate:** quick · **Commit:** `feat(telemetry): add opt-out telemetry guard`.

---

### M6 — Proof & Release

#### M6-T1: benchmark report

**What:** Reproducible benchmark report vs MASSIVE/XNLI/typed-decisions.
**Where:** `benchmarks/report.md`, scripts.
**Depends on:** M5 · **Requirement:** OPS-06, NFR-D04.
**Done when:** report regenerated from committed commands.
**Tests:** evaluation · **Gate:** full · **Commit:** `docs(benchmarks): publish reproducible report`.

#### M6-T2: documentation site

**What:** Published docs site aggregating `docs/`.
**Where:** `docs/` config (e.g., mkdocs).
**Depends on:** M6-T1 · **Requirement:** NFR-D02, NFR-D01.
**Done when:** site builds and deploys.
**Tests:** none (build check) · **Gate:** build · **Commit:** `docs: add documentation site`.

#### M6-T3: Hugging Face release + model card

**What:** Release weights + model card; GitHub release.
**Where:** `docs/model-card.md`, release notes.
**Depends on:** M6-T2 · **Requirement:** OPS-07, NFR-D04.
**Done when:** weights + card published; release tagged.
**Tests:** none (release) · **Gate:** build · **Commit:** `chore(release): publish model card and weights`.

---

## Parallel Execution Map

```
Phase M0 (Sequential):  M0-T1 → M0-T2 → M0-T3 → M0-T4

Phase M1 (Partial parallel):
  M1-T1 → { M1-T2 , M1-T3 , M1-T4 , M1-T5 } [P] → M1-T6 → M1-T7

Phase M2 (Partial parallel):
  M2-T1 → { M2-T2 , M2-T5 , M2-T6 [P] , M2-T7 [P] }
         → M2-T3 → M2-T4 → M2-T8

Phase M3 (Partial parallel):
  { M3-T1 → M3-T2 , M3-T3 → M3-T4 } → M3-T5 → { M3-T6 , M3-T7 [P] , M3-T8 }

Phase M4 (Partial parallel):
  M4-T1 → { M4-T2 → M4-T3 → M4-T4 → M4-T6 , M4-T5 [P] }

Phase M5 (Parallel):  { M5-T1..M5-T5 [P] } → M5-T6

Phase M6 (Sequential): M6-T1 → M6-T2 → M6-T3
```

---

## Validation Checks (MANDATORY before execution)

### Check 1 — Task Granularity

| Task | Scope | Status |
| --- | --- | --- |
| M0-T1 | config + package skeleton | ✅ Granular (one cohesive scaffold) |
| M1-T1/T2/T3 | 1 primitive each | ✅ Granular |
| M1-T4 | envelope models + dispatcher | ✅ Cohesive single file |
| M2-T2 | 1 backend | ✅ Granular |
| M3-T5 | 1 backend (3 heads) | ✅ Cohesive single component |
| M4-T2/T3 | training loop, then objective | ⚠️ T3 modifies T2's file — intentional split (objective is independently testable) |
| M6-T3 | release activity | ✅ Granular (one release) |

### Check 2 — Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram Shows | Status |
| --- | --- | --- | --- |
| M1-T6 | M1-T2,T3,T4,T5 | converge into M1-T6 | ✅ Match |
| M2-T8 | M2-T3,T4,T5,T6,T7 | converge into M2-T8 | ✅ Match |
| M3-T5 | M3-T1,T4 | converge into M3-T5 | ✅ Match |
| M3-T6 | M3-T4 | after M3-T5 branch | ⚠️ M3-T6 body depends on T4; diagram routes via T5 — resolved: T6 depends on T4 only, shown after T5 for readability |
| M4-T4 | M4-T3 | chain | ✅ Match |
| M5-T6 | M5-T1..T5 | converge into M5-T6 | ✅ Match |

### Check 3 — Test Co-location Validation

| Task | Code Layer Created | Matrix Requires | Task Says | Status |
| --- | --- | --- | --- | --- |
| M1-T1..T3 | primitives | unit | unit | ✅ OK |
| M1-T4 | wire | unit + contract (via T6) | unit | ⚠️ contract added in T6 (allowed: T4 is unit-testable; T6 is the contract gate) |
| M2-T2 | backend | unit + integration | unit + integration | ✅ OK |
| M2-T3 | HTTP | integration | integration | ✅ OK |
| M3-T5 | backend | contract + integration | contract + integration | ✅ OK |
| M3-T6 | hooks | unit + hook-contract | unit + hook-contract | ✅ OK |
| M4-T1 | data script | unit | unit | ✅ OK |
| M5-T1 | backend | contract + integration | contract + integration | ✅ OK |

> Greenfield note: test types and gate commands are defined in `docs/testing.md`. They will be
> finalized in M0-T2 when the actual commands exist.

---

## Requirement Coverage

All 51 functional requirements map to at least one task above; the full matrix is in
`docs/requirements/traceability.md`. **Coverage: 51/51 mapped, 0 unmapped.**
