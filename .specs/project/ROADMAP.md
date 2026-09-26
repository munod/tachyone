# Roadmap

**Current Milestone:** Complete (M0–M6)
**Status:** M0–M6 COMPLETE; GPU training run done, numbers + adapters published (`v0.3.0`).
Open items live in `BACKLOG.md` (B-1 `nl`/`de`/`fr`/`it` ECE, B-7 public probes).

Milestones follow the approved phase plan. Each milestone is a shippable increment with
an explicit exit criterion. Detailed tasks live in `docs/tasks.md`; features live in
`.specs/features/`.

---

## M0 — Bootstrap (Phase 0)

**Goal:** A reproducible Python 3.12 + uv project skeleton with quality gates and license,
so every later change can be linted, typed, and tested.
**Exit:** `uv sync` succeeds, `uv run ruff check .`, `uv run pyright`, `uv run pytest`
all pass (even if tests are trivial), and CI runs the same on push/PR.

### Features

**Project scaffolding** - DONE
- `pyproject.toml` with `requires-python = ">=3.12,<3.13"`, extras, entry points
- `uv.lock`, `.python-version`, `src/jeba/` layout, `tests/`
- ruff + pytest + pyright dev dependencies and config

**Repo hygiene & CI** - DONE
- Apache-2.0 LICENSE, README, CONTRIBUTING, AGENTS.md
- `.github/workflows/ci.yml` (ruff, pyright, pytest)
- `.gitignore`, `.editorconfig`

---

## M1 — Wire Contract (Phase 1)

**Goal:** Freeze the Jev-compatible data contract before any backend exists.
**Exit:** golden contract tests pass for all primitive types and error shapes; a fake
deterministic backend lets us validate the full request/response cycle offline.

### Features

**Primitives & wire** - DONE
- `primitives.py`: `Choice` / `Score` / `Noul` pydantic models + `Answer` types
- `wire.py`: request/response models with exact Jev field names, limits, and errors
- Contract tests: envelope shape and field names, 255-option cap, 2–10 score levels, 401/422/429/529

---

## M2 — LLM Backend + Serving (Phase 2)

**Goal:** End-to-end system: a real HTTP server answering `/v1/systemone` via structured
outputs from existing LLMs, usable by existing Jev clients.
**Exit:** a Jev client repointed at `jeba-serve` returns correct answers; SDK + CLI work;
backend is swappable behind a stable interface.

### Features

**Pluggable backend interface + LLM backend** - DONE
- `backends/base.py` contract, `backends/llm.py` structured-output implementation

**Server, SDK, CLI** - DONE
- `serve.py` (`/v1/systemone`, `/predict`, `/predict/batch`, `/health`)
- Python SDK + `jeba` / `jeba-serve` entry points
- Retry/backoff on 429/529; API-key handling

---

## M3 — Local Encoder Backend (Phase 3)

**Goal:** Offline encoder backend (ModernBERT/mmBERT + 3 heads) with single forward pass,
language routing, and calibrated confidence.
**Exit:** all three primitives answered without network; router selects checkpoint by
script/language; confidence reflects probability distribution.

### Features

**Encoder backend & agent** - DONE
- `agent.py` single forward pass + batching + `sort_by_length`
- `backends/encoder.py` (3 heads), checkpoint loading/eviction, `max_loaded`

**Router & calibration** - DONE
- `router.py` script/language detection → checkpoint selection
- `calibration.py` confidence from distributions + temperature fitting hook

---

## M4 — Training & Calibration (Phase 4)

**Goal:** Reproducible data generation, LoRA/QLoRA fine-tuning, and RLCD proper-scoring
calibration within RTX 3060 12GB budget.
**Exit:** documented accuracy + ECE + latency on held-out sets, reproducible from scripts.

### Features

**Data & fine-tuning** - DONE
- `training/generate_data.py` synthetic JSONL; `training/finetune_rlcd.py`
- `training/fit_calibration.py` temperature/ECE fitting

**Evaluation** - DONE (harness) / OPEN (public probes)
- Accuracy/ECE/latency harness shipped (`training/evaluate.py`, `benchmarks/`)
- MASSIVE / XNLI / typed-decisions probes **not yet run or published**

---

## M5 — Ecosystem & Acceleration (Phase 5)

**Goal:** Practical integrations and speed paths without touching the core contract.
**Exit:** ONNX backend, MCP stdio server, LangChain adapter, Docker image all work.

### Features

**Acceleration** - DONE
- `backends/onnx.py`; TileLang fast path behind an extra
**Integrations** - DONE
- `mcp/` stdio server, `integrations/langchain.py`, `docker/` + compose

---

## M6 — Proof & Release (Phase 6)

**Goal:** Public credibility: benchmarks, documentation site, and a Hugging Face release.
**Exit:** reproducible benchmark report, docs site published, weights + model card released.

### Features

**Benchmarks & release** - DONE (synthetic report) / OPEN (public probes)
- `benchmarks/` publishes a reproducible synthetic report; MASSIVE / XNLI / typed-decisions
  comparisons not yet published
- Docs site, model card, GitHub release

---

## Future Considerations

- Additional encoder checkpoints (typed-decisions variant, larger multilingual)
- Provider registry for LLM backends (OpenAI-compatible, Anthropic, local llama.cpp)
- Streaming/multi-question batching optimizations; KV-free single-pass scheduling
- Optional telemetry (opt-out) for adoption metrics — only if it never blocks offline use
