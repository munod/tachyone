# Roadmap

> Canonical machine-facing roadmap lives in `.specs/project/ROADMAP.md`. This page is the
> human-facing view with exit criteria and risk notes. Task-level detail: `docs/tasks.md`.

**Current phase:** M2 LLM Backend + Serving ✅ complete — M3 Local Encoder next
**Delivery model:** one shippable increment per milestone; no fixed dates.

```mermaid
graph LR
    M0[M0 Bootstrap] --> M1[M1 Wire Contract]
    M1 --> M2[M2 LLM Backend + Serve]
    M2 --> M3[M3 Local Encoder]
    M3 --> M4[M4 Training + Calibration]
    M4 --> M5[M5 Ecosystem + Accel]
    M5 --> M6[M6 Proof + Release]
```

## M0 — Bootstrap

**Goal:** Reproducible Python 3.12 + uv project with quality gates and license.
**Depends on:** nothing.
**Exit criteria:**
- `uv sync` installs from `uv.lock` on Python 3.12.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest` all pass.
- CI runs the same gates on push and PR.
- Apache-2.0 `LICENSE`, `README.md`, `CONTRIBUTING.md`, `AGENTS.md` present.

**Risks:** Python version drift (see B-001); uv lockfile churn.

## M1 — Wire Contract

**Goal:** Freeze the Jev-exact contract and primitives before backends.
**Depends on:** M0.
**Exit criteria:**
- Primitives validated with limits (choice ≤255, score 2–10 levels).
- `wire.py` matches Jev field names/types; golden fixtures pass.
- All documented error shapes (401/422/429/529) reproduce.
- A `FakeBackend` answers the full cycle offline.

**Risks:** misreading Jev edge behavior; mitigated by golden fixtures and `docs/protocol.md`.

## M2 — LLM Backend + Serving

**Goal:** End-to-end product using structured outputs of existing LLMs.
**Depends on:** M1.
**Exit criteria:**
- `POST /v1/systemone` served by FastAPI; SDK + CLI work.
- A real Jev client repointed at `jeba-serve` returns correct answers.
- Backend is swappable behind `backends/base.py`; no wire change.
- Retry/backoff verified for 429/529.

**Risks:** structured-output reliability across providers (OD-1); provider drift.

## M3 — Local Encoder Backend

**Goal:** Offline single-pass encoder with routing and calibrated confidence.
**Depends on:** M2 (server reused), M4 (for a trained checkpoint — can start with base weights).
**Exit criteria:**
- All three primitives answered offline, no key, no network.
- Router selects English vs multilingual checkpoint with documented overhead.
- Confidence derived from distribution; `return_details` exposes probabilities.
- Batching (`predict_batch`, `sort_by_length`) and lifecycle (`preload`, `max_loaded`, eviction) work.

**Risks:** base-model quality before tuning; VRAM pressure with two checkpoints.

## M4 — Training & Calibration

**Goal:** Reproducible training and calibration within 12GB VRAM.
**Depends on:** M3 (architecture), M1 (data contract).
**Exit criteria:**
- `generate_data.py` → deterministic JSONL.
- LoRA/QLoRA fine-tuning completes on RTX 3060 12GB.
- RLCD proper-scoring training produces calibrated probabilities.
- Temperature fitting reduces ECE on a held-out split.
- Accuracy/ECE/latency report reproducible from committed scripts.

**Risks:** OOM (B-002); small calibration sets; synthetic-data bias.

## M5 — Ecosystem & Acceleration

**Goal:** Integration surface and speed paths, contract unchanged.
**Depends on:** M3, M4.
**Exit criteria:**
- ONNX backend + optional fast path (TileLang/CUDA graphs) behind extras.
- MCP stdio server, LangChain adapter, Docker/compose all work.
- Extras isolate optional dependencies; graceful fallback when unavailable.

**Risks:** ONNX op coverage; CUDA-graph constraints (OD-4).

## M6 — Proof & Release

**Goal:** Public credibility.
**Depends on:** M5.
**Exit criteria:**
- Benchmark report comparable to MASSIVE / XNLI / typed-decisions.
- Docs site published.
- Hugging Face weights + model card and a GitHub release.

**Risks:** benchmark comparability; weights distribution model (OD-3).

## Milestone → feature mapping

| Milestone | Feature spec |
| --- | --- |
| M0, M1 | `.specs/features/wire-contract/` |
| M2 | `.specs/features/llm-backend/` |
| M3 | `.specs/features/encoder-backend/` |
| M4 | `.specs/features/training-calibration/` |
| M5, M6 | `.specs/features/ecosystem/` |

## Future considerations

- Provider registry for LLM backends (OpenAI-compatible, local llama.cpp).
- Additional checkpoints (typed-decisions variant, larger multilingual).
- Streaming multi-question responses.
- Optional opt-out telemetry (only if it never blocks offline use).
