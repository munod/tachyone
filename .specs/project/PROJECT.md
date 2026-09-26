# jeba

**Vision:** jeba is a local-first, multilingual decision engine that answers atomic
`choice` / `score` / `noul` questions in a single forward pass and speaks the exact
TypeSafe Jev `/v1/systemone` wire protocol, so any existing Jev client can be repointed
at it unchanged.

**For:** Python application developers, agent builders, and edge/on-prem teams who need
calibrated structured decisions without sending data to a hosted LLM API.

**Solves:** Hosted "System One" decision APIs are fast but remote, closed, and paid per
call. Autoregressive LLMs are local-capable but slow, expensive, and poorly calibrated
for atomic decisions. jeba gives you the Jev contract you already build against, running
on your own box, with an optional LLM backend today and a trained encoder backend next.

> Status: **implemented and released (`v0.3.0`)**. All milestones M0–M6 are complete, plus the
> post-M6 work: multilingual quality (B-1, LoRA rank 16 → 64), CUDA-graph fast path (B-2),
> confidence thresholding / System-2 handoff (B-3), and input-noise robustness (B-4). The trained
> adapters are published (`munod/jeba-en`, `munod/jeba-multi`) and the `v0.3.0` GitHub Release
> exists; `nl` per-language calibration remains the open quality target. See
> `.specs/project/STATE.md` and `.specs/project/BACKLOG.md`.

## Goals

- **G1 — Wire parity (measurable):** 100% pass on a golden contract test suite where the
  same Jev request produces structurally identical jeba responses for all three primitive
  types, including documented 401 / 422 / 429 / 529 error shapes.
- **G2 — End-to-end in one phase (measurable):** a working `POST /v1/systemone` server on
  the LLM backend, plus Python SDK and CLI, that answers a real repointed Jev client in
  Phase 2 — before any model training.
- **G3 — Local encoder (measurable):** a ModernBERT/mmBERT-based backend that produces all
  three primitives in a single forward pass, with measured p50 latency and calibration
  (ECE) reported on a public benchmark set.
- **G4 — Multilingual (measurable):** correct primitive output for inputs across 100+
  languages via script/language routing, verified on a held-out multilingual suite.
- **G5 — Zero hosted dependency in core (measurable):** installing from source
  (`git clone … && uv sync`, base extra) must import, answer offline, and never require a
  network call or API key.
- **G6 — Extensions without breakage (measurable):** router control, hooks, and
  `predict_batch` are additive and never alter the canonical `/v1/systemone` shape.

## Tech Stack

**Core:**

- Language: Python 3.12 (`>=3.12,<3.13`) — pinned; torch/transformers do not yet support 3.14
- Package/environment manager: `uv` (pyproject.toml + uv.lock)
- Data validation: pydantic v2
- HTTP server (optional extra `serve`): FastAPI + uvicorn
- Model layer (optional extra `train` / encoder): torch + transformers
- Lint/format: ruff · Types: pyright · Tests: pytest · CI: GitHub Actions

**Key dependencies:** pydantic, FastAPI/uvicorn (`serve`), torch/transformers (`train`),
onnxruntime (`onnx`), model-context-protocol (`mcp`), langchain-core (`langchain`).

## Scope

**v1 / near-term includes (Phases 0–2):**

- Bootstrapped Python 3.12 + uv project, ruff/pytest/pyright, CI, Apache-2.0 license
- Canonical primitives (`choice` / `score` / `noul`) and Jev-exact `wire.py`
- `POST /v1/systemone` server, Python SDK, CLI, repointed-Jev-client e2e proof
- Pluggable LLM backend using structured outputs of existing LLMs

**Later (Phases 3–6):**

- Local encoder backend (ModernBERT/mmBERT + 3 heads), script/language router, calibration
- Data generation + LoRA/QLoRA + RLCD proper-scoring training and temperature fitting
- ONNX / fast-path, MCP, LangChain, Docker; benchmarks and Hugging Face release

**Explicitly out of scope (for now):**

- Hosted SaaS control plane, billing, or multi-tenant auth (jeba is local-first)
- Autoregressive generation / free-form chat — jeba answers atomic structured questions only
- Reimplementing the Jev SDK's hosted client, or depending on `typesafe-sdk` in core
- A GUI/web console (CLI + HTTP + SDK only)
- Training a foundation model from scratch (we fine-tune existing encoders)

## Constraints

- **Technical:** Python must be 3.12 (torch/transformers lack 3.14 support). Target dev
  hardware is a single RTX 3060 12GB, 12 CPUs, 31GB RAM — training must fit (LoRA/QLoRA).
- **Contract:** `/v1/systemone` must remain drop-in compatible with Jev; public API changes
  require updating the contract test in the same commit (Laya repo convention).
- **Dependency:** the core package must never require a hosted service; LLM backend is an
  optional extra.
- **Timeline:** none fixed; phased delivery with a shippable increment per milestone.
- **Resources:** small team / solo; bias to lean, low-ceremony process.

## Decisions vs open questions

- **Decided (locked, do not reopen):** phased strategy (LLM now, encoder later);
  Jev drop-in with additive extensions; multilingual from early (mmBERT-base); local-first;
  Python 3.12 + uv + ruff/pytest/pyright + GitHub Actions; Apache-2.0.
- **Open (see STATE.md):** exact LLM provider abstraction surface, telemetry default,
  packaging of model weights, ONNX/TileLang sequencing.

See `docs/overview.md` for the full vision and `docs/roadmap.md` for milestones.
