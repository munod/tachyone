# Architecture

**Status:** Design (approved decisions locked; open items flagged).
**Related:** `docs/protocol.md`, `docs/adr/`, `docs/requirements/`, `docs/training.md`.

This document defines **how Tachyone is built**. It is organized around one hard boundary —
the frozen wire — and one flexible seam — the pluggable backend.

---

## Architectural principles

1. **Contract-first.** `wire.py` + `docs/protocol.md` are frozen before backends exist.
   Backends are interchangeable behind the seam; the wire never depends on which one runs.
2. **Local-first, offline-capable core.** Base install must answer with no network and no
   API key. Heavy/optional dependencies live behind pip extras.
3. **Additive extensions only.** Router control, hooks, batching, integrations never mutate
   the canonical request/response shape.
4. **Single forward pass where possible.** The local backend answers all primitives without
   autoregressive decoding; batching and length-sorting keep latency low.
5. **Calibration is a feature.** `confidence` is derived from calibrated distributions, not
   asserted; training optimizes strictly proper scoring rules.
6. **One logical change per commit; public API changes update the contract test in the same commit.**

---

## System context

```mermaid
graph TB
    subgraph Clients
        JC["Existing Jev client"]
        SDK["tachyone Python SDK"]
        CLI["tachyone CLI"]
        MCPC["MCP host"]
        LC["LangChain app"]
    end

    subgraph tachyone
        SRV["serve.py<br/>FastAPI /v1/systemone"]
        MCP["mcp/ stdio server"]
        INT["integrations/langchain.py"]
        WIRE["wire.py + primitives.py"]
        AGENT["agent.py<br/>forward pass, batching"]
        ROUTER["router.py<br/>script/language"]
        CAL["calibration.py<br/>confidence / temperature"]
        SCH["schemas.py<br/>decide()"]
    end

    subgraph Backends
        BA["backends/base.py"]
        BE["encoder.py<br/>ModernBERT/mmBERT"]
        BL["llm.py<br/>structured outputs"]
        BO["onnx.py<br/>runtime"]
    end

    JC --> SRV
    SDK --> SRV
    CLI --> SRV
    CLI --> AGENT
    MCPC --> MCP --> AGENT
    LC --> INT --> AGENT

    SRV --> WIRE
    MCP --> WIRE
    INT --> WIRE
    WIRE --> BA
    BA --> BE
    BA --> BL
    BA --> BO
    BE --> ROUTER
    BE --> AGENT
    AGENT --> CAL
    SCH --> WIRE
```

---

## Repository structure

```text
projeto_tachyone/
├── pyproject.toml            # uv project, py>=3.12,<3.13, extras, entry points
├── uv.lock                   # committed lockfile
├── .python-version           # 3.12
├── AGENTS.md  README.md  CONTRIBUTING.md  CHANGELOG.md  LICENSE (Apache-2.0)
├── src/tachyone/
│   ├── __init__.py           # public SDK surface
│   ├── primitives.py         # Choice / Score / Noul + Answer models (pydantic v2)
│   ├── wire.py               # /v1/systemone request/response + error mapping
│   ├── agent.py              # single forward pass, batching, sort_by_length
│   ├── router.py             # script/language → checkpoint selection + lifecycle
│   ├── calibration.py        # confidence derivation + temperature fitting
│   ├── handoff.py            # assess()/assess_response() System-2 thresholding
│   ├── hooks.py              # additive lifecycle hooks (never change the wire shape)
│   ├── presets.py            # named question presets + schema helpers
│   ├── schemas.py            # JSON Schema / pydantic → questions (decide())
│   ├── client.py             # SDK client for /v1/systemone
│   ├── cli.py                # `tachyone` entry point + presets
│   ├── serve.py              # FastAPI: /v1/systemone, /predict, /predict/batch, /health
│   ├── config.py             # env-var configuration + validation
│   ├── fast.py               # optional CUDA-graph fast path (TACHYONE_FAST)
│   ├── telemetry.py          # no-op opt-out guard (ADR-0011)
│   ├── backends/
│   │   ├── base.py           # Backend Protocol + PredictionResult
│   │   ├── fake.py           # model-free backend for tests/CI
│   │   ├── encoder.py        # local ModernBERT/mmBERT + 3 heads
│   │   ├── llm.py            # structured outputs of existing LLMs
│   │   └── onnx.py           # onnxruntime backend
│   ├── mcp/                  # stdio MCP server (`tachyone-mcp-server`)
│   └── integrations/
│       └── langchain.py      # Runnable adapter
├── training/
│   ├── generate_data.py      # synthetic JSONL generation
│   ├── finetune_rlcd.py      # LoRA/QLoRA + RLCD proper scoring
│   ├── fit_calibration.py    # temperature / ECE fitting
│   ├── predict.py            # batch predictions for calibration
│   ├── evaluate.py           # accuracy / ECE / latency report
│   └── package_hf.py         # Hub packaging (README, config, weights)
├── tests/                    # contract, unit, integration, e2e, conditional-by-extra
├── benchmarks/               # fast-path + report harness (public probes pending)
├── docker/                   # Dockerfile + compose.yaml
└── .github/workflows/ci.yml  # ruff + pyright + pytest + offline import + docs
```

**pip extras:** `serve`, `fast`, `onnx`, `langchain`, `mcp`, `train`.
**Entry points:** `tachyone`, `tachyone-serve`, `tachyone-mcp-server`.

---

## Components

### `primitives.py`

- **Purpose:** Canonical pydantic v2 models for the three primitives and their answers.
- **Location:** `src/tachyone/primitives.py`
- **Interfaces:**
  - `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion` (discriminated union `Question` on `type`)
  - `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer` (discriminated union `Answer`)
  - validators: choice 1..255 options, score 2..10 levels, probability key coverage.
- **Dependencies:** pydantic v2 only.
- **Boundary:** Pure data + validation. No I/O, no model knowledge.

### `wire.py`

- **Purpose:** Envelope models, backend dispatch, and error mapping for `/v1/systemone`.
- **Location:** `src/tachyone/wire.py`
- **Interfaces:**
  - `SystemOneRequest`, `SystemOneResponse`, `Usage`
  - `answer(request: SystemOneRequest, backend: Backend) -> SystemOneResponse`
  - error constructors for 401/422/429/529.
- **Dependencies:** `primitives`, `backends.base`.
- **Boundary:** Knows the contract, not the engine.

### `backends/base.py` (the seam)

- **Purpose:** Stable interface every engine implements.
- **Location:** `src/tachyone/backends/base.py`
- **Interfaces:**
  - `class Backend(Protocol):` `async def predict(self, questions, *, state, model, return_details=False) -> PredictionResult`
  - `PredictionResult(answers: dict[str, Answer], usage: Usage)`
- **Dependencies:** `primitives`.
- **Boundary:** The only thing `wire.py` couples to. Swap implementations freely.

### `backends/llm.py` (Phase 2)

- **Purpose:** Answer primitives via structured outputs of existing LLMs.
- **Location:** `src/tachyone/backends/llm.py`
- **Interfaces:** implements `Backend`; provider selection via config.
- **Dependencies:** optional extra (provider SDK/HTTP client).
- **Boundary:** Never required by core; no key/network needed unless selected.
- **Resolved:** OD-1 provider abstraction surface (ADR-0008).

### `backends/encoder.py` + `agent.py` (Phase 3)

- **Purpose:** Local single-forward-pass inference with three task heads.
- **Location:** `src/tachyone/backends/encoder.py`, `src/tachyone/agent.py`
- **Interfaces:**
  - `Agent.predict_batch(states, questions, *, sort_by_length=True)`
  - `Agent.preload(checkpoints)`, `Agent.unload(checkpoint)`
  - `EncoderBackend` implements `Backend`.
- **Dependencies:** torch/transformers, `router`, `calibration`, `agent`.
- **Reuses:** `router.py` for checkpoint choice; `calibration.py` for confidence.

### `router.py` (Phase 3)

- **Purpose:** Detect script/language and select the right checkpoint; manage lifecycle.
- **Location:** `src/tachyone/router.py`
- **Interfaces:**
  - `Router.route(text) -> checkpoint_id` (target overhead < 0.5 ms)
  - `Router.preload(...)`, `Router.attach(...)`, `Router.unload(...)`, `max_loaded`
- **Dependencies:** lightweight detection only (no model inference).
- **Boundary:** Routing decisions never change the wire shape.

### `calibration.py` (Phase 3–4)

- **Purpose:** Derive `confidence` from distributions; fit temperature to minimize ECE.
- **Location:** `src/tachyone/calibration.py`
- **Interfaces:**
  - `confidence(probabilities) -> float`
  - `fit_temperature(logits, labels) -> Temperature`
  - `apply_temperature(probabilities, temperature) -> probabilities`
- **Dependencies:** numpy/torch (extra `train`), pure-python fallback for the function itself.

### `schemas.py`

- **Purpose:** Turn JSON Schema / pydantic models into decision primitives.
- **Location:** `src/tachyone/schemas.py`
- **Interfaces:** `decide(schema, *, return_details=False) -> questions | result`.
- **Dependencies:** pydantic; JSON Schema parsing.
- **Reuses:** `primitives.py`.

### `serve.py`

- **Purpose:** HTTP surface.
- **Location:** `src/tachyone/serve.py`
- **Interfaces:**
  - `POST /v1/systemone` (canonical)
  - `POST /predict`, `POST /predict/batch`, `GET /health` (extensions)
- **Dependencies:** optional extra `serve` (FastAPI + uvicorn).
- **Boundary:** Transport + auth; delegates to `wire.answer`.

### `mcp/` and `integrations/langchain.py` (Phase 5)

- **Purpose:** Agent-framework integration.
- **Interfaces:** MCP stdio tools; LangChain `Runnable`.
- **Dependencies:** extras `mcp`, `langchain`.

### `cli.py`

- **Purpose:** `tachyone "text" --preset triage --predict` and `--serve`.
- **Interfaces:** presets `router`, `guard`, `moderation`, `triage`, `email`; flags `--predict`,
  `--serve`, `--preset`, `--backend`, `--model`, `--threshold`, `--list-presets`.
- **Dependencies:** argparse (stdlib); core.

### `config.py`

- **Purpose:** Env-var configuration, single source of truth.
- **Interface (env vars):** `TACHYONE_HOST`, `TACHYONE_PORT`, `TACHYONE_DEVICE`, `TACHYONE_PRELOAD`,
  `TACHYONE_MODELS`, `TACHYONE_MODELS_DIR`, `TACHYONE_ADAPTERS`, `TACHYONE_OFFLINE`, `TACHYONE_FAST`, `TACHYONE_THREADS`,
  `TACHYONE_API_KEY`, `TACHYONE_BACKEND`, `TACHYONE_LLM_BASE_URL`, `TACHYONE_LLM_API_KEY`, `TACHYONE_LLM_MODEL`,
  `TACHYONE_LLM_TIMEOUT`, `TACHYONE_LLM_RETRIES` (full table under [Configuration](#configuration)).
- **Boundary:** Never logs secrets; validates at startup.

---

## Execution flows

### Canonical request (encoder backend)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as serve.py
    participant W as wire.py
    participant R as router.py
    participant A as agent.py
    participant E as backends/encoder.py
    participant K as calibration.py

    C->>S: POST /v1/systemone (Bearer)
    S->>S: auth check (401 on failure)
    S->>W: parse SystemOneRequest (422 on failure)
    W->>E: predict(questions, state, model)
    E->>R: route(state) -> checkpoint
    R-->>E: checkpoint_id
    E->>A: single forward pass (batch, sort_by_length)
    A->>K: derive confidence from distributions
    K-->>A: confidence
    A-->>E: answers
    E-->>W: PredictionResult(answers, usage)
    W-->>S: SystemOneResponse
    S-->>C: 200 JSON
```

### Backend selection

```mermaid
flowchart TD
    Req["Request arrives"] --> Cfg{"TACHYONE_BACKEND?"}
    Cfg -->|"llm (default; needs TACHYONE_LLM_*)"| LLM["LLMBackend"]
    Cfg -->|"encoder (local, offline once cached)"| Enc["EncoderBackend"]
    Cfg -->|"onnx"| ONNX["OnnxBackend"]
    Cfg -->|"fake (model-free, tests/CI)"| Fake["FakeBackend"]
    Enc --> Wire["wire.answer -> SystemOneResponse"]
    LLM --> Wire
    ONNX --> Wire
    Fake --> Wire
```

All four paths must pass the same contract tests. The wire output is identical in shape.
ADR-0004 intended the local encoder to become the default from M3; that switch was never made —
`DEFAULT_BACKEND` is still `llm` (see `docs/adr/README.md` → Implementation notes).

---

## Data models

The wire models are specified in `docs/protocol.md`. Internally:

```python
# Illustrative only.
class PredictionResult(BaseModel):
    answers: dict[str, Answer]
    usage: Usage

class CheckpointInfo(BaseModel):
    id: str                 # e.g. "tachyone-en", "tachyone-multi"
    languages: list[str]    # ["en"] or ["*"] for multilingual
    context: int
    size_params: int

class RouteDecision(BaseModel):
    checkpoint_id: str
    detected_script: str
    detected_language: str | None
```

---

## Backend strategy (decided)

| Phase | Backend | Role | Offline? |
| --- | --- | --- | --- |
| M2 | `llm.py` | Fast end-to-end, structured outputs | No (unless local provider) |
| M3 | `encoder.py` | Default local single-pass engine | Yes |
| M5 | `onnx.py` | Portable/accelerated runtime | Yes |

The seam is `backends/base.py`. Adding a backend must not change `wire.py` or `docs/protocol.md`.
See `docs/adr/ADR-0002-pluggable-backend-phasing.md`.

---

## Routing & multilingual strategy (decided)

- Detect script/language cheaply (no model forward pass), target overhead < 0.5 ms.
- Select English (`ModernBERT-large`-class) vs multilingual (`mmBERT-base`, 100+ languages).
- Lifecycle: `preload`, `max_loaded`, LRU `evict`, `unload`, `attach`.
- Fallback to multilingual when language is uncertain.
- Router decisions are exposed via hooks and optional extension fields, never in the canonical answer.

---

## Calibration & confidence (decided)

- Training uses **RLCD** against strictly proper scoring rules → calibrated probabilities.
- `confidence` for `choice`/`score` is a monotone function of the distribution (typically the
  selected mass). `noul` returns a single probability with no separate confidence.
- Temperature fitting minimizes ECE on a held-out calibration split.
- Distributions are always part of the canonical answer; `return_details=True` is accepted for
  API stability and is currently a no-op (no request flag or SDK argument exposes it).
- ECE target and method: `docs/training.md`.

---

## Hooks (additive extension)

Signature pattern: a callable or object registered for lifecycle events.

| Hook | Fires when |
| --- | --- |
| `on_predict_start` | Before inference for a request |
| `on_predict_end` | After inference completes |
| `on_route` | After a routing decision |
| `on_load` | A checkpoint is loaded |
| `on_evict` | A checkpoint is evicted |
| `on_error` | An error occurs during inference |

**Rule:** hooks may observe and log; they must not alter the canonical `/v1/systemone` shape.
A raising hook triggers `on_error` and serving continues. This mirrors Laya's public-API
contract (`tests/test_hooks_api.py` in Laya is the inspiration for Tachyone's hook contract test).

---

## Error handling strategy

| Scenario | Handling | Wire impact |
| --- | --- | --- |
| Invalid key | auth dependency | 401 |
| Malformed body | pydantic validation | 422 |
| Too many options / bad levels | primitive validators | 422 |
| Rate limit | upstream provider pass-through (`llm` backend); SDK backs off | 429 |
| Overload | upstream provider pass-through (`llm` backend); SDK backs off | 529 |
| Backend/internal failure | `wire.BackendError` (code `internal_error`) | 500 |
| Hook raises | `on_error`, continue | none |
| Extra missing | import-guard with actionable message | runtime error naming extra |

---

## Configuration

All configuration via environment variables (prefix `TACHYONE_`), documented in `config.py`:

| Var | Default | Purpose |
| --- | --- | --- |
| `TACHYONE_HOST` | `127.0.0.1` | Server bind host |
| `TACHYONE_PORT` | `8000` | Server port |
| `TACHYONE_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `TACHYONE_BACKEND` | `llm` | `llm` / `encoder` / `onnx` / `fake` (ADR-0004 intended `encoder` from M3; never switched — see `docs/adr/README.md`) |
| `TACHYONE_MODELS` | built-in ids | Available checkpoints |
| `TACHYONE_MODELS_DIR` | `~/.cache/tachyone/models` | Local weights cache (see ADR-0010) |
| `TACHYONE_ADAPTERS` | built-in adapters | `id=repo|path` overrides; empty value disables the adapter |
| `TACHYONE_OFFLINE` | unset | `1` = cache-only, no network (`local_files_only` on every Hub call) |
| `TACHYONE_PRELOAD` | empty | Checkpoints to load at startup |
| `TACHYONE_FAST` | unset | `1` = opt into the CUDA-graph fast path when a CUDA device is present |
| `TACHYONE_THREADS` | `0` (runtime decides) | CPU thread budget |
| `TACHYONE_API_KEY` | unset | Enables Bearer auth; unset = auth disabled (dev only) |
| `TACHYONE_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint for the `llm` backend |
| `TACHYONE_LLM_API_KEY` | unset (falls back to `OPENAI_API_KEY`) | Provider key for the `llm` backend |
| `TACHYONE_LLM_MODEL` | `gpt-4o-mini` | Model id for the `llm` backend |
| `TACHYONE_LLM_TIMEOUT` | `30.0` | Per-request timeout (s) |
| `TACHYONE_LLM_RETRIES` | `2` | Retry count (0–10) |
| `TACHYONE_TELEMETRY` | reserved | No telemetry exists (ADR-0011); the variable is a reserved no-op |
| `DO_NOT_TRACK` | unset | Honored unconditionally; `telemetry_enabled()` is always `False` |

No secrets are committed to the repository.

---

## Decisions vs open questions

**Decided (locked):** Jev drop-in; pluggable backend + phase order; multilingual mmBERT from M3;
local-first/offline core; Apache-2.0 + opt-out telemetry; Python 3.12 + uv.

**Resolved (all five):** OD-1 LLM provider surface (ADR-0008) · OD-2 telemetry default (ADR-0011) ·
OD-3 weights distribution (ADR-0010) · OD-4 ONNX vs TileLang sequencing (ADR-0012) ·
OD-5 non-contract extension/500 schema (ADR-0009). See `.specs/project/STATE.md`.
