# jeba

> **Local-first, multilingual System One decision engine that speaks the TypeSafe Jev `/v1/systemone` protocol.**

jeba answers atomic structured questions — `choice`, `score`, and `noul` — and returns typed
values with probabilities and calibrated confidence. Point an existing Jev client at a jeba
server and it works unchanged; run the local encoder backend for fully offline inference.

**Status: `v0.2.0` released.** All milestones M0–M6 are complete, plus the post-M6
**B-1 multilingual quality** work (localized data, per-`(primitive, language)` temperature) and
**B-2 fast path** (per-shape CUDA graphs with bf16 weights). The wire contract, an
OpenAI-compatible LLM backend, a local encoder (ModernBERT/mmBERT + LoRA), an ONNX backend,
FastAPI serving, an SDK/CLI, MCP + LangChain integrations, a training pipeline, and a docs site
all ship. LoRA adapters are published on the Hugging Face Hub. See the
[CHANGELOG](CHANGELOG.md) for releases, next steps in
[`.specs/project/BACKLOG.md`](.specs/project/BACKLOG.md), and
[`.specs/project/STATE.md`](.specs/project/STATE.md) for decisions and blockers.

---

## Why jeba

Hosted decision APIs are fast but remote, closed, and metered; autoregressive LLMs are
local-capable but slow and poorly calibrated for atomic judgments. jeba sits in between:

- **Drop-in** — same request/response as Jev; repoint the client and keep your code.
- **Local-first** — the base install runs offline with no API key; the HTTP server is one mode, not the only one.
- **Fast** — the local backend is non-autoregressive: one forward pass, no decoding loop; `JEBA_FAST=1` adds a CUDA-graph path (2.7× p50 on an RTX 3060).
- **Calibrated** — probabilities trained with strictly proper scoring (RLCD) plus per-`(primitive, language)` temperature fitting.
- **Multilingual** — mmBERT-based checkpoint covering 100+ languages via automatic script/language routing.
- **Additive** — router control, hooks, `predict_batch`, MCP, and LangChain extend the contract without breaking it.

## How it compares

| | Jev | Laya | Needle | **jeba** |
| --- | --- | --- | --- | --- |
| Wire contract | `/v1/systemone` (hosted) | `/v1/systemone` (self-hosted) | Own API | **Jev-exact, self-hosted** |
| Hosted dependency | Required | None | None | **None in core** |
| LLM backend | — | No | No | **Yes (optional)** |
| Encoder backend | Own model | Yes | Yes (2-bit) | **Yes (ModernBERT/mmBERT + LoRA)** |
| ONNX backend | No | Yes | Yes | **Yes (`onnx` extra)** |
| MCP / LangChain | No | Yes | No | **Yes** |
| Multilingual | Yes | Yes | Partial | **Yes (100+ languages)** |
| License | Proprietary service | Apache-2.0 | Open | **Apache-2.0** |

## Install & quickstart

```bash
uv sync                 # core: offline, no key, no heavy deps
uv sync --extra serve   # + FastAPI server (+ integration/e2e test deps)

# Answer a question from the CLI (offline, model-free):
uv run jeba --predict --preset triage --backend fake "refund please"

# The local encoder (downloads base weights + LoRA adapters, needs the train extra):
uv sync --extra train
uv run jeba --predict --preset triage --backend encoder "Quero cancelar minha assinatura agora"

# Run the HTTP server:
uv run jeba-serve
```

```python
# Python SDK
from jeba import JebaClient
from jeba.primitives import ScoreQuestion

client = JebaClient("http://127.0.0.1:8000")
result = client.system_one(
    "This product is amazing!",
    {"sentiment": ScoreQuestion(
        instructions="Rate the sentiment.",
        criteria=["very negative", "negative", "neutral", "positive", "very positive"],
    )},
)
print(result.answers["sentiment"].score)
```

```bash
# Drop-in: point an existing Jev client at jeba
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":"hello","model":"jeba-latest","questions":{"q":{"type":"noul","instructions":"Is this a greeting?"}}}'
```

## Local encoder & published models

`JEBA_BACKEND=encoder` (default, offline once cached) loads `answerdotai/ModernBERT-large` or
`jhu-clsp/mmBERT-base` and applies the published LoRA adapter. Override with
`JEBA_ADAPTERS="jeba-en=acme/tuned-en,jeba-multi="`; force cache-only with `JEBA_OFFLINE=1`.
On a CUDA device, `JEBA_FAST=1` opts into the per-shape CUDA-graph forward (bf16 weights) with
graceful fallback.

- Adapters: [`munod/jeba-en`](https://huggingface.co/munod/jeba-en) ·
  [`munod/jeba-multi`](https://huggingface.co/munod/jeba-multi)
- Measured on a single RTX 3060 12GB (full tables: [`benchmarks/report.md`](benchmarks/report.md)):

  | Checkpoint | Overall | `choice` | `noul` | `score` | ECE |
  | --- | --- | --- | --- | --- | --- |
  | English (ModernBERT-large + LoRA + choice head) | 0.781 | 0.708 | 0.744 | 0.892 | 0.077 |
  | Multilingual (mmBERT-base + LoRA + choice head) | 0.711 | 0.734 | 0.716 | 0.682 | 0.073 |

  The dedicated `choice` head (L-002) and localized per-record-RNG data (B-1) lifted multilingual
  `choice` from ~0.25 (chance) to 0.73. Per-language calibration for `es`/`nl`/`de` and
  multilingual `score` remain the next targets. The CUDA-graph fast path (`JEBA_FAST=1`) improves
  p50 10.3 → 3.8 ms with no top-label changes.

## Architecture at a glance

```mermaid
graph LR
    C["Jev client / SDK / CLI"] --> S["serve.py<br/>/v1/systemone"]
    S --> W["wire.py<br/>frozen contract"]
    W --> B["backends/base.py"]
    B --> E["encoder.py<br/>local + LoRA"]
    B --> L["llm.py<br/>structured outputs"]
    B --> O["onnx.py<br/>portable"]
    E --> R["router.py<br/>language routing"]
    E --> A["agent.py<br/>single-pass batching"]
    E --> H["hooks.py<br/>observability"]
    A --> K["calibration.py<br/>confidence"]
```

Full detail: [`docs/architecture.md`](docs/architecture.md) · Contract: [`docs/protocol.md`](docs/protocol.md).

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest                       # contract + unit + integration + e2e
uv run mkdocs build --strict        # docs site (uv sync --group docs)
```

## Roadmap

| Phase | Milestone | Status |
| --- | --- | --- |
| M0 | Bootstrap | ✅ |
| M1 | Wire Contract | ✅ |
| M2 | LLM Backend + Serve | ✅ |
| M3 | Local Encoder | ✅ |
| M4 | Training & Calibration | ✅ |
| M5 | Ecosystem & Acceleration | ✅ |
| M6 | Proof & Release | ✅ (`v0.1.0`) |
| B-1 | Multilingual quality + per-language temperature | ✅ (`v0.2.0`) |
| B-2 | Fast path (CUDA graphs) | ✅ (`v0.2.0`) |

Details: [`docs/roadmap.md`](docs/roadmap.md) · Tasks: [`docs/tasks.md`](docs/tasks.md).

## Documentation map

| Doc | Contents |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | Vision, personas, use cases, success metrics, non-goals |
| [`docs/protocol.md`](docs/protocol.md) | The `/v1/systemone` contract |
| [`docs/architecture.md`](docs/architecture.md) | Components, flows, backend strategy, hooks |
| [`docs/adr/`](docs/adr/) | Architecture decision records |
| [`docs/requirements/`](docs/requirements/) | Functional, non-functional, traceability |
| [`docs/testing.md`](docs/testing.md) | Contract/parity tests, gates, benchmarks |
| [`docs/training.md`](docs/training.md) | Data generation, LoRA/QLoRA, RLCD, calibration |
| [`docs/huggingface.md`](docs/huggingface.md) | Publishing and loading adapters |
| [`docs/release.md`](docs/release.md) | Release checklist |
| [`docs/model-card.md`](docs/model-card.md) | Hugging Face model card |
| [`benchmarks/report.md`](benchmarks/report.md) | Reproducible benchmark report |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes (Keep a Changelog) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Conventions, gates, contract-test rule |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Short version: conventional commits, one logical
change per commit, English docs/PRs, and **any public API change must update the contract test
in the same commit**.

## License

[Apache-2.0](LICENSE).
