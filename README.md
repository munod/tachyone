<p align="center">
  <img src="docs/assets/logo-wordmark.svg" alt="Tachyone — System 1 inference engine" width="440">
</p>

# Tachyone

> **Local-first, multilingual System One decision engine that speaks the TypeSafe Jev `/v1/systemone` protocol.**
>
> **LLMs generate text. Tachyone produces calibrated decisions. Don't ask a model to decide — ask Tachyone.**

Tachyone answers atomic structured questions — `choice`, `score`, and `noul` — and returns typed
values with probabilities and calibrated confidence. Point an existing Jev client at a Tachyone
server and it works unchanged; run the local encoder backend for fully offline inference.

**Status: `v0.5.0` released.** All milestones M0–M6 are complete, plus the post-M6
**B-1 multilingual quality** work (localized data, per-`(primitive, language)` temperature, and a
multilingual LoRA raised to rank 64), **B-2 fast path** (per-shape CUDA graphs with bf16 weights),
**B-3 confidence thresholding / System-2 handoff**, and **B-4
input-noise robustness** — plus **B-7 public probes** (typed-decisions, MASSIVE and XNLI with
licences, chance levels and reproduction commands in `benchmarks/probes.md`), **B-8
calibration-asset hardening** (an unusable adapter asset warns by name instead of degrading
silently), **B-10's LLM prompt fix** (probe `JSON ok` 0.083 → 0.889 with parsing kept strict),
the **B-5a multi-domain data refactor** (five committed domains, `per_domain` reporting and
byte-identical output for every existing config — both gates pass when re-measured on the
corrected labels), and **B-11 + B-12**, which derive *every* label in all three primitives from
the text it accompanies (ADR-0014 + ADR-0015: all datasets regenerated, both adapters republished,
and the label audit published beside the accuracy), and **B-5 / ADR-0016**, which ships the
per-domain `choice`-head bank and its deterministic gate, the optional `choice_head` request hint,
the frozen-trunk bank fitter, and republished `tachyone-en` as the **five-domain** adapter
(`choice` 1.000 and every domain ≥ 0.963, with the support `noul`/`score` trade published beside
it rather than summarized). The wire
contract, an OpenAI-compatible LLM backend, a local encoder
(ModernBERT/mmBERT + LoRA), an ONNX backend, FastAPI serving, an SDK/CLI, MCP + LangChain
integrations, a training pipeline, and a docs site all ship. LoRA adapters are published on the
Hugging Face Hub. See the
[CHANGELOG](CHANGELOG.md) for releases, next steps in
[`.specs/project/BACKLOG.md`](.specs/project/BACKLOG.md), and
[`.specs/project/STATE.md`](.specs/project/STATE.md) for decisions and blockers.

---

## Why Tachyone

Hosted decision APIs are fast but remote, closed, and metered; autoregressive LLMs are
local-capable but slow and poorly calibrated for atomic judgments. Tachyone sits in between:

- **Drop-in** — same request/response as Jev; repoint the client and keep your code.
- **Local-first** — the base install runs offline with no API key; the HTTP server is one mode, not the only one (the CLI's default backend is `llm`, so pass `--backend encoder` or `--backend fake` for a key-free run).
- **Fast** — the local backend is non-autoregressive: one forward pass, no decoding loop; `TACHYONE_FAST=1` adds a CUDA-graph path (2.7× p50 on an RTX 3060).
- **Calibrated** — probabilities trained with strictly proper scoring (RLCD) plus per-`(primitive, language)` temperature fitting.
- **Multilingual** — mmBERT-based checkpoint covering 100+ languages via automatic script/language routing.
- **Additive** — router control, hooks, `predict_batch`, MCP, and LangChain extend the contract without breaking it.

## How it compares

Mirrors the canonical table in [`docs/overview.md`](docs/overview.md#how-tachyone-compares).

| | Jev | Laya | Needle | **Tachyone** |
| --- | --- | --- | --- | --- |
| Wire contract | `/v1/systemone` (hosted) | `/v1/systemone` (self-hosted) | Own tool/embedding API | **Jev-exact, self-hosted** |
| Hosted dependency | Required | None | None (device engine) | **None in core** |
| LLM backend | — | No (encoder only) | No | **Yes (optional)** |
| Encoder backend | Own model | Yes | Yes (2-bit) | **Yes (ModernBERT/mmBERT + LoRA)** |
| ONNX backend | No | Yes | Yes | **Yes (`onnx` extra)** |
| MCP / LangChain | No | Yes | No | **Yes** |
| Multilingual | Yes | Yes (mmBERT) | Partial | **Yes (100+ routed, 6 trained)** |
| Fast path | Proprietary | No | Quantized | **CUDA graphs (`fast` extra)** |
| Extension model | n/a | Router, hooks | Grammar, telemetry | **Router, hooks, batch (additive)** |
| License | Proprietary service | Apache-2.0 | Open | **Apache-2.0** |

## Install & quickstart

```bash
uv sync                 # core: offline, no key, no heavy deps
uv sync --extra serve   # + FastAPI server (+ integration/e2e test deps)

# Answer a question from the CLI (offline, model-free):
uv run tachyone --predict --preset triage --backend fake "refund please"

# The local encoder (downloads base weights + LoRA adapters, needs the train extra):
uv sync --extra train
uv run tachyone --predict --preset triage --backend encoder "Quero cancelar minha assinatura agora"

# Run the HTTP server:
uv run tachyone-serve
```

```python
# Python SDK
from tachyone import TachyoneClient
from tachyone.primitives import ScoreQuestion

client = TachyoneClient("http://127.0.0.1:8000")
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
# Drop-in: point an existing Jev client at tachyone
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":"hello","model":"tachyone-latest","questions":{"q":{"type":"noul","instructions":"Is this a greeting?"}}}'
```

## Local encoder & published models

`TACHYONE_BACKEND=encoder` (local, offline once cached) loads `answerdotai/ModernBERT-large` or
`jhu-clsp/mmBERT-base` and applies the published LoRA adapter. Override with
`TACHYONE_ADAPTERS="tachyone-en=acme/tuned-en,tachyone-multi="`; force cache-only with `TACHYONE_OFFLINE=1`.
On a CUDA device, `TACHYONE_FAST=1` opts into the per-shape CUDA-graph forward (bf16 weights) with
graceful fallback.

- Adapters: [`munod/tachyone-en`](https://huggingface.co/munod/tachyone-en) ·
  [`munod/tachyone-multi`](https://huggingface.co/munod/tachyone-multi)
- Measured on a single RTX 3060 12GB (B-5b training on a single L4 23GB; full tables:
  [`benchmarks/report.md`](benchmarks/report.md)):

  | Checkpoint | Overall | `choice` | `noul` | `score` | ECE |
  | --- | --- | --- | --- | --- | --- |
  | English (ModernBERT-large + five-domain LoRA r=16 + choice-head bank), support split | **0.964** | **1.000** | 0.946 | 0.946 | 0.023 |
  | Multilingual (mmBERT-base + LoRA r=64 + fitted choice-head bank), five-domain split | **0.9975** | **0.992** | 1.000 | 1.000 | 0.001 |

  The English row is the support split; on the **five-domain** eval the same adapter scores
  **0.964** overall with every domain ≥ 0.963, where the previous support-only weights scored
  0.879. Its `noul`/`score` on support moved 0.992/0.978 → 0.946/0.946 — the five-domain trunk
  trades those two for `choice` 0.946 → 1.000 and four new domains.
  **B-5b (multilingual)** went the other way around: the previous adapter scores **0.561**
  zero-shot on the same five-domain rows (worst new domain 0.438), and the published artifact
  posts **0.9975** with every domain ≥ 0.993, gate strict **1.000**, all six languages at
  ECE ≤ 0.004, and 0.996 on held-out text — on the support-only *routed* split the product-level
  number moves 0.743 → **0.895**. The whole trade-off is tabled in
  [`docs/benchmarks.md`](docs/benchmarks.md).

  Every label comes from the text it accompanies (B-11 + B-12 / ADR-0014 + ADR-0015): `noul` from
  its phrase bank, `score` from the tone's level, `choice` from the option the state names, with
  defaults for the empty boundary states — so the audit shipped beside these tables reports **0
  contradictory rows** (the pre-B-11 labels contradicted 121 of 241 request-toned English rows, and
  7.8% of `score` rows carried a "near-tie" the text never showed). The dedicated `choice` head
  (L-002) and localized per-record-RNG data (B-1) lifted multilingual `choice` from ~0.25 (chance);
  on the five-domain split all six languages meet ECE ≤ 0.05, but on the support-only routed
  split `nl` (ECE 0.167) — with `de` 0.083 and `it` 0.056 — is still above the 0.05 target
  (NFR-C06, BACKLOG B-1). The CUDA-graph fast path (`TACHYONE_FAST=1`) improves p50 by **2.6×**
  on English (17.63 → 6.81 ms) and **3.7×** on the multilingual five-domain path
  (13.97 → 3.82 ms), always with **0 top-label changes** — the keyed gate and the per-domain
  heads run after the encode, which the graphed path never sees.

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
| B-3 | Confidence thresholding / System-2 handoff | ✅ (`v0.3.0`) |
| B-4 | Input-noise robustness + clean/noisy split | ✅ (`v0.3.0`) |
| — | Multilingual LoRA rank 16 → 64 | ✅ (`v0.3.0`) |

Details: [`docs/roadmap.md`](docs/roadmap.md) · Tasks: [`docs/tasks.md`](docs/tasks.md).

## Documentation map

| Doc | Contents |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | Vision, personas, use cases, success metrics, non-goals |
| [`docs/compare.md`](docs/compare.md) | **Why not a small LLM / Jev / another scorer?** — numbers and method |
| [`docs/use-cases.md`](docs/use-cases.md) | **Five recipes with tested JSON in/out** (classification, routing, priority, risk, triage) |
| [`docs/integrations.md`](docs/integrations.md) | **FastAPI · MCP · LangChain/LangGraph · n8n · Power Automate · Azure Functions · Docker** |
| [`docs/protocol.md`](docs/protocol.md) | The `/v1/systemone` contract |
| [`docs/architecture.md`](docs/architecture.md) | Components, flows, backend strategy, hooks |
| [`docs/cli.md`](docs/cli.md) | CLI flags, presets, modes, exit codes |
| [`docs/mcp.md`](docs/mcp.md) | MCP stdio server (`tachyone-mcp-server`) |
| [`docs/langchain.md`](docs/langchain.md) | LangChain/LangGraph `Runnable` adapter |
| [`docs/docker.md`](docs/docker.md) | Image, Compose, healthcheck, config in containers |
| [`docs/cookbook-handoff.md`](docs/cookbook-handoff.md) | Confidence thresholding + System-2 handoff |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Accuracy/ECE/latency numbers and limitations |
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones, exit criteria, backlog |
| [`docs/tasks.md`](docs/tasks.md) | Atomic task plan (M0–M6) |
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
