# Overview

**Status:** Implemented and released (`v0.2.0`). All milestones M0–M6 are complete, plus the
post-M6 multilingual-quality (B-1) and CUDA-graph fast-path (B-2) work. See
[`benchmarks.md`](benchmarks.md) for measured results and `.specs/project/STATE.md` for the
authoritative decision log.

## What jeba is

jeba is a **local-first, multilingual decision engine** for atomic structured questions. It
answers three primitive question types — `choice`, `score`, and `noul` — and speaks the
**exact TypeSafe Jev `/v1/systemone` wire protocol**, so an existing Jev client can be
repointed at a jeba server with no code changes.

"System 1" is the useful mental model: a fast, non-autoregressive, single forward pass that
produces a calibrated answer, as opposed to the slow, sequential "System 2" reasoning of a
chat LLM. jeba is deliberately **not** a chatbot and **not** a general completion endpoint.

## Why it exists

Three reference points shaped the design:

| Reference | What we take | What we do differently |
| --- | --- | --- |
| **TypeSafe Jev** (`docs.typesafe.ai`) | The `/v1/systemone` wire contract and primitive semantics | We run it locally/offline; no hosted dependency in core |
| **Laya** (open-source System 1 engine) | Single-pass encoder primitives, router, hooks, calibration mindset, single wire protocol | We add a pluggable backend and an LLM backend for a fast end-to-end start |
| **Needle** (on-device foundation model) | Confidence calibration, small-footprint deployment, opt-out telemetry discipline | We fine-tune existing encoders instead of a from-scratch foundation model |

The gap jeba fills: hosted decision APIs are fast but remote/closed/paid; autoregressive
LLMs are local-capable but slow and poorly calibrated for atomic judgments. jeba offers a
familiar contract, running on your own hardware, with an LLM backend today and a trained
encoder backend next.

## Value proposition

- **Drop-in:** same request/response as Jev — repoint the client, keep the code.
- **Local-first:** base install runs offline with no API key; server mode is optional.
- **Fast:** non-autoregressive single forward pass for the local backend (target: single-digit
  milliseconds on GPU, tens of milliseconds on CPU).
- **Calibrated:** probabilities and `confidence` from strictly proper scoring (RLCD) + temperature fitting.
- **Multilingual:** mmBERT-based checkpoint covering 100+ languages via script/language routing.
- **Additive:** router control, hooks, and `predict_batch` extend without breaking the contract.

## Personas & use cases

| Persona | Pain | jeba use case |
| --- | --- | --- |
| **Python app developer** | Wants structured decisions without training models | Clone the repo, `uv sync`, call `decide()` / SDK, get primitives |
| **Agent builder** | Needs a fast, cheap guard/triage/moderation classifier | MCP tool / LangChain runnable; presets `guard`, `triage`, `moderation` |
| **Edge / on-prem / air-gapped team** | Cannot send data to a hosted API | Local encoder backend, no network, no key |
| **Migrating Jev user** | Locked to a hosted contract | Repoint base URL to `jeba-serve` unchanged |
| **Multilingual product team** | English-only classifiers | Router selects multilingual checkpoint automatically |
| **Researcher / ML engineer** | Opaque confidence | Reproducible training + ECE/latency reports |

### Concrete scenarios

- **Content triage:** classify incoming text into a fixed set of categories with calibrated confidence.
- **Guardrail:** `noul` "is this request disallowed?" gate in front of a chat model.
- **Scoring:** rate a response on a 1–5 ordered scale with a legend.
- **Routing:** pick the best of N downstream tools/models with `choice`.
- **Schema extraction:** turn a JSON Schema/pydantic model into decision questions via `decide()`.

## How jeba compares

| Dimension | Jev | Laya | Needle | **jeba** |
| --- | --- | --- | --- | --- |
| Wire contract | `/v1/systemone` (hosted) | `/v1/systemone` (self-hosted) | Own tool/embedding API | **Jev-exact, self-hosted** |
| Hosted dependency | Required | None | None (device engine) | **None in core** |
| LLM backend | — | No (encoder only) | No | **Yes (Phase 2, optional)** |
| Encoder backend | Own model | Yes | Yes (2-bit) | **Phase 3 (ModernBERT/mmBERT)** |
| Multilingual | Yes | Yes (mmBERT) | Partial | **Yes (mmBERT, 100+)** |
| Extension model | n/a | Router, hooks | Grammar, telemetry | **Router, hooks, batch (additive)** |
| License | Proprietary service | Apache-2.0 | Open | **Apache-2.0** |

## Success metrics

- **Contract:** golden Jev parity suite green for all primitives and error shapes.
- **Adoption proof:** a real Jev client answers via jeba in Phase 2.
- **Quality:** documented accuracy and ECE on public probes (MASSIVE, XNLI, typed-decisions).
- **Performance:** published p50/p95 latency and memory for encoder/ONNX/fast paths.
- **Portability:** base install works offline on CPU and GPU; no hosted dependency.

## Non-goals

- Hosted SaaS, billing, or multi-tenant auth.
- Free-form generation or chat — atomic structured decisions only.
- Reimplementing or depending on the hosted Jev SDK in core.
- A GUI/web console.
- Training a foundation model from scratch.

## When to hand off to System-2

jeba is deliberately a **System One**: it answers atomic questions fast and returns calibrated
`confidence`. It does not reason, so the intended pattern for hard inputs is to **abstain** and
let a slower System Two (a frontier LLM, a human, or a longer pipeline) decide. This is a
client-side, additive layer over the existing `confidence`; the `/v1/systemone` shape is
unchanged (ADR-0001).

```python
from jeba import assess_response

report = assess_response(response, threshold=0.6)  # τ is task-dependent; see the cookbook
if report.abstain:
    answer = system_two(state, questions)  # hand off
```

`jeba.handoff` also exposes normalized `entropy` and `margin` beside `confidence`, and the CLI
adds a sibling `handoff` object when you pass `--threshold`. The full pattern, suggested τ per
decision shape, and CLI examples live in [`cookbook-handoff.md`](cookbook-handoff.md).

## Status & documentation map

| Doc | Purpose |
| --- | --- |
| `docs/roadmap.md` | Phases, milestones, exit criteria |
| `docs/requirements/functional.md` | Functional requirements (IDs) |
| `docs/requirements/non-functional.md` | Non-functional requirements (IDs) |
| `docs/requirements/traceability.md` | Requirement → design → task matrix |
| `docs/protocol.md` | The `/v1/systemone` contract in detail |
| `docs/architecture.md` | Components, boundaries, flows, backend strategy |
| `docs/cookbook-handoff.md` | Abstain / System-2 handoff pattern |
| `docs/adr/` | Architecture decision records |
| `docs/tasks.md` | Atomic task breakdown with verification |
| `docs/testing.md` | Contract/parity tests, gates, benchmarks |
| `docs/training.md` | Data generation, LoRA/QLoRA, RLCD, calibration |
