---
hide:
  - navigation
  - toc
---

<div class="tachyone-hero">
<img class="tachyone-hero__logo" src="assets/logo-wordmark.svg" alt="Tachyone — System 1 inference engine">
<span class="tachyone-hero__eyebrow">v0.5.0 · Apache-2.0 · local-first</span>
<h1 class="tachyone-hero__title">Ultra-fast, non-autoregressive encoder engine for structured intent and choice classification.</h1>
<p class="tachyone-hero__sub"><strong>LLMs generate text. Tachyone produces calibrated decisions.</strong><br>
3.83&nbsp;ms p50 latency · non-generative (no hallucinated labels) · calibrated confidence · local-first on consumer GPUs.</p>
<p class="tachyone-hero__cta">
<a class="tachyone-btn tachyone-btn--primary" href="#quickstart">Get started</a>
<a class="tachyone-btn tachyone-btn--ghost" href="https://github.com/munod/tachyone">View on GitHub</a>
</p>
</div>

Tachyone answers atomic typed questions — `choice`, `score`, and `noul` — in a **single forward pass**
and returns probabilities plus calibrated `confidence`. It speaks the exact TypeSafe Jev
`/v1/systemone` wire protocol, so an existing Jev client can be repointed at a Tachyone server with
**no code changes**.

<div class="tachyone-metrics">
<div class="tachyone-metric"><span class="tachyone-metric__value">3.39 ms</span><span class="tachyone-metric__label">p50 fast path (CUDA graphs, RTX 3060, mmBERT, batch=1)</span></div>
<div class="tachyone-metric"><span class="tachyone-metric__value">99.8%</span><span class="tachyone-metric__label">English five-domain accuracy (v0.9.0, B-15)</span></div>
<div class="tachyone-metric"><span class="tachyone-metric__value">98.3%</span><span class="tachyone-metric__label">English accuracy on phrasing <em>never seen in training</em> (holdout, 0% template overlap)</span></div>
<div class="tachyone-metric"><span class="tachyone-metric__value">0.000559</span><span class="tachyone-metric__label">fast-path probability parity Δ (0 top-label flips)</span></div>
</div>

## Why Tachyone

<div class="grid cards" markdown>

- :material-lightning-bolt:{ .lg .middle } **Sub-5&nbsp;ms speed**

    ---

    Per-shape **CUDA graphs** with **bf16**-resident weights, wired behind `TACHYONE_FAST=1`.
    Graceful fallback on CPU/MPS: the fast path never changes the response shape.

- :material-target:{ .lg .middle } **Permutation-equivariant `choice` head**

    ---

    A dedicated low-rank bilinear residual (`r=32`) on top of the cosine baseline, so option
    order never biases the answer and the number of options is free (1–255).

- :material-earth:{ .lg .middle } **Native multilingual**

    ---

    An mmBERT-based checkpoint (LoRA rank 64) trained and **calibrated per language** for `pt`,
    `es`, `fr`, `de`, `it`, and `nl`, with automatic script/language routing.

- :material-shield-check:{ .lg .middle } **Calibrated confidence**

    ---

    Strictly proper scoring (RLCD) plus per-`(primitive, language)` calibration: temperatures
    keyed by a detector that reads the state **and** the localized question, and — since
    B-16 — a fitted `peakedness × strength` confidence for multilingual `choice`/`score`.
    On phrasing never seen in training, **all 12 `(primitive, language)` cells now sit at
    ECE ≤ 0.05** (B-17, 2026-10-10 — from 5 of 12 at B-15; worst cell `score/fr`
    0.0257, `score/it` 0.1495 → **0.0014**): **B-1's NFR-C06 acceptance is CLOSED** on the
    frozen yardstick. Every label is derived from the text it accompanies
    (B-11 + B-12), and the label audit published with the accuracy reports **0
    contradictory rows**.

- :material-compare:{ .lg .middle } **Decisions, not text**

    ---

    **LLMs generate text. Tachyone produces calibrated decisions.** Head-to-head against a small
    local model, against Jev, and against open System One scorers — with the numbers and the
    method notes: [Compare](compare.md).

- :material-power-plug:{ .lg .middle } **Local-first, drop-in**

    ---

    Base install runs offline with no API key. Same `/v1/systemone` contract as Jev; extensions
    (router control, hooks, `predict_batch`, MCP, LangChain) are additive only.

- :material-file-code:{ .lg .middle } **Portable runtimes**

    ---

    Encoder, **ONNX**, and an optional LLM backend behind the same seam; the wire never depends
    on which engine answers.

</div>

## Quickstart

!!! note "Not yet on PyPI"
    Tachyone is distributed from source for now. Install directly from GitHub (requires `git` and
    Python 3.12).

=== "uv"

    ```bash
    git clone https://github.com/munod/tachyone
    cd tachyone
    uv sync
    uv run tachyone --predict --preset triage --backend fake "refund please"
    ```

=== "pip"

    ```bash
    git clone https://github.com/munod/tachyone
    cd tachyone
    pip install .
    tachyone --predict --preset triage --backend fake "refund please"
    ```

=== "Python SDK"

    ```python
    # Requires tachyone installed (see the uv / pip tabs)
    from tachyone import TachyoneClient
    from tachyone.primitives import ChoiceQuestion

    client = TachyoneClient("http://127.0.0.1:8000")
    result = client.system_one(
        "I need a refund for a duplicate charge.",
        {"team": ChoiceQuestion(
            instructions="Which team should handle this request?",
            criteria={"billing": "payments and refunds", "technical": "bugs and outages",
                      "sales": "pricing and upgrades", "other": "general questions"},
        )},
    )
    ans = result.answers["team"]
    print(ans.choice, ans.confidence, ans.probabilities)
    ```

=== "Serve (drop-in Jev)"

    ```bash
    git clone https://github.com/munod/tachyone
    cd tachyone
    uv sync --extra serve
    uv run tachyone-serve

    curl -s http://127.0.0.1:8000/v1/systemone \
      -H "Content-Type: application/json" \
      -d '{"state":"hello","model":"tachyone-latest",
           "questions":{"q":{"type":"noul","instructions":"Is this a greeting?"}}}'
    ```

## Benchmarks

Measured on NVIDIA L4 23GB (v0.9.0, B-15; full tables and reproduction commands in the
[benchmark report](https://github.com/munod/tachyone/blob/main/benchmarks/report.md)).

**Overall accuracy by language — multilingual, five-domain split (routed pair)**

<div class="tachyone-chart">
<div class="tachyone-bar"><span>pt</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:96.5%"></span></span><span class="tachyone-bar__value">96.5%</span></div>
<div class="tachyone-bar"><span>es</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:95.8%"></span></span><span class="tachyone-bar__value">95.8%</span></div>
<div class="tachyone-bar"><span>fr</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:95.0%"></span></span><span class="tachyone-bar__value">95.0%</span></div>
<div class="tachyone-bar"><span>it</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:93.3%"></span></span><span class="tachyone-bar__value">93.3%</span></div>
<div class="tachyone-bar"><span>de</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:91.8%"></span></span><span class="tachyone-bar__value">91.8%</span></div>
<div class="tachyone-bar"><span>nl</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:80.7%"></span></span><span class="tachyone-bar__value">80.7%</span></div>
<p class="tachyone-chart__caption">Both published checkpoints pinned (the router sends empty or no-signal states to the English one by design — L-013 — ~13–15% of rows; the multilingual checkpoint alone answers 99.6% when it answers everything). Labels are derived from the text (B-11/B-12, 0 contradictory); temperature fitted per (primitive, language).</p>
</div>

**The honest generalization number — holdout phrasing, 0% of templates shared with training**

<div class="tachyone-chart">
<div class="tachyone-bar"><span>pt</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:95.4%"></span></span><span class="tachyone-bar__value">95.4%</span></div>
<div class="tachyone-bar"><span>es</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:95.0%"></span></span><span class="tachyone-bar__value">95.0%</span></div>
<div class="tachyone-bar"><span>fr</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:94.9%"></span></span><span class="tachyone-bar__value">94.9%</span></div>
<div class="tachyone-bar"><span>it</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:94.7%"></span></span><span class="tachyone-bar__value">94.7%</span></div>
<div class="tachyone-bar"><span>de</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:92.8%"></span></span><span class="tachyone-bar__value">92.8%</span></div>
<div class="tachyone-bar"><span>nl</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:83.1%"></span></span><span class="tachyone-bar__value">83.1%</span></div>
<p class="tachyone-chart__caption">B-17's holdout evals (2026-10-10): training excluded the holdout template pool, so this measures phrasing generalization instead of recognition (overall <strong>92.6%</strong> multilingual / 98.3% English — the generalization gap collapsed from −11.1 to −1.7 points after the teacher expanded the tone banks). The distance between the two charts is the generalization gap — and B-1's per-language ECE ≤ 0.05 target now passes <strong>all 12 cells</strong> on this set (worst `score/fr` 0.0257): <strong>B-1's NFR-C06 acceptance CLOSED</strong> (numbers in BACKLOG B-17; the frozen yardstick's sha pins never moved).</p>
</div>

**p50 latency — stock forward vs CUDA-graph fast path**

<div class="tachyone-chart">
<div class="tachyone-bar"><span>stock</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill tachyone-bar__fill--muted" style="width:74.8%"></span></span><span class="tachyone-bar__value">13.91 ms</span></div>
<div class="tachyone-bar"><span>fast</span><span class="tachyone-bar__track"><span class="tachyone-bar__fill" style="width:20.6%"></span></span><span class="tachyone-bar__value">3.82 ms</span></div>
<p class="tachyone-chart__caption">mmBERT + <code>checkpoints/multi</code>, batch=1, 189 states. 3.64× p50 speedup; p95 15.88 → 3.94 ms. Measured on the v0.9.0 artifact (NVIDIA L4, 2026-10-08) — the fast path itself is unchanged.</p>
</div>

## How it compares

Mirrors the canonical table in [`overview.md`](overview.md#how-tachyone-compares).

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

## Start here

- [Overview](overview.md) — vision, personas, non-goals.
- [Compare](compare.md) — **why not a small LLM, why not Jev, why not another scorer.**
- [Use cases](use-cases.md) — **five recipes with real JSON in and JSON out.**
- [Protocol](protocol.md) — the frozen `POST /v1/systemone` contract.
- [Architecture](architecture.md) — components, flows, backend strategy.
- [CLI reference](cli.md) — every `tachyone` flag with copy-paste examples.
- [Cookbook](cookbook-handoff.md) — uncertainty thresholding and System-2 handoff.
- [MCP server](mcp.md) · [LangChain](langchain.md) · [LangGraph](integrations/langgraph.md) ·
  [n8n](integrations/n8n.md) · [Power Automate](integrations/power-automate.md) ·
  [Azure Functions](integrations/azure-functions.md) · [FastAPI](integrations/fastapi.md) ·
  [Docker](docker.md) — [all integrations](integrations.md).
- [Training](training.md) — data → LoRA/RLCD → calibration → evaluation.
- [Benchmarks](benchmarks.md) — accuracy, ECE, and latency.
- [Testing](testing.md) · [Release](release.md) · [Hugging Face](huggingface.md) ·
  [Model card](model-card.md).
- [Roadmap](roadmap.md) · [Tasks](tasks.md) · [Requirements](requirements/functional.md) ·
  [Decisions](adr/README.md).
