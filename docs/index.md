---
hide:
  - navigation
  - toc
---

<div class="jeba-hero">
<img class="jeba-hero__logo" src="assets/logo-wordmark.svg" alt="jeba — System 1 inference engine">
<span class="jeba-hero__eyebrow">v0.3.0 · Apache-2.0 · local-first</span>
<h1 class="jeba-hero__title">Ultra-fast, non-autoregressive encoder engine for structured intent and choice classification.</h1>
<p class="jeba-hero__sub">3.83&nbsp;ms p50 latency · non-generative (no hallucinated labels) · calibrated confidence · local-first on consumer GPUs.</p>
<p class="jeba-hero__cta">
<a class="jeba-btn jeba-btn--primary" href="#quickstart">Get started</a>
<a class="jeba-btn jeba-btn--ghost" href="https://github.com/munod/jeba">View on GitHub</a>
</p>
</div>

jeba answers atomic typed questions — `choice`, `score`, and `noul` — in a **single forward pass**
and returns probabilities plus calibrated `confidence`. It speaks the exact TypeSafe Jev
`/v1/systemone` wire protocol, so an existing Jev client can be repointed at a jeba server with
**no code changes**.

<div class="jeba-metrics">
<div class="jeba-metric"><span class="jeba-metric__value">3.83 ms</span><span class="jeba-metric__label">p50 fast path (CUDA graphs, RTX 3060, mmBERT, batch=1)</span></div>
<div class="jeba-metric"><span class="jeba-metric__value">85.3%</span><span class="jeba-metric__label">multilingual overall accuracy</span></div>
<div class="jeba-metric"><span class="jeba-metric__value">95.6%</span><span class="jeba-metric__label">Spanish (<code>es</code>) overall accuracy</span></div>
<div class="jeba-metric"><span class="jeba-metric__value">0.0044</span><span class="jeba-metric__label">fast-path probability parity Δ (0 top-label flips)</span></div>
</div>

## Why jeba

<div class="grid cards" markdown>

- :material-lightning-bolt:{ .lg .middle } **Sub-5&nbsp;ms speed**

    ---

    Per-shape **CUDA graphs** with **bf16**-resident weights, wired behind `JEBA_FAST=1`.
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

    Strictly proper scoring (RLCD) plus per-`(primitive, language)` temperature scaling, with
    per-language ECE reported. The `ECE ≤ 0.05` target is met for two of the six multilingual
    languages (`pt` 0.024 and `es` 0.038); `de`/`fr`/`it`/`nl` remain above it (worst `nl` 0.104).

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
    jeba is distributed from source for now. Install directly from GitHub (requires `git` and
    Python 3.12).

=== "uv"

    ```bash
    git clone https://github.com/munod/jeba
    cd jeba
    uv sync
    uv run jeba --predict --preset triage --backend fake "refund please"
    ```

=== "pip"

    ```bash
    git clone https://github.com/munod/jeba
    cd jeba
    pip install .
    jeba --predict --preset triage --backend fake "refund please"
    ```

=== "Python SDK"

    ```python
    # Requires jeba installed (see the uv / pip tabs)
    from jeba import JebaClient
    from jeba.primitives import ChoiceQuestion

    client = JebaClient("http://127.0.0.1:8000")
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
    git clone https://github.com/munod/jeba
    cd jeba
    uv sync --extra serve
    uv run jeba-serve

    curl -s http://127.0.0.1:8000/v1/systemone \
      -H "Content-Type: application/json" \
      -d '{"state":"hello","model":"jeba-latest",
           "questions":{"q":{"type":"noul","instructions":"Is this a greeting?"}}}'
    ```

## Benchmarks

Measured on a single RTX 3060 12GB (full tables and reproduction commands in the
[benchmark report](https://github.com/munod/jeba/blob/main/benchmarks/report.md)).

**Overall accuracy by language — multilingual checkpoint**

<div class="jeba-chart">
<div class="jeba-bar"><span>es</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:95.6%"></span></span><span class="jeba-bar__value">95.6%</span></div>
<div class="jeba-bar"><span>pt</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:88.5%"></span></span><span class="jeba-bar__value">88.5%</span></div>
<div class="jeba-bar"><span>it</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:88.0%"></span></span><span class="jeba-bar__value">88.0%</span></div>
<div class="jeba-bar"><span>fr</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:87.1%"></span></span><span class="jeba-bar__value">87.1%</span></div>
<div class="jeba-bar"><span>de</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:86.3%"></span></span><span class="jeba-bar__value">86.3%</span></div>
<div class="jeba-bar"><span>nl</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:66.3%"></span></span><span class="jeba-bar__value">66.3%</span></div>
<p class="jeba-chart__caption">Held-out synthetic split; temperature fitted per (primitive, language). Calibrated ECE is in-sample. Multilingual LoRA at rank 64.</p>
</div>

**p50 latency — stock forward vs CUDA-graph fast path**

<div class="jeba-chart">
<div class="jeba-bar"><span>stock</span><span class="jeba-bar__track"><span class="jeba-bar__fill jeba-bar__fill--muted" style="width:85.6%"></span></span><span class="jeba-bar__value">10.27 ms</span></div>
<div class="jeba-bar"><span>fast</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:31.9%"></span></span><span class="jeba-bar__value">3.83 ms</span></div>
<p class="jeba-chart__caption">mmBERT + <code>checkpoints/multi</code>, batch=1. 2.68× p50 speedup; p95 11.05 → 4.12 ms.</p>
</div>

## How it compares

Mirrors the canonical table in [`overview.md`](overview.md#how-jeba-compares).

| | Jev | Laya | Needle | **jeba** |
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
- [Protocol](protocol.md) — the frozen `POST /v1/systemone` contract.
- [Architecture](architecture.md) — components, flows, backend strategy.
- [CLI reference](cli.md) — every `jeba` flag with copy-paste examples.
- [Cookbook](cookbook-handoff.md) — uncertainty thresholding and System-2 handoff.
- [MCP server](mcp.md) · [LangChain](langchain.md) · [Docker](docker.md) — integrations.
- [Training](training.md) — data → LoRA/RLCD → calibration → evaluation.
- [Benchmarks](benchmarks.md) — accuracy, ECE, and latency.
- [Testing](testing.md) · [Release](release.md) · [Hugging Face](huggingface.md) ·
  [Model card](model-card.md).
- [Roadmap](roadmap.md) · [Tasks](tasks.md) · [Requirements](requirements/functional.md) ·
  [Decisions](adr/README.md).
