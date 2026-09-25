---
hide:
  - navigation
  - toc
---

<div class="jeba-hero">
<span class="jeba-hero__eyebrow">v0.2.0 · Apache-2.0 · local-first</span>
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
<div class="jeba-metric"><span class="jeba-metric__value">73.4%</span><span class="jeba-metric__label">multilingual <code>choice</code> accuracy</span></div>
<div class="jeba-metric"><span class="jeba-metric__value">86.5%</span><span class="jeba-metric__label">Portuguese (<code>pt</code>) overall accuracy</span></div>
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

    An mmBERT-based checkpoint trained and **calibrated per language** for `pt`, `es`, `fr`,
    `de`, `it`, and `nl`, with automatic script/language routing.

- :material-shield-check:{ .lg .middle } **Calibrated confidence**

    ---

    Strictly proper scoring (RLCD) plus per-`(primitive, language)` temperature scaling, with
    per-language ECE reported. The `ECE ≤ 0.05` target is met for `pt`/`fr`/`it` and most
    `score` cases; `es`/`nl`/`de` calibration is in progress.

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

=== "uv"

    ```bash
    uv add jeba
    uv run jeba --predict --preset triage --backend fake "refund please"
    ```

=== "pip"

    ```bash
    pip install jeba
    jeba --predict --preset triage --backend fake "refund please"
    ```

=== "Python SDK"

    ```python
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
<div class="jeba-bar"><span>pt</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:86.5%"></span></span><span class="jeba-bar__value">86.5%</span></div>
<div class="jeba-bar"><span>fr</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:85.5%"></span></span><span class="jeba-bar__value">85.5%</span></div>
<div class="jeba-bar"><span>it</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:82.7%"></span></span><span class="jeba-bar__value">82.7%</span></div>
<div class="jeba-bar"><span>nl</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:60.6%"></span></span><span class="jeba-bar__value">60.6%</span></div>
<div class="jeba-bar"><span>de</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:59.8%"></span></span><span class="jeba-bar__value">59.8%</span></div>
<div class="jeba-bar"><span>es</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:51.2%"></span></span><span class="jeba-bar__value">51.2%</span></div>
<p class="jeba-chart__caption">Held-out synthetic split; temperature fitted per (primitive, language). Calibrated ECE is in-sample.</p>
</div>

**p50 latency — stock forward vs CUDA-graph fast path**

<div class="jeba-chart">
<div class="jeba-bar"><span>stock</span><span class="jeba-bar__track"><span class="jeba-bar__fill jeba-bar__fill--muted" style="width:85.6%"></span></span><span class="jeba-bar__value">10.27 ms</span></div>
<div class="jeba-bar"><span>fast</span><span class="jeba-bar__track"><span class="jeba-bar__fill" style="width:31.9%"></span></span><span class="jeba-bar__value">3.83 ms</span></div>
<p class="jeba-chart__caption">mmBERT + <code>checkpoints/multi</code>, batch=1. 2.68× p50 speedup; p95 11.05 → 4.12 ms.</p>
</div>

## How it compares

| | Jev | Laya | Needle | **jeba** |
| --- | --- | --- | --- | --- |
| Wire contract | `/v1/systemone` (hosted) | `/v1/systemone` (self-hosted) | Own API | **Jev-exact, self-hosted** |
| Hosted dependency | Required | None | None | **None in core** |
| LLM backend | — | No | No | **Yes (optional)** |
| Encoder backend | Own model | Yes | Yes (2-bit) | **Yes (ModernBERT/mmBERT + LoRA)** |
| Multilingual | Yes | Yes | Partial | **Yes (100+ routed, 6 trained)** |
| Fast path | Proprietary | No | Quantized | **CUDA graphs (`fast` extra)** |
| License | Proprietary service | Apache-2.0 | Open | **Apache-2.0** |

## Start here

- [Overview](overview.md) — vision, personas, non-goals.
- [Protocol](protocol.md) — the frozen `POST /v1/systemone` contract.
- [Architecture](architecture.md) — components, flows, backend strategy.
- [Training](training.md) — data → LoRA/RLCD → calibration → evaluation.
- [Benchmarks](benchmarks.md) — accuracy, ECE, and latency.
