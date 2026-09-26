# Compare

**LLMs generate text. Tachyone produces calibrated decisions.**

Three questions decide most evaluations of a decision engine. This page answers them with
measured numbers where we have them, and says plainly where a number is still pending rather
than filling the gap with a claim.

1. [Why not just a small model on Ollama / llama.cpp?](#1-why-not-just-a-small-model)
2. [Why not just use Jev?](#2-why-not-just-jev)
3. [Why not another open System One scorer?](#3-why-not-another-open-system-one-scorer)

---

## 1. Why not just a small model?

**Don't ask a model to decide. Ask Tachyone.**

A chat model asked to "classify this ticket" does three things you did not ask for: it generates
tokens one at a time until it decides to stop, it emits a string that a parser may or may not
accept, and it states a confidence that was never fitted to anything. Tachyone answers a typed
question in a single forward pass and returns a probability distribution over exactly the options
you supplied.

| Dimension | Small local LLM (`Ollama` / `llama.cpp`) | **Tachyone** |
| --- | --- | --- |
| What comes back | A token stream you must parse | A typed wire response; `probabilities` sum to `1.0` by contract |
| Schema compliance | Must be constrained and can still fail | **100% by construction** — no free text exists to leave the schema |
| Latency | Scales with the number of generated tokens | One forward pass: **3.83 ms p50** fast path · 10.27 ms stock (RTX 3060, batch=1) |
| Footprint on disk | 4.8 GB (`Ling-3.0-tiny` Q4) · 5.6 GB (`Ornith-1.5-9B` Q4) | **1.61 GB** English · **1.28 GB** multilingual (trunk + LoRA adapter) |
| Confidence | Self-reported; not calibrated | Fitted with temperature scaling: **ECE 0.023** (en) · 0.038 (es) |
| Determinism | Depends on sampling settings | Same input → same answer |
| Adapting it | Prompt engineering, every consumer re-does it | Training data → published LoRA adapter anyone can load |
| License | Model-dependent | **Apache-2.0** |

Footprint figures are the files actually on disk (ModernBERT-large trunk 1.58 GB + 30 MB adapter;
mmBERT-base trunk 1.23 GB + 54 MB adapter). Accuracy, ECE and latency come from the
[benchmark report](https://github.com/munod/tachyone/blob/main/benchmarks/report.md).

The memory and latency columns are the ones people underestimate. An encoder answer costs one
forward pass regardless of how many options you present; an autoregressive answer costs you one
decoding step per token, on a model several times larger, before you have even parsed the result.

### And when a small LLM *is* the right answer

When the input is open-ended, when there is no labeled data to train on, or when the task needs
multi-step reasoning rather than an atomic judgment — use the LLM. Tachyone is deliberately a
**System One**: it answers fast and returns calibrated `confidence`, and it abstains when it
should. The intended composition is not "Tachyone *or* Ollama", it is **Tachyone first, LLM when
the confidence is too low**:

```python
from tachyone import assess_response

report = assess_response(response, threshold=τ)
if report.abstain:
    answer = system_two(state, questions)  # your local LLM, frontier API, or a human
```

That way the large model is invoked on the 5% of inputs that actually need it, not on every
ticket. Full pattern, suggested τ per decision shape and CLI examples:
[Cookbook: abstain and hand off to System-2](cookbook-handoff.md).

---

## 2. Why not just Jev?

Because Tachyone **speaks Jev's contract without Jev's dependencies**. The wire shape, field
names, primitives and error statuses are frozen to match `POST /v1/systemone` exactly
(ADR-0001), and the parity suite in `tests/test_contract_wire.py` keeps them that way. For an
existing Jev client, migration is a base-URL change and nothing else.

| | Jev | **Tachyone** |
| --- | --- | --- |
| Where it runs | Hosted service | Your machine, your GPU, your VPC |
| Hosted dependency | Required | **None in the base install** — offline, no API key |
| What leaves your network | Every request | Nothing (ADR-0004, no telemetry — ADR-0011) |
| License | Proprietary service | **Apache-2.0** |
| Contract | `/v1/systemone` | **`/v1/systemone`, byte-compatible** |
| Extensions | Service-side | Router, hooks, `predict_batch`, MCP, LangChain — additive only |

The full dimension-by-dimension table (Laya, Needle included) is the canonical one in
[Overview: How Tachyone compares](overview.md#how-tachyone-compares), mirrored in the README.

If your constraint is "I already pay for Jev and it works", the honest answer is: keep it. Tachyone
is for the cases where hosted is not an option — air-gapped, on-prem, per-request cost, or simply
wanting the decision to happen next to the data.

---

## 3. Why not another open System One scorer?

There is more than one open model in this category now, which is good for the category. The
relevant question is which one you can put into production and what you can promise about it.

[**`pngwn/system-one-qwen3.5-4b-scorer`**](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer)
is a genuine peer: a Jev-shaped, single-pass scorer (Qwen3.5-4B-Base + LoRA r=16 + scalar scoring
head) that returns a distribution over the caller's options in one forward pass, with a fitted
temperature. As published on its own model card (held-out test split, n=576):

| Metric (their model card) | Value |
| --- | --- |
| Accuracy, all 9 task families | 0.707 |
| ECE after T=1.75 fitted on val | 0.044 |
| Latency at 4 options | 112.3 ms per question |
| Trunk | Qwen3.5-4B (4.66 B params) — 8.8 GB trunk + 136 MB adapter ≈ **8.9 GB** on disk |
| License | **CC-BY-NC-4.0** (non-commercial, inherited from the ticket data) |

What separates it from Tachyone for production use is not the idea — the idea is right and shared —
but three engineering facts:

- **Footprint:** 9.4 GB of trunk + adapter against 1.28–1.61 GB, on a 12 GB consumer GPU that also
  has to hold the activations.
- **Latency:** 112.3 ms at 4 options versus a single-digit-millisecond forward pass; and Tachyone's
  latency does not grow with the option count the same way (the `choice` head scores 1–255 options
  in one pass).
- **License:** CC-BY-NC-4.0 rules out commercial deployment of that checkpoint; Tachyone and its
  adapters are Apache-2.0, and the evaluation data stays in your hands.

A head-to-head run of Tachyone on **that model's own public test rows** — same GPU, same rows, same
metric code — is what makes this section more than a table of claims. It is produced by the
comparison harness under `benchmarks/` and recorded in the committed
[benchmark report](https://github.com/munod/tachyone/blob/main/benchmarks/report.md), alongside the
method notes (hardware, seed, temperature-fitting procedure, exact command).

!!! note "Numbers on this page"
    Accuracy and ECE for Tachyone come from the held-out **synthetic** split and are labeled as
    such in [`benchmarks.md`](benchmarks.md#known-limitations); public-probe evaluation is tracked
    as `B-7` in the project backlog. Figures attributed to other models are quoted from their own
    model cards, not re-measured here, and are labeled accordingly.
