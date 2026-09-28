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

!!! note "Latency depends on the question, not just on the engine"
    The 3.83 / 10.27 ms figures are p50 on short, four-option benchmark states. Measured
    end-to-end on real corpora, the same stock path runs **22.2 ms p50** on Tachyone's own support
    records and **72.6 ms** on a nine-family public probe where option counts reach 77 — against
    112.9 / 350.2 ms for the open peer scorer and 1.4–6.8 **seconds** for the small LLMs. The
    full method and both tables are in [§3](#3-why-not-another-open-system-one-scorer).

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

### Head-to-head: same rows, same metric code, same GPU

`benchmarks/compare.py` answers **the same rows** with every engine through one implementation of
accuracy, 10-bin ECE and Brier on a single RTX 3060 12GB. It runs on **two** evaluation sets,
because a benchmark run on someone else's training data measures domain coverage as much as it
measures the engine — quoting only one of these tables would mislead in either direction.

#### A. Their distribution — `pngwn/system-one-decisions` test split, 64 rows per task family

They trained on this data; Tachyone and both LLMs are zero-shot here.

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm (ling-tiny) | 36 | 32 | 0.333 | 0.231 | 0.150 | 0.736 | 0.384 |
| llm (ornith-9b) | 36 | 36 | 0.611 | 0.242 | 0.263 | 0.641 | 0.513 |
| systemone-qwen3.5-4b | 576 | 576 | 0.705 | 0.134 | 0.046 | 0.373 | 0.746 |
| **tachyone (encoder)** | 576 | 576 | 0.241 | 0.291 | 0.031 | 0.760 | 0.261 |

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llm (ling-tiny) | 1662.23 | 50326.40 | 0.1 | 0.889 | 9494 | 4836 |
| llm (ornith-9b) | 6844.39 | 20876.05 | 0.1 | 1.000 | 8749 | 5512 |
| systemone-qwen3.5-4b | 350.23 | 4660.68 | 0.9 | 1.000 | 9147 | 8921 |
| **tachyone (encoder)** | 71.93 | 536.78 | 6.7 | 1.000 | 2994 | 3626 |

Accuracy by task family (calibrated):

| Task family | ling-tiny | ornith-9b | systemone | **tachyone** |
| --- | ---: | ---: | ---: | ---: |
| ag_news | 0.500 | 1.000 | 0.922 | 0.344 |
| banking77 | 0.000 | 0.750 | 0.891 | 0.047 |
| go_emotions | 0.750 | 0.500 | 0.859 | 0.359 |
| mmlu | 0.500 | 0.750 | 0.688 | 0.203 |
| tickets_language | 0.500 | 1.000 | 0.891 | 0.641 |
| tickets_priority | 0.000 | 0.500 | 0.469 | 0.031 |
| tickets_queue | 0.250 | 0.000 | 0.234 | 0.047 |
| tickets_type | 0.500 | 0.750 | 0.750 | 0.297 |
| yelp_score | 0.000 | 0.250 | 0.641 | 0.203 |

The Tachyone row was re-run on the B-11 adapters (2026-09-28): 0.229 → **0.241**, with the peers
untouched — nothing about *their* data or *their* weights moved.

#### B. Tachyone's distribution — our English support records, 64 rows per primitive

We trained on this data; the peer scorer and both LLMs are zero-shot here. 192 test rows, 192
validation rows for the temperature fit.

> **Re-scored on the B-11 labels (2026-09-28).** The gold labels under this table changed with
> ADR-0014 (121 of the 241 request-toned `noul` rows used to read `0`), so **all four engines were
> re-run on the same corrected rows** — keeping three engines on the old labels and one on the new
> would break the "same rows" rule this comparison exists for. Their accuracies move for that
> reason alone; the peer's and the LLMs' models did not change.

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm (ling-tiny) | 48 | 42 | 0.438 | 0.171 | 0.069 | 0.606 | 0.471 |
| llm (ornith-9b) | 48 | 48 | 0.750 | 0.099 | 0.293 | 0.509 | 0.457 |
| systemone-qwen3.5-4b | 192 | 192 | 0.599 | 0.135 | 0.081 | 0.514 | 0.585 |
| **tachyone (encoder)** | 192 | 192 | 0.953 | 0.025 | 0.019 | 0.076 | 0.946 |

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llm (ling-tiny) | 1196.30 | 15671.15 | 0.2 | 0.875 | 5017 | 4832 |
| llm (ornith-9b) | 3233.96 | 8400.71 | 0.2 | 1.000 | 5815 | 5510 |
| systemone-qwen3.5-4b | 111.27 | 150.31 | 7.9 | 1.000 | 9182 | 8754 |
| **tachyone (encoder)** | 22.40 | 35.72 | 32.0 | 1.000 | 2427 | 1743 |

Accuracy by primitive (calibrated):

| Primitive | ling-tiny | ornith-9b | systemone | **tachyone** |
| --- | ---: | ---: | ---: | ---: |
| choice | 0.438 | 0.812 | 0.703 | 0.953 |
| noul | 0.625 | 0.938 | 0.609 | 1.000 |
| score | 0.250 | 0.500 | 0.484 | 0.906 |

#### How to read these two tables

- **Each engine wins at home.** 0.705 vs 0.241 on their data; 0.953 vs 0.599 on ours. That gap is
  domain coverage — it says the released adapters are narrow (support tickets, 4 teams), not that
  one architecture beats the other. Broadening the training data is tracked as `B-5`.
- **Tachyone is faster and smaller on *both* turfs:** 4.9× the peer's throughput on their rows
  (71.9 vs 350.2 ms p50) and 5.0× on ours (22.4 vs 111.3 ms), with **5× less VRAM** (1.7 vs
  8.6 GiB) and a 1.6 GB footprint against 8.9 GB. Option count drives the peer's cost (its p95 is
  4.7 s on 77-option questions); Tachyone's `choice` head scores 1–255 options in one pass.
- **The small LLMs failed the contract, and the prompt was part of the reason.** With the
  original system prompt `ling-tiny` produced a contract-valid answer for **3/36** and **20/48**
  questions and `ornith-9b` for 32/36 and 40/48; after the prompt fix (`B-10`, table below) those
  are **32/36** and **44/48**, **36/36** and **48/48** — the 2026-09-28 home re-run gives 42/48
  and 48/48 (run-to-run generation variance; the compliance column is label-independent).
  Failures still count as *wrong* in the accuracy column. When `ornith-9b` answers it is
  respectable (0.611 / 0.750) — at 3.2–6.8 s p50, i.e. **20–29× slower than the peer and
  95–144× slower than Tachyone**.
- **Calibration has to be read with confidence.** Tachyone ships well-calibrated on its own turf
  (ECE raw **0.025**). Off-domain its confidence carries no signal: raw ECE **0.291**, the fit
  pushes T into the 20.0 ceiling of the grid, and mean confidence collapses to **0.261**. A table
  quoting only `ECE cal` (0.031) would look *better* than the peer's 0.046 while being useless —
  `Brier` (0.760 vs 0.373) and `Conf` expose it. **ECE alone is not a quality metric.**

#### What the prompt fix (`B-10`) changed

The four LLM rows above were re-run after `src/tachyone/backends/llm.py` started spelling out the
`answers` wrapper for **all three** primitives (one complete example, an explicit rule that the
wrapper is mandatory, and bans on the value errors actually observed: a boolean `noul`, the
criteria text echoed back, a bare string instead of an answer object, and an option label used as
the question id). The parser was not loosened — `build_answers` still rejects the inner shape, it
just now names the missing wrapper and echoes the top-level keys it saw. Same rows, same servers,
same flags, greedy decoding (`--temp 0 --seed 42`):

| Engine | Set | `JSON ok` before → after | answered before → after | Accuracy before → after |
| --- | --- | ---: | ---: | ---: |
| llm (ling-tiny) | probe, n=36 | 0.083 → **0.889** | 3 → 32 | 0.028 → 0.333 |
| llm (ornith-9b) | probe, n=36 | 0.889 → **1.000** | 32 → 36 | 0.556 → 0.611 |
| llm (ling-tiny) | home, n=48 | 0.417 → **0.917** | 20 → 44 | 0.125 → 0.312 |
| llm (ornith-9b) | home, n=48 | 0.833 → **1.000** | 40 → 48 | 0.583 → 0.688 |

Compliance is the metric this change targets; **accuracy and ECE are not comparable before vs
after** (the prompt moves the answer distribution and a different set of rows gets answered), so
both are recorded rather than one. The home rows' accuracy above was measured on the pre-B-11
labels; re-scored on the corrected labels (2026-09-28) the same post-fix rows give **0.438**
(`ling-tiny`) and **0.750** (`ornith-9b`), while compliance — label-independent — stays at
0.875 / 1.000 for that run. Latency moves for the same reason: with the wrapper failures
gone, `ling-tiny` stops spending its second attempt on a shape it was never going to fix — probe
p50 7.16 s → 1.66 s. Four probe and four home questions still fail with `ling-tiny`; those are the
residual `0.889` / `0.917`.

#### Method

| | |
| --- | --- |
| Hardware | single RTX 3060 12GB · Python 3.12 · torch 2.14.0+cu130 · transformers 5.17.0 |
| Tachyone | `encoder` backend, **stock** forward (the `fast` extra is not installed here), driven through `tachyone.wire.answer`, **B-11 adapters** (`checkpoints/en`, ADR-0014) |
| Labels | the gold labels under table B are the **B-11 corrected** `noul` labels; all four engines were re-run on them 2026-09-28, because mixing label sets would break the "same rows" rule |
| Peer scorer | Qwen3.5-4B-Base + its published adapter, bf16, `max_len=384`, option batch 4, scored by **its own** `system_one.py` imported from a local download — no third-party code is vendored in this repository |
| LLMs | `llama-server` (Q4_K_M, `--temp 0 --seed 42`) behind Tachyone's `llm` backend, `TACHYONE_LLM_RETRIES=1` (two attempts per question), system prompt as of the `B-10` fix |
| Temperature | fitted per engine on the capped **validation** split (grid 0.25–20.00, step 0.05; their card used 0.25–6.0) and applied to test |
| Latency | sequential, batch=1, warmup excluded; includes request marshalling and, for the LLMs, HTTP + generation |

**The harness validates against their published numbers.** Independent of their model card, this
run reproduces: temperature **1.75** (card 1.75), accuracy **0.705** (card 0.707), ECE
**0.046** (card 0.044), 537 validation / 576 test rows, and **8 of 9** per-task accuracies
identical to the card (`mmlu` 0.688 vs 0.703 is the only deviation). Tachyone's row likewise
reproduces ours: 0.953 on the 192-row subset against 0.945 on the full 1,500-row split.

#### Limitations, before quoting anything

- **Our accuracy is in-sample synthetic** (`B-9`, lesson L-005): the eval split shares states with
  training data. External evaluation on MASSIVE / XNLI / typed-decisions is now published in
  [`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md) —
  0.330 / 0.013 / 0.333 against chance levels of 0.20–0.50 / 0.017 / 0.333, i.e. the same story
  from the other side: domain coverage, not architecture (and MASSIVE's 0.013 — *below* its
  0.017 chance — is the multilingual `choice` regression of B-11 seen from outside).
- **Each engine is at home in exactly one table.** Neither number generalizes; that is why both
  are published.
- **Different `n` for the LLMs** (36 and 48 rows): a single failing question costs multiple
  autoregressive attempts — one 52-option attempt took **46 s** — so they ran on the first 4 rows
  per family / 16 per primitive, subsets of the same ordered rows the other engines scored.
- **`p95` for the LLMs is failure-dominated** (50.3 s for `ling-tiny` and 20.9 s for `ornith-9b`
  on the probe), not a steady-state tail.
- **The peer and its data are CC-BY-NC-4.0**: evaluation only, never redistributed. Tachyone and
  its adapters are Apache-2.0.

Reproduce it from `benchmarks/README.md` → *Head-to-head comparison*; artifacts are the
`benchmarks/results/compare_*.json` files.

!!! note "Figures quoted from other projects"
    The peer's model-card numbers (0.707 accuracy, 0.044 ECE, 112.3 ms at 4 options) are quoted
    from its own card and labeled as such; everything in the four tables above is **measured
    here** on the hardware and rows described in the method.
