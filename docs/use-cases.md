# Use cases

Five production-shaped recipes. Nothing here is hand-written prose pretending to be output:
every request lives in `docs/assets/examples/*.request.json` and is embedded below verbatim,
and every response was captured from a live `--backend encoder` run against the published
adapters, then quantized to 4 decimal places for readability (the distribution still sums to
exactly `1.0`, and `confidence` equals the selected mass).

`tests/test_docs_examples.py` validates every pair against the wire on each CI run — a docs
example that stops matching the contract fails the build.

## How to read a response

| Primitive | The decision | Why you can trust it |
| --- | --- | --- |
| `choice` | `choice` is the argmax of `probabilities` | Keys are **your** labels — there is no free text to drift |
| `score` | the selected level is the argmax of `probabilities`; `score` is the expected value across the legend | `legend` maps index → your own criteria |
| `noul` | a single probability in `[0, 1]` | No separate `confidence`; use `max(p, 1-p)` as certainty |

`confidence` is the probability mass of the selected option: `1.0` means the distribution is
concentrated, `0.5` on a four-level scale means the model genuinely sees two live candidates —
which is exactly when you should [hand off to System-2](cookbook-handoff.md).

## Run any of these

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  --data @docs/assets/examples/ticket-classification.request.json
```

or from Python (stdlib only — the SDK's typed form is on the [home page](index.md#quickstart)):

```python
import json
import urllib.request

payload = json.load(open("docs/assets/examples/ticket-classification.request.json"))
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/systemone",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
```

The same bodies work unchanged against a Jev server: they are `/v1/systemone` requests.

---

## 1. Support ticket classification

**Decide:** which category does this ticket belong to? **Primitive:** `choice`.

```json
--8<-- "assets/examples/ticket-classification.request.json"
```

```json
--8<-- "assets/examples/ticket-classification.response.json"
```

**What to do with it:** write `technical` to the ticket and move on — `confidence` 0.9955 leaves
nothing to escalate. The `probabilities` are useful even when you only store the winner: a
`billing` mass of 0.0045 is what lets a downstream audit reconstruct *why* the routing was right.
The built-in `triage` preset ships the same shape (team + urgency + frustration + churn risk):
`tachyone --predict --preset triage`.

## 2. Ticket routing

**Decide:** which queue receives it first? **Primitive:** `choice` over queue names, with
descriptions carrying the semantics (queue ids stay stable for your tooling, descriptions do the
classifying).

```json
--8<-- "assets/examples/ticket-routing.request.json"
```

```json
--8<-- "assets/examples/ticket-routing.response.json"
```

**What to do with it:** enqueue to `billing-queue`. Routing is the canonical `choice` case — the
`probabilities` keys are your queue ids, so a misroute is a data problem, never a string the model
invented. Option count is free (1–255) and the dedicated `choice` head is permutation-equivariant,
so reordering queues in the payload cannot change the answer.

## 3. Incident prioritization

**Decide:** what priority does this incident get? **Primitive:** `score` over an ordered scale.
Note the state is a **structured JSON object**, not a string — the wire takes `string | object | array`.

```json
--8<-- "assets/examples/incident-priority.request.json"
```

```json
--8<-- "assets/examples/incident-priority.response.json"
```

**What to do with it:** argmax gives `P2 major` (0.5113), `score` 1.607 is the expected value
across the legend. This one is deliberately instructive: `P3 moderate` still holds 0.2508, so
`confidence` is only 0.5113. That is a **calibrated shrug**, not a decision — with `τ = 0.6` this
incident goes to the on-call human or to a System-2 model instead of silently landing on the wrong
page.

## 4. Risk assessment

**Decide:** how risky is this request, and does compliance have to see it? **Primitives:** `score`
+ `noul` in one round trip.

```json
--8<-- "assets/examples/risk-assessment.request.json"
```

```json
--8<-- "assets/examples/risk-assessment.response.json"
```

**What to do with it:** `high` carries 0.7319 of the mass and `needs_compliance_review` is 0.7266 —
both clear a `τ = 0.6` gate, so the workflow blocks and routes to compliance. Ask both questions in
one request: it is one forward pass, and the two answers are computed over the same state.

## 5. Document triage

**Decide:** what document is this, and can a machine process it? **Primitives:** `choice` +
`noul`.

```json
--8<-- "assets/examples/document-triage.request.json"
```

```json
--8<-- "assets/examples/document-triage.response.json"
```

**What to do with it:** `invoice` at 0.7106 — above `τ = 0.6`, so it proceeds — while
`identity` still holds 0.2298, which is the kind of near-miss worth logging for review sampling.
`needs_human_review` = 1.0 is the gate: amounts and banking details are confirmed by a person
before anything is posted.

---

## Choosing the questions well

The examples are only as good as the criteria you write:

- **Labels are yours.** `criteria` keys become the `probabilities` keys, so name queues, teams and
  severities exactly as your tooling spells them.
- **Descriptions carry the meaning.** `"billing": "invoices, payments, refunds"` does more for
  accuracy than a clever `instructions` sentence.
- **Scales are ordered and short.** `score` takes 2–10 levels; incident and risk scales work best
  at 3–4.
- **Set τ per decision shape.** `0.6` above is illustrative, not a default — see
  [Choosing τ](cookbook-handoff.md#choosing).

## Reproducing these responses

```bash
uv sync --extra train
TACHYONE_BACKEND=encoder uv run tachyone-serve
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  --data @docs/assets/examples/incident-priority.request.json
```

Capture environment: single RTX 3060 12GB, published adapters, `2026-09-26`. Four examples
route to `munod/tachyone-en`; `document-triage` contains German invoice text, so the router
picks `munod/tachyone-multi` on its own (`Latin script detected as de`) — that routing is part
of what the example demonstrates. Values are quantized to 4 decimals (`score` to 3), with the
rounding residual absorbed by the largest mass so the distribution still sums to exactly `1.0`
and `confidence` still equals the selected mass. `tests/test_docs_examples.py` re-checks all
three invariants on every run.

## When to reach for an LLM instead

Open-ended input, no labeled examples, or a decision that needs multi-step reasoning: use a model,
and keep Tachyone as the fast gate in front of it. See
[Compare: why not just a small model?](compare.md#1-why-not-just-a-small-model).
