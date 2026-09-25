# Cookbook: abstain and hand off to System-2

jeba is a fast, local **System One**: it answers atomic questions in one forward pass and returns
calibrated `confidence`. It is not a reasoner. The intended pattern for hard inputs is to let a
slower **System Two** (a frontier LLM, a human reviewer, or a longer pipeline) take over when jeba
is not confident enough.

Because the wire contract is frozen (ADR-0001), this pattern is **client-side and additive**: you
read the `confidence` that already ships in every `choice`/`score` answer (and the probability in
`noul`) and decide whether to hand off. Nothing about `/v1/systemone` changes.

## The signal

`jeba.handoff` turns an answer or a whole response into a typed abstention signal:

- `assess(probabilities, threshold=τ)` → `Uncertainty(confidence, entropy, margin, threshold, abstain)`.
- `assess_response(response, threshold=τ)` → `HandoffReport(abstain, threshold, signals)`.

`abstain` is true when the selected mass is **strictly below** `τ`. Alongside `confidence` you get
two extra meters:

- `entropy` — Shannon entropy normalized to `[0, 1]` (`0` certain, `1` uniform).
- `margin` — gap between the top two probabilities in `[0, 1]` (larger = more dominant).

## Choosing τ

There is **no universal default** — the right threshold depends on how costly a wrong answer is
versus a handoff. Rough starting points:

| Shape of the decision | Suggested τ | Rationale |
| --- | --- | --- |
| Safety gate (`jailbreak`, `threat`) | low (≈ 0.3) | Prefer acting; a miss is costly, but so is over-escalation |
| Routing / triage | medium (≈ 0.5) | Balanced: hand off ambiguous tickets |
| Automation that acts on the result | high (≈ 0.8) | Only auto-act when clearly sure |

Tune on your own labelled data: plot accuracy versus confidence and pick the point where the
accuracy above the threshold is good enough for the downstream action.

## Python

```python
from jeba import JebaClient, assess_response

client = JebaClient("http://127.0.0.1:8000", api_key=None)
response = client.system_one(
    "I was charged twice and I want a refund now",
    questions,
    model="jeba-latest",
)

report = assess_response(response, threshold=0.6)
if report.abstain:
    # Route to a frontier LLM (or a human) with the full response for context.
    answer = system_two(state, questions, context=response.model_dump(mode="json"))
else:
    answer = response.answers
```

You can inspect per-question detail before deciding:

```python
for qid, signal in report.signals.items():
    print(qid, signal.type, f"confidence={signal.confidence:.2f}", f"entropy={signal.entropy:.2f}")
```

## CLI

`--threshold` preserves the canonical response and appends a sibling `handoff` object:

```bash
uv run jeba --predict --preset triage --backend fake --threshold 0.6 "refund please"
```

```json
{
  "model": "jeba-latest",
  "answers": { "...": "unchanged canonical answers" },
  "usage": { "input_tokens": 3, "output_tokens": 4 },
  "handoff": {
    "abstain": true,
    "threshold": 0.6,
    "signals": {
      "department": { "type": "choice", "confidence": 0.5, "entropy": 0.79, "margin": 0.33, "abstain": true }
    }
  }
}
```

Without `--threshold`, the output is byte-identical to a normal prediction.

## Notes

- `noul` has no separate `confidence`, so the helper uses its `probability` directly.
- `score` uses the answer `confidence` and the level distribution, never the expected
  `score` position.
- This does not change accuracy or ECE; it is purely a decision layer over the existing,
  calibrated `confidence` (see `docs/overview.md` and `NFR-C06`).
