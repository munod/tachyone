# The jeba Wire Protocol (`/v1/systemone`)

**Status:** Frozen design (ADR-0001). This is the compatibility authority for jeba.
**Compatibility target:** TypeSafe Jev "System One" `/v1/systemone`.
**Rule:** Any change to a field in this document is a breaking change and MUST update the
contract test suite in the same commit (see `docs/testing.md`).

---

## Transport

```
POST /v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

- The endpoint always returns JSON.
- Multiple questions are evaluated **in parallel and independently** within one request.
- jeba honors the same semantics: ask many atomic questions, compose results in your code.

---

## Request

```json
{
  "state": "the text or structured context to judge",
  "model": "jeba-latest",
  "questions": {
    "q1": { "type": "noul",   "instructions": "Is this text a greeting?" },
    "q2": { "type": "choice", "instructions": "Pick a category.",
            "criteria": { "news": "journalism", "blog": null, "other": null } },
    "q3": { "type": "score",  "instructions": "Rate the sentiment.",
            "criteria": ["very negative","negative","neutral","positive","very positive"] }
  }
}
```

### Top-level fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `state` | `string \| object \| array` | Yes | The subject being judged. Passed to questions verbatim. |
| `model` | `string` | Yes | Requested model id. jeba maps this to a backend/checkpoint. |
| `questions` | `map<string, Question>` | Yes | Question id → question. Ids are echoed in `answers`. |

---

## Questions

A `Question` is a discriminated union on `type`.

### `noul`

Binary yes/no judgment.

```json
{
  "type": "noul",
  "instructions": "Is the message polite?",
  "criteria": { "true": "polite (optional elaboration)", "false": "impolite (optional)" }
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"noul"` | Yes | |
| `instructions` | `string \| object \| array` | Yes | The judgment to make. |
| `criteria.true` | `string \| object \| array` | No | Optional clarification of `true`. |
| `criteria.false` | `string \| object \| array` | No | Optional clarification of `false`. |

### `choice`

Pick one option from a labeled set.

```json
{
  "type": "choice",
  "instructions": "Which topic best matches?",
  "criteria": { "sports": "sports coverage", "tech": null, "politics": null }
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"choice"` | Yes | |
| `instructions` | `string \| object \| array` | Yes | |
| `criteria` | `map<string, string \| null>` | Yes | Option → optional description. **Max 255 options.** |

### `score`

Rate on an ordered scale.

```json
{
  "type": "score",
  "instructions": "Rate the quality.",
  "criteria": ["poor", "fair", "good", "excellent"]
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"score"` | Yes | |
| `instructions` | `string \| object \| array` | Yes | |
| `criteria` | `string[]` | Yes | Ordered levels. **Between 2 and 10 inclusive.** |

---

## Response

```json
{
  "model": "jeba-latest",
  "answers": {
    "q1": { "type": "noul", "noul": 0.97 },
    "q2": { "type": "choice", "choice": "tech", "probabilities": { "sports": 0.02, "tech": 0.95, "politics": 0.03 }, "confidence": 0.95 },
    "q3": { "type": "score", "score": 2.56, "legend": { "0": "poor", "1": "fair", "2": "good", "3": "excellent" }, "probabilities": { "0": 0.01, "1": 0.05, "2": 0.31, "3": 0.63 }, "confidence": 0.63 }
  },
  "usage": { "input_tokens": 214, "output_tokens": 18 }
}
```

### `Answer` variants

| Type | Fields |
| --- | --- |
| `noul` | `type:"noul"`, `noul: float` in `0..1`. **No separate `confidence`.** |
| `choice` | `type:"choice"`, `choice: string`, `probabilities: map<option,float>`, `confidence: float` |
| `score` | `type:"score"`, `score: number`, `legend: map<index,string>`, `probabilities: map<index,float>`, `confidence: float` |

**Confidence** is derived from the probability distribution: a distribution concentrated on
one option yields confidence near 1; a near-uniform distribution yields low confidence.
For `choice`/`score`, `confidence` is typically the probability mass of the selected option
(or a monotone function of the distribution — see `calibration.py`). `noul` is a single
probability and carries no separate confidence.

**Invariants:**
- `probabilities` keys exactly equal declared options (`choice`) or level indices (`score`).
- Values in `[0,1]`; they SHOULD sum to 1.0 (backends normalize before returning).
- `answers` keys exactly equal request `questions` keys.
- `legend` maps each score index to its `criteria` label.

---

## Usage

```json
{ "usage": { "input_tokens": 214, "output_tokens": 18 } }
```

Token accounting for cost/telemetry. Encoder backends may report an estimate or zero for
`output_tokens` (no generation); this is an additive, backward-compatible interpretation.

---

## Errors

| Status | Meaning | When | Client behavior |
| --- | --- | --- | --- |
| `401 Unauthorized` | Missing/invalid API key | Bad or absent `Authorization` | Re-authenticate; do not retry |
| `422 Unprocessable Entity` | Request validation failed | Malformed body, too many options, bad score levels | Fix request; do not retry |
| `429 Too Many Requests` | Rate limited | Too many requests | Retry with exponential backoff |
| `529 Overloaded` | Server overloaded | Capacity exceeded | Retry with exponential backoff |

The retry contract for `429`/`529` uses **exponential backoff**; clients should add jitter
and cap attempts. jeba's own SDK implements this; the server surfaces the status faithfully.

---

## Compatibility rules

1. **No field renames.** Field names and nesting match Jev exactly.
2. **No new required fields.** Additive optional fields are allowed only when clients that
   ignore them still work.
3. **Extensions are opt-in and additive.** Router overrides, hooks, and `predict_batch` live
   on extension endpoints or optional fields — never in the canonical response required shape.
4. **Unknown request fields:** default behavior is to ignore (forward-compatible), unless a
   future ADR decides otherwise.
5. **Contract test is law.** `tests/test_contract_wire.py` (planned) encodes this document.

---

## Additive extensions (non-canonical)

These are jeba extensions and MUST NOT alter `/v1/systemone` output shape.

| Extension | Surface | Purpose |
| --- | --- | --- |
| Router control | optional request field / extension endpoint | Force a checkpoint or language |
| Hooks | Python API | `on_predict_start`, `on_predict_end`, `on_route`, `on_load`, `on_evict`, `on_error` |
| `predict_batch` | `/predict/batch` (extension) | Batch several states in one call |
| `return_details` | optional request flag / SDK arg | Include full distributions |
| Presets | CLI/SDK | `router`, `guard`, `moderation`, `triage`, `email` |

See `docs/architecture.md` for the extension contracts and `docs/adr/ADR-0002-pluggable-backend-phasing.md`.
