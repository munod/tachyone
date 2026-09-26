# Power Automate

An HTTP action posts the canonical body and the response lands in the run as ordinary JSON, so
expressions can read the decision directly. Typical flow: **new ticket arrives → classify →
branch on `confidence` → start an approval or assign the queue**.

- Transport: `POST /v1/systemone` over HTTPS
- Status: **payload verified** — the body below was executed against a local `tachyone-serve`
  and returned the documented shape; Power Automate itself was not run in CI.

## 1. Give Tachyone an HTTPS endpoint

The **HTTP** action is a premium connector and calls a **public HTTPS** URL (it does not use the
on-premises data gateway). So either:

- deploy `tachyone-serve` behind HTTPS — Azure Container Apps, App Service behind nginx, or your
  own gateway — with `TACHYONE_API_KEY` set and the key stored in a **Connection** or a
  Environment Variable; or
- keep Tachyone fully private and expose only an Azure Function that calls it (see
  [Azure Functions](azure-functions.md)).

```bash
TACHYONE_HOST=0.0.0.0 TACHYONE_API_KEY=change-me uv run tachyone-serve
```

Never publish the bare port to the internet: terminate TLS and rate limiting at the front
door — see [the security checklist](../integrations.md#before-you-expose-it-beyond-localhost).

## 2. Add the HTTP action

| Field | Value |
| --- | --- |
| Method | `POST` |
| URI | `https://<your-gateway>/v1/systemone` |
| Headers | `Content-Type` = `application/json` · `Authorization` = `Bearer change-me` |
| Body | the JSON below (raw text) |

```json
--8<-- "assets/examples/ticket-routing.request.json"
```

Response captured from a real encoder run:

```json
--8<-- "assets/examples/ticket-routing.response.json"
```

## 3. Read the decision

| What | Expression |
| --- | --- |
| Winning queue | `body('Tachyone')['answers']['queue']['choice']` |
| Confidence | `body('Tachyone')['answers']['queue']['confidence']` |
| Full distribution | `body('Tachyone')['answers']['queue']['probabilities']` |

Name the HTTP action **Tachyone** so the expression above resolves. The same values also appear
as dynamic content, which is usually easier in the designer.

## 4. Branch on confidence

Add a **Condition** on `confidence` **is less than** `0.6`:

- **If yes** → *Start an approval* (or post to Teams): the engine is telling you it is too close
  to call.
- **If no** → *Update ticket* with the chosen queue and continue automatically.

τ is a per-decision knob, not a universal default — see
[Choosing τ](../cookbook-handoff.md#choosing). This branch is the System-2 handoff pattern from
the [cookbook](../cookbook-handoff.md), drawn in a designer instead of written in Python.

## 5. Batch for queue drains

For a scheduled flow that classifies many items, call `POST /predict/batch` with
`{"requests": [ … ]}` — one HTTP action instead of an Apply-each loop. The response is
`{"results": [ …one canonical response per item… ]}` and the canonical shape is untouched.

## Failure handling

| Status | Meaning | Flow should |
| --- | --- | --- |
| `401` | Missing/wrong key | Fix the connection; do not retry |
| `422` | Body failed validation | Terminate with the `error.body.details` in the run history |
| `429` / `529` | Throttled / overloaded | Retry with exponential backoff (Configure after → Configure run after → has failed) |

## Related

- [Integrations](../integrations.md) — entry points and the security checklist.
- [n8n](n8n.md) · [Azure Functions](azure-functions.md) — same payload, other runtimes.
- [Protocol](../protocol.md) — every field and error status.
