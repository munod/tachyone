# n8n

Any n8n workflow that can do HTTP can make decisions. The **HTTP Request** node posts the
canonical body and the response comes back as ordinary JSON, so a branch condition can read
`answers.<id>.choice` and `answers.<id>.confidence` directly — no code node required.

- Transport: `POST /v1/systemone`
- Status: **payload verified** — the requests below were executed against a local
  `tachyone-serve`; the n8n node UI itself is not run in CI.

## 1. Make the server reachable

n8n usually runs in Docker, so put both services on the same compose network. Add this to the
repo's [`docker/compose.yaml`](https://github.com/munod/tachyone/blob/main/docker/compose.yaml)
(or to a sibling file you start alongside it):

```yaml
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    depends_on:
      tachyone:
        condition: service_healthy
```

Compose puts both on one network where the service name resolves, so inside n8n the URL is
`http://tachyone:8000/v1/systemone`. Running n8n outside that compose file? Point it at the
published port on the host instead — `http://host.docker.internal:8000/v1/systemone` (add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the n8n service).

## 2. HTTP Request node

| Field | Value |
| --- | --- |
| Method | `POST` |
| URL | `http://tachyone:8000/v1/systemone` |
| Authentication | Generic Credential → Header Auth → `Authorization` = `Bearer change-me` |
| Send Headers | `Content-Type` = `application/json` |
| Send Body | As JSON |
| Body | the payload below |

```json
--8<-- "assets/examples/ticket-routing.request.json"
```

Response (captured from a real encoder run):

```json
--8<-- "assets/examples/ticket-routing.response.json"
```

The same payload from a shell, if you want to check the node before building it:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" \
  --data @docs/assets/examples/ticket-routing.request.json
```

## 3. Read the decision in expressions

| What | Expression |
| --- | --- |
| Winning queue | `{{ $json.answers.queue.choice }}` |
| Confidence | `{{ $json.answers.queue.confidence }}` |
| Full distribution | `{{ JSON.stringify($json.answers.queue.probabilities) }}` |

Wire these into a **Set** node (`queue = …`) and your downstream nodes — Slack, email, a
database insert — receive a plain string.

## 4. Branch on confidence instead of trusting it

Add an **IF** node: `{{ $json.answers.queue.confidence }}` **is smaller than** `0.6` →

- **true** → route to a human queue (or call a large model): the decision was too close to call.
- **false** → continue automatically.

That is the System-2 handoff pattern from the [cookbook](../cookbook-handoff.md), expressed as a
workflow branch rather than Python. Set τ per decision shape — `0.6` is illustrative, not a
default.

## 5. Classify many items in one call

For a batch node (or a Schedule Trigger draining a queue), use the additive
`POST /predict/batch` instead of looping:

```json
{
  "requests": [
    { "state": "…", "model": "tachyone-latest", "questions": { "queue": { "type": "choice", "instructions": "Which queue should receive this ticket first?", "criteria": { "billing-queue": "invoices, payments, refunds", "technical-queue": "bugs, outages, system errors", "sales-queue": "pricing, new contracts", "general-queue": "everything else" } } } },
    { "state": "…", "model": "tachyone-latest", "questions": { "queue": { "type": "choice", "instructions": "Which queue should receive this ticket first?", "criteria": { "billing-queue": "invoices, payments, refunds", "technical-queue": "bugs, outages, system errors", "sales-queue": "pricing, new contracts", "general-queue": "everything else" } } } }
  ]
}
```

The response is `{"results": [ …one canonical response per item… ]}`; the canonical
`/v1/systemone` shape is untouched.

## Failure handling

| Status | Meaning | What the node should do |
| --- | --- | --- |
| `401` | Missing/wrong API key | Fix the credential; do not retry |
| `422` | Body failed validation (bad options, >255 choices, <2 or >10 score levels) | Turn on *Continue (using error output)* and inspect `error.body.details` |
| `429` / `529` | Upstream throttling / overload | Retry with exponential backoff + jitter |

## Related

- [Integrations](../integrations.md) — the other entry points and the security checklist.
- [Power Automate](power-automate.md) · [Azure Functions](azure-functions.md) — the same payload
  on the Microsoft stack.
- [Protocol](../protocol.md) — every field, every error status.
