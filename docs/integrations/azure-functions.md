# Azure Functions

The function is a thin HTTPS façade: it receives a request from Power Apps, Logic Apps, a Teams
bot or a Service Bus queue, posts the canonical body to a private `tachyone-serve`, and returns
the decision. Tachyone never has to be reachable from outside your network.

- Transport: `POST /v1/systemone` from the function to Tachyone
- Status: **payload verified** — the body and response below were executed against a local
  `tachyone-serve`; the Azure Functions host was not run in CI.

## 1. Where Tachyone runs

| Option | Latency profile | When |
| --- | --- | --- |
| Container Apps / AKS next to the function (VNet) | GPU: single-digit ms p50 | You need the fast path |
| Same VM, CPU encoder | tens of ms | Moderate volume, simplest ops |
| CPU + [`onnx` extra](../architecture.md#backend-strategy-decided) | portable, no torch | Constrained runtimes |

Azure Functions itself has **no GPU**, so the encoder runs in a neighbour service, not inside the
function. Cold-start the *model*, not the function: preload the checkpoint and keep the container
warm (`TACHYONE_PRELOAD`).

## 2. The function (Python v2 model)

```python
import json
import os
import urllib.request

import azure.functions as func

app = func.FunctionApp()


@app.route(route="decide", auth_level=func.AuthLevel.FUNCTION)
def decide(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": {"code": "invalid_json", "message": "body is not JSON"}}),
            status_code=400,
            mimetype="application/json",
        )

    upstream = urllib.request.Request(
        f"{os.environ['TACHYONE_URL'].rstrip('/')}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if key := os.environ.get("TACHYONE_API_KEY"):
        upstream.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(upstream, timeout=15) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:  # pass the contract's error through
        return func.HttpResponse(exc.read(), status_code=exc.code, mimetype="application/json")
    except (urllib.error.URLError, TimeoutError) as exc:
        return func.HttpResponse(
            json.dumps({"error": {"code": "unreachable", "message": str(exc)}}),
            status_code=502,
            mimetype="application/json",
        )

    return func.HttpResponse(json.dumps(body), status_code=200, mimetype="application/json")
```

Only stdlib is used, so the function needs no extra packages beyond `azure-functions`. It is a
pass-through: the response body is the wire response, byte for byte.

`local.settings.json`:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "TACHYONE_URL": "http://tachyone.internal:8000",
    "TACHYONE_API_KEY": "change-me"
  }
}
```

## 3. Call it with the canonical body

```json
--8<-- "assets/examples/ticket-routing.request.json"
```

What the function returns (captured from a real encoder run against the same body):

```json
--8<-- "assets/examples/ticket-routing.response.json"
```

Read it in a downstream Logic Apps expression:
`body('Decide')['answers']['queue']['choice']`, and branch when
`body('Decide')['answers']['queue']['confidence']` is less than `0.6` — the System-2 handoff from
the [cookbook](../cookbook-handoff.md).

## 4. Trigger shapes that fit decisions

- **HTTP** — synchronous classification inside an app or a Power App.
- **Service Bus / Queue** — drain a ticket queue: one function invocation per message, or
  `POST /predict/batch` for a chunk of messages in one call.
- **Timer** — periodic triage of accumulated items.

## Failure handling

The function forwards Tachyone's contract errors untouched: `401` (bad key), `422` (body failed
validation — check `error.body.details`), `429`/`529` (retry with exponential backoff + jitter).
A `502` from the function means Tachyone itself was unreachable — alert on that separately from
the contract statuses.

## Related

- [Integrations](../integrations.md) — entry points and the security checklist.
- [FastAPI server](fastapi.md) — what the function is talking to.
- [Power Automate](power-automate.md) — the caller side of this pattern.
- [Protocol](../protocol.md) — the response body this function passes through.
