# FastAPI server

`create_app()` in `tachyone.serve` is a plain FastAPI application factory. You can run it as a
standalone service, or mount it inside the app you already have — the routes, auth and error
mapping come with it.

- Module: `tachyone.serve` (`create_app`, `main`)
- Extra: `serve` (`uv sync --extra serve` → FastAPI + uvicorn)
- Tests: `tests/integration/test_serve.py`, `tests/e2e/test_jev_client.py`

## Run it standalone

```bash
uv sync --extra serve
TACHYONE_BACKEND=encoder uv run tachyone-serve
```

There are **no command-line flags**: configuration is environment-driven (`TACHYONE_HOST`,
`TACHYONE_PORT`, `TACHYONE_BACKEND`, `TACHYONE_API_KEY`, `TACHYONE_OFFLINE`, …), so the same
image runs identically under systemd, Docker or a PaaS.

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"0.3.0","backend":"encoder","device":"auto","models":[]}
```

## Mount it in your own FastAPI app

```python
from fastapi import FastAPI

from tachyone.backends import build_backend
from tachyone.config import Config
from tachyone.serve import create_app

config = Config.from_env()
tachyone_app = create_app(config, build_backend(config))

app = FastAPI()
app.mount("/", tachyone_app)          # your routes first, then /v1/systemone
```

Or take the handler's behaviour without mounting: the route body is three lines, and transport
must stay a thin wrapper — validation and calibration live in `tachyone.wire.answer`.

```python
from tachyone.wire import answer, parse_request

response = await answer(parse_request(payload), backend)
return response.model_dump(mode="json")
```

## Calling it from another Python service

If both processes run on the same box, skip HTTP entirely:

```python
from tachyone import TachyoneClient

client = TachyoneClient("http://127.0.0.1:8000", api_key="change-me")
result = client.system_one(state, questions)
print(result.answers["queue"].choice, result.answers["queue"].confidence)
```

The client implements the wire's retry contract (exponential backoff on `429`/`529`).

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Canonical Jev contract |
| `POST` | `/predict` | Additive; `model` defaults to `tachyone-latest` |
| `POST` | `/predict/batch` | Additive; `{"results": [...]}` — never changes the canonical shape |
| `GET` | `/health` | `status`, `version`, `backend`, `device`, `models` |

Set `TACHYONE_API_KEY` to require `Authorization: Bearer <key>` on every endpoint; a bad or
missing key returns `401` with the contract's error body.

## Production notes

- **Preload the checkpoint** (`TACHYONE_PRELOAD`) so the first request is not the one paying for
  the model load.
- **No TLS, no rate limiting** in-process — terminate both at a reverse proxy. See
  [Integrations → before you expose it](../integrations.md#before-you-expose-it-beyond-localhost).
- **GPU is optional**: the `encoder` backend runs on CPU too (tens of milliseconds), and the
  `onnx` extra gives a portable alternative for constrained runtimes.
- **Autoscaling:** each worker holds its own weights; scale out only within your VRAM budget.

## Related

- [Docker](../docker.md) — the same app in a container with a healthcheck.
- [Protocol](../protocol.md) — the frozen contract these routes serve.
- [Architecture](../architecture.md#servepy) — where transport ends and the backends begin.
