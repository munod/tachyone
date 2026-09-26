# Docker

Image and Compose file live under `docker/`:

| File | Purpose |
| --- | --- |
| `docker/Dockerfile` | Python 3.12-slim + uv, `serve` extra only, non-root (`tachyone`, uid 10001) |
| `docker/compose.yaml` | one `tachyone` service on port 8000 with a `/health` healthcheck |
| `docker/.dockerignore` | drops `.venv`, `.opencode`, `training`, `*.safetensors`, … |

CI does **not** build the image; `tests/test_docker_assets.py` asserts the Dockerfile/compose
contents statically so they cannot rot.

## Quick start

```bash
docker compose -f docker/compose.yaml up --build
```

```bash
curl -s http://127.0.0.1:8000/health
# {"status": "ok", ...}
```

The compose file sets `context: ..` and `dockerfile: docker/Dockerfile`, so the command works
from anywhere as long as `-f docker/compose.yaml` points at the file in the repo.

## What the image runs

```dockerfile
RUN uv sync --locked --no-dev --extra serve
CMD ["uv", "run", "--no-sync", "tachyone-serve"]
```

- Only the `serve` extra is installed — no `train`, `onnx`, `mcp`, `langchain` or `fast`.
- `--locked` means the image build fails if `uv.lock` is out of date instead of silently
  re-resolving.
- `TACHYONE_BACKEND=fake` is baked in as the default so the container starts **without a key**.
- `TACHYONE_HOST=0.0.0.0` / `TACHYONE_PORT=8000`, `EXPOSE 8000`, non-root user.

## Configuration

Pass environment variables through `compose.yaml` → `services.tachyone.environment` (or `-e` with
plain `docker run`):

```yaml
services:
  tachyone:
    environment:
      TACHYONE_BACKEND: "encoder"       # fake | encoder | llm | onnx
      TACHYONE_API_KEY: "change-me"     # enables Bearer auth on /v1/systemone
      TACHYONE_LLM_BASE_URL: "https://api.openai.com/v1"
      TACHYONE_LLM_API_KEY: "sk-..."    # only for TACHYONE_BACKEND=llm
```

Full list: [Configuration](architecture.md#configuration). Secrets are read from the
environment at runtime — never committed, and never baked into the image.

### Using the local encoder in a container

Weights are cached under `~/.cache/tachyone/models` (`TACHYONE_MODELS_DIR`). The image runs as a
non-root user, so mount a volume to keep them between restarts:

```bash
docker run --rm -p 8000:8000 \
  -e TACHYONE_BACKEND=encoder \
  -v tachyone-models:/home/tachyone/.cache/tachyone/models \
  tachyone:local
```

First boot needs network to fetch weights from the Hugging Face Hub; add `-e TACHYONE_OFFLINE=1`
for subsequent air-gapped starts once the cache is warm (see
[ADR-0010](adr/ADR-0010-weights-distribution.md) and the prefetch note in
[ADR notes](adr/README.md#implementation-notes-post-acceptance)).

### With authentication

```bash
docker compose -f docker/compose.yaml run --rm -e TACHYONE_API_KEY=change-me tachyone
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"state":"refund please","model":"tachyone-latest","questions":{"q":{"type":"noul","instructions":"Is this a refund request?"}}}'
```

Unset `TACHYONE_API_KEY` in development: auth is then disabled (documented as dev-only).

## Healthcheck & restart

`compose.yaml` polls `GET /health` every 30s (5s timeout, 3 retries) and uses
`restart: unless-stopped`. `/health` is a liveness probe only — it does not require a key and
does not touch a backend.

## Endpoints available in the container

Same as a local `tachyone-serve` (see [Architecture](architecture.md#execution-flows)):

- `POST /v1/systemone` — canonical Jev contract
- `POST /predict`, `POST /predict/batch` — additive extensions
- `GET /health`

## Related

- [`docs/testing.md`](testing.md) — what CI actually builds and runs.
- `docs/adr/ADR-0004-local-first.md` — why the base image must not need a key.
- `docs/adr/ADR-0006-license-telemetry.md` — Apache-2.0, no telemetry.
