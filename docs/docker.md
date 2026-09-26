# Docker

Image and Compose file live under `docker/`:

| File | Purpose |
| --- | --- |
| `docker/Dockerfile` | Python 3.12-slim + uv, `serve` extra only, non-root (`jeba`, uid 10001) |
| `docker/compose.yaml` | one `jeba` service on port 8000 with a `/health` healthcheck |
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
CMD ["uv", "run", "--no-sync", "jeba-serve"]
```

- Only the `serve` extra is installed — no `train`, `onnx`, `mcp`, `langchain` or `fast`.
- `--locked` means the image build fails if `uv.lock` is out of date instead of silently
  re-resolving.
- `JEBA_BACKEND=fake` is baked in as the default so the container starts **without a key**.
- `JEBA_HOST=0.0.0.0` / `JEBA_PORT=8000`, `EXPOSE 8000`, non-root user.

## Configuration

Pass environment variables through `compose.yaml` → `services.jeba.environment` (or `-e` with
plain `docker run`):

```yaml
services:
  jeba:
    environment:
      JEBA_BACKEND: "encoder"       # fake | encoder | llm | onnx
      JEBA_API_KEY: "change-me"     # enables Bearer auth on /v1/systemone
      JEBA_LLM_BASE_URL: "https://api.openai.com/v1"
      JEBA_LLM_API_KEY: "sk-..."    # only for JEBA_BACKEND=llm
```

Full list: [Configuration](architecture.md#configuration). Secrets are read from the
environment at runtime — never committed, and never baked into the image.

### Using the local encoder in a container

Weights are cached under `~/.cache/jeba/models` (`JEBA_MODELS_DIR`). The image runs as a
non-root user, so mount a volume to keep them between restarts:

```bash
docker run --rm -p 8000:8000 \
  -e JEBA_BACKEND=encoder \
  -v jeba-models:/home/jeba/.cache/jeba/models \
  jeba:local
```

First boot needs network to fetch weights from the Hugging Face Hub; add `-e JEBA_OFFLINE=1`
for subsequent air-gapped starts once the cache is warm (see
[ADR-0010](adr/ADR-0010-weights-distribution.md) and the prefetch note in
[ADR notes](adr/README.md#implementation-notes-post-acceptance)).

### With authentication

```bash
docker compose -f docker/compose.yaml run --rm -e JEBA_API_KEY=change-me jeba
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"state":"refund please","model":"jeba-latest","questions":{"q":{"type":"noul","instructions":"Is this a refund request?"}}}'
```

Unset `JEBA_API_KEY` in development: auth is then disabled (documented as dev-only).

## Healthcheck & restart

`compose.yaml` polls `GET /health` every 30s (5s timeout, 3 retries) and uses
`restart: unless-stopped`. `/health` is a liveness probe only — it does not require a key and
does not touch a backend.

## Endpoints available in the container

Same as a local `jeba-serve` (see [Architecture](architecture.md#execution-flows)):

- `POST /v1/systemone` — canonical Jev contract
- `POST /predict`, `POST /predict/batch` — additive extensions
- `GET /health`

## Related

- [`docs/testing.md`](testing.md) — what CI actually builds and runs.
- `docs/adr/ADR-0004-local-first.md` — why the base image must not need a key.
- `docs/adr/ADR-0006-license-telemetry.md` — Apache-2.0, no telemetry.
