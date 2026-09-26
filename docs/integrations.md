# Integrations

Everything below talks to the same `POST /v1/systemone` contract. Adding a platform never adds a
field: an answer produced inside a workflow, an agent tool, or your own FastAPI service is the
same bytes an HTTP client would get from `tachyone-serve`.

## Pick your entry point

| You are building | Use | Start here |
| --- | --- | --- |
| A Python service (no HTTP hop) | `TachyoneClient` / `tachyone.wire.answer` | [Use cases](use-cases.md) |
| Anything that can do HTTP | `tachyone-serve` | [FastAPI server](integrations/fastapi.md) |
| An agent or MCP host | `tachyone-mcp-server` (stdio) | [MCP server](mcp.md) |
| A LangChain / LangGraph pipeline | `create_runnable()` | [LangChain](langchain.md) · [LangGraph](integrations/langgraph.md) |
| A container / compose stack | the published image | [Docker](docker.md) |
| A no-code workflow (n8n, Power Automate, Azure) | HTTP against `tachyone-serve` | [n8n](integrations/n8n.md) · [Power Automate](integrations/power-automate.md) · [Azure Functions](integrations/azure-functions.md) |

## Status of each recipe

Honest labels, so nobody mistakes a documented payload for a certified integration:

| Label | Meaning |
| --- | --- |
| **Shipped + tested** | Automated tests in this repo exercise the code path (`tests/integration/`, `tests/test_langchain.py`, `tests/test_mcp.py`, `tests/test_serve.py`). |
| **Payload verified** | The exact request shown was executed against a local `tachyone-serve` and returned the documented shape. The third party's own UI, connector or runtime was **not** run in CI. |

| Integration | Transport | Status |
| --- | --- | --- |
| [FastAPI server](integrations/fastapi.md) | HTTP | Shipped + tested |
| [MCP server](mcp.md) | stdio | Shipped + tested (skips without the `mcp` extra) |
| [LangChain](langchain.md) | in-process | Shipped + tested (skips without the `langchain` extra) |
| [LangGraph](integrations/langgraph.md) | in-process | Shipped + tested |
| [Docker](docker.md) | container | Shipped + tested |
| [n8n](integrations/n8n.md) | HTTP | Payload verified |
| [Power Automate](integrations/power-automate.md) | HTTP | Payload verified |
| [Azure Functions](integrations/azure-functions.md) | HTTP | Payload verified |

## HTTP is the common denominator

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  --data @docs/assets/examples/ticket-routing.request.json
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/systemone` | The canonical Jev contract — use this from any platform |
| `POST /predict`, `POST /predict/batch` | Additive extensions (batch never changes the canonical shape) |
| `GET /health` | Liveness, plus the active backend/device/models |

## Before you expose it beyond localhost

`tachyone-serve` binds `127.0.0.1` by default and the server does **not** rate-limit or
terminate TLS — that is deliberate (the wire defines `429`/`529`, the server does not invent
them). Any workflow tool running on another host needs all four of these:

```bash
TACHYONE_HOST=0.0.0.0          # listen beyond loopback
TACHYONE_API_KEY=change-me     # Bearer auth on every endpoint
# plus: TLS and rate limiting at a reverse proxy (nginx, Caddy, API gateway)
```

Prefer keeping it private: run the workflow engine and `tachyone-serve` on the same network, a
VPN, or an Azure VNet, and never publish the port to the internet.

## Related

- [Protocol](protocol.md) — the response every integration returns.
- [Use cases](use-cases.md) — the JSON bodies used throughout this section.
- [Compare](compare.md) — why a decision engine in the flow instead of a model call.
