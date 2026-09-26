# MCP server

Tachyone ships a stdio **Model Context Protocol** server so any MCP host (Claude Desktop, an MCP-aware
agent, a custom client) can call the decision engine as a tool, without HTTP.

- Console script: `tachyone-mcp-server` → `tachyone.mcp.server:main`
- Extra: `mcp` (`uv sync --extra mcp`)
- Tool exposed: **`tachyone_predict`**
- Tests: `tests/test_mcp.py` (marker: the test skips when the `mcp` extra is absent)

## Install

```bash
uv sync --extra mcp
```

Without the extra, `create_server()` raises
`the mcp extra is required for the MCP server: uv sync --extra mcp`.

## Run

```bash
uv run tachyone-mcp-server
```

The server builds a backend from the environment (`Config.from_env()`) and then blocks on stdio.
There is no port and no `--port` flag.

```bash
# model-free, always works
TACHYONE_BACKEND=fake uv run tachyone-mcp-server

# local encoder, offline once the weights are cached
TACHYONE_BACKEND=encoder TACHYONE_OFFLINE=1 uv run tachyone-mcp-server
```

## Registering with an MCP host

```json
{
  "mcpServers": {
    "tachyone": {
      "command": "uv",
      "args": ["run", "--no-sync", "tachyone-mcp-server"],
      "cwd": "/path/to/projeto_tachyone",
      "env": { "TACHYONE_BACKEND": "encoder" }
    }
  }
}
```

`--no-sync` avoids re-resolving the lockfile on every spawn. If you install Tachyone into a venv
instead, point `command` at that interpreter's `tachyone-mcp-server`.

## The tool

```text
tachyone_predict(state: str, questions: dict, model: str = "tachyone-latest") -> dict
```

| Argument | Type | Meaning |
| --- | --- | --- |
| `state` | `str` | the text being judged |
| `questions` | `dict` | canonical `/v1/systemone` question objects, keyed by id |
| `model` | `str` | model id echoed back (default `tachyone-latest`) |

The return value is the **canonical response payload** — exactly what `POST /v1/systemone`
returns:

```json
{
  "model": "tachyone-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.5, "technical": 0.17, "sales": 0.17, "other": 0.17 },
      "confidence": 0.5
    }
  },
  "usage": { "input_tokens": 1, "output_tokens": 1 }
}
```

Invalid questions raise the contract error (`422 Unprocessable Entity`), so a host sees a normal
tool error rather than a malformed payload.

## Boundary

- The MCP layer owns **transport only**. Question validation, calibration, confidence and the
  response shape all come from `tachyone.wire.answer`, so an MCP answer is byte-compatible with an
  HTTP answer.
- `predict_payload()` is plain Python and does **not** need the `mcp` extra — only
  `create_server()`/`main()` do, which is what keeps the tool testable in CI.
- Like the rest of the core, nothing here requires a network call unless you select the `llm`
  backend.

## Related

- [`docs/langchain.md`](langchain.md) — the `Runnable` adapter for LangChain/LangGraph.
- [`docs/protocol.md`](protocol.md) — the response shape the tool returns.
- `docs/adr/ADR-0009-extension-endpoints.md` — additive extension surface.
