# LangChain adapter

`jeba.integrations.langchain` wraps a jeba backend as a LangChain/LangGraph **`Runnable`**, so a
graph node can make a typed `choice` / `score` / `noul` decision and keep the canonical response.

- Module: `jeba.integrations.langchain`
- Extra: `langchain` (`uv sync --extra langchain` → `langchain-core` + `langgraph`)
- Tests: `tests/test_langchain.py`

## Install

```bash
uv sync --extra langchain
```

Without the extra, `create_runnable()` raises `the langchain extra is required: uv sync --extra langchain`.
`predict()` needs **no** extra — it is plain Python over `jeba.wire.answer`.

## Two entry points

### `predict(backend, state, questions, *, model="jeba-latest") -> dict`

Runs one request and returns the canonical payload. Verified output with the model-free backend:

```python
from jeba.backends.fake import FakeBackend
from jeba.integrations.langchain import predict

questions = {
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "technical": "bugs"},
    },
}

predict(FakeBackend(), "please refund", questions)
# {'model': 'jeba-latest',
#  'answers': {'urgent': {'type': 'noul', 'noul': 0.5},
#              'team': {'type': 'choice', 'choice': 'billing',
#                       'probabilities': {'billing': 0.5, 'technical': 0.5},
#                       'confidence': 0.5}},
#  'usage': {'input_tokens': 3, 'output_tokens': 2}}
```

### `create_runnable(backend, *, model="jeba-latest") -> Runnable`

```python
from jeba.backends import build_backend
from jeba.config import Config
from jeba.integrations.langchain import create_runnable

backend = build_backend(Config.from_env())          # JEBA_BACKEND decides
runnable = create_runnable(backend, model="jeba-latest")

runnable.invoke({
    "state": "please refund",
    "questions": questions,
})
# -> {'model': ..., 'answers': {...}, 'usage': {...}}
```

Input contract for `invoke`:

| Key | Type | Required |
| --- | --- | --- |
| `state` | `str` | yes |
| `questions` | `dict` of canonical question objects | yes |
| `model` | `str` | no (falls back to the `model=` passed to `create_runnable`) |

Inside LangGraph:

```python
from langchain_core.runnables import RunnableLambda

decide = RunnableLambda(lambda payload: runnable.invoke(payload))

graph.add_node("decide", decide)
```

## Choosing a backend

```python
# model-free (tests, demos)
from jeba.backends.fake import FakeBackend
runnable = create_runnable(FakeBackend())

# whatever JEBA_BACKEND says (fake / encoder / onnx / llm)
backend = build_backend(Config.from_env())
runnable = create_runnable(backend)
```

## Boundary rules

- The adapter is **additive**: it calls `jeba.wire.answer`, so field names, nesting and error
  statuses are identical to `POST /v1/systemone`. It never invents a LangChain-specific shape.
- `predict()`/`create_runnable()` run `asyncio.run(...)` internally — do not call them from
  inside a running event loop.
- Invalid questions raise the contract `422`; let it propagate so the graph sees the failure.
- Optional deps stay optional: importing the module without the extra is fine, only
  `create_runnable()` needs `langchain-core` (ADR: no hosted/heavy dependency in core).

## Related

- [`docs/mcp.md`](mcp.md) — the same capability over the MCP stdio protocol.
- [`docs/protocol.md`](protocol.md) — the response shape.
- `docs/adr/ADR-0009-extension-endpoints.md` — why extensions must not alter the wire.
