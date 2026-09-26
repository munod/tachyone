# CLI reference

`tachyone` is a **single command with flags** — there is no subcommand syntax (`tachyone download`,
`tachyone serve …` and similar do not exist; use `--serve`, and prefetch weights with `hf download`,
see [ADR-0010](adr/ADR-0010-weights-distribution.md)).

Entry points (declared in `pyproject.toml`):

| Command | Module | Purpose |
| --- | --- | --- |
| `tachyone` | `tachyone.cli:main` | print questions, predict locally or against a server, start the server |
| `tachyone-serve` | `tachyone.serve:main` | HTTP server (`serve` extra) |
| `tachyone-mcp-server` | `tachyone.mcp.server:main` | MCP stdio server (`mcp` extra) — see [MCP](mcp.md) |

## Flags

```text
usage: tachyone [-h] [--preset {email,guard,moderation,router,triage}]
            [--questions QUESTIONS] [--predict] [--threshold THRESHOLD]
            [--url URL] [--backend {llm,encoder,onnx,fake}] [--model MODEL]
            [--list-presets] [--serve] [--version]
            [text]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `text` (positional) | — | the state to evaluate; required with `--predict` |
| `--preset NAME` | — | use a ready-made question set instead of `--questions` |
| `--questions JSON` | — | questions as a JSON object (alternative to `--preset`) |
| `--predict` | off | run inference and print answers; without it the CLI only prints the questions |
| `--threshold T` | — | add a sibling `handoff` object when confidence < `T` (requires `--predict`, `0 ≤ T ≤ 1`) |
| `--url URL` | — | answer against a running Tachyone server instead of locally |
| `--backend ID` | `TACHYONE_BACKEND` | `llm` / `encoder` / `onnx` / `fake` for a local run |
| `--model ID` | `tachyone-latest` | model id echoed in the request |
| `--list-presets` | — | print preset names and exit |
| `--serve` | — | start the HTTP server and exit |
| `--version` | — | print `tachyone <version>` and exit |

`--questions` and `--preset` are mutually exclusive in practice: `--questions` wins if both are
given, and the CLI errors with `provide --preset or --questions` when neither is present.

## Modes

### 1. Print the questions (offline, no model)

Resolves a preset or your JSON into canonical questions. No backend, no network, no key.

```bash
uv run tachyone "refund please" --preset triage
```

```json
{
  "model": "tachyone-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": { "type": "score", "instructions": "How urgent is this request?", "criteria": ["not urgent", "soon", "blocking"] }
  }
}
```

### 2. Predict locally

```bash
# model-free, offline — always works
uv run tachyone "refund please" --preset triage --predict --backend fake

# local encoder (needs the `train` extra for real weights; cached after the first run)
uv run tachyone "refund please" --preset triage --predict --backend encoder
```

> **The default backend is `llm`** (see `config.py` `DEFAULT_BACKEND` and the
> [ADR-0010 implementation note](adr/README.md#implementation-notes-post-acceptance)). A bare
> `tachyone "…" --predict` therefore calls an OpenAI-compatible provider and needs `TACHYONE_LLM_*`
> credentials — it is not an offline command.

### 3. Predict against a server

```bash
uv run tachyone --serve &                                  # or: uv run tachyone-serve
uv run tachyone "refund please" --preset triage --predict --url http://127.0.0.1:8000
```

The client reads `TACHYONE_API_KEY` from the environment for Bearer auth and retries `429`/`529`
with exponential backoff plus jitter.

### 4. Handoff to a System-2 model

```bash
uv run tachyone "refund please" --preset triage --predict --backend fake --threshold 0.5
```

The canonical response is unchanged; a sibling `handoff` object is appended:

```json
{
  "answers": { "...": "..." },
  "handoff": {
    "abstain": false,
    "threshold": 0.5,
    "signals": {
      "department": {
        "type": "choice",
        "confidence": 0.5,
        "entropy": 0.896,
        "margin": 0.333,
        "abstain": false
      }
    }
  }
}
```

Details and the underlying helpers: [`cookbook-handoff.md`](cookbook-handoff.md).

### 5. Ad-hoc questions

```bash
uv run tachyone "hello" --questions '{"g":{"type":"noul","instructions":"Is this a greeting?"}}' \
  --predict --backend fake
```

```json
{
  "model": "tachyone-latest",
  "answers": { "g": { "type": "noul", "noul": 0.5 } },
  "usage": { "input_tokens": 1, "output_tokens": 1 }
}
```

## Presets

```bash
uv run tachyone --list-presets     # email  guard  moderation  router  triage
```

| Preset | Questions |
| --- | --- |
| `router` | `complexity` (choice) · `needs_tools` (noul) |
| `guard` | `jailbreak` (noul) · `injection` (noul) · `data_exfiltration` (noul) |
| `moderation` | `toxicity` (score) · `harassment` (noul) · `threat` (noul) |
| `triage` | `department` (choice) · `urgency` (score) · `frustration` (score) · `churn_risk` (noul) |
| `email` | `intent` (choice) · `needs_reply` (noul) · `priority` (score) |

## Exit codes

| Code | When |
| --- | --- |
| `0` | success (including a plain question printout) |
| `1` | backend/provider failure (e.g. missing key on the `llm` backend), or an unhandled transport error |
| `2` | argparse usage error (`parser.error`) — bad `--questions` JSON, `--threshold` without `--predict`, `--threshold` outside `[0,1]`, missing `text` with `--predict` |

Validation errors from the contract (`422` for bad questions) surface as a wire error with status
`422` and are not `0`.

## Configuration

Everything else comes from environment variables — see the
[Configuration table](architecture.md#configuration) (`TACHYONE_BACKEND`, `TACHYONE_OFFLINE`,
`TACHYONE_ADAPTERS`, `TACHYONE_LLM_*`, `TACHYONE_API_KEY`, …).

For an offline/air-gapped box, prefetch weights once with
`hf download <repo> --local-dir ~/.cache/tachyone/models` (or run one warm-up prediction online),
then set `TACHYONE_OFFLINE=1`.
