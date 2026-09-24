# AGENTS.md

Compact, high-signal guide for agents and contributors working in this repository.

## Status

**M5 (Ecosystem & Acceleration) complete.** `src/jeba/` has the wire contract, a pluggable
backend seam (LLM, model-free `FakeBackend`, local encoder, ONNX), language routing with
checkpoint lifecycle, confidence/calibration, batched single-pass inference, hooks, an optional
fast path, FastAPI serving, an SDK, a CLI, preset/schema helpers, an MCP stdio server, a
LangChain adapter, and a no-op telemetry guard. `training/` has the full data → LoRA/RLCD →
calibration → evaluation pipeline. M6 (Proof & Release) is next; the GPU training run is still
pending. Current milestone and blockers: `.specs/project/STATE.md`.

## What this project is

jeba — a local-first, multilingual decision engine that answers atomic `choice` / `score` /
`noul` questions and speaks the **TypeSafe Jev `/v1/systemone`** wire protocol as a drop-in
contract. Read first:

- `docs/overview.md` — vision, personas, non-goals
- `docs/protocol.md` — the frozen wire contract (authority)
- `docs/architecture.md` — components, flows, backend strategy
- `docs/tasks.md` — atomic task plan (M0–M6)
- `.specs/project/STATE.md` — decisions, blockers, open questions

## Stack (decided)

- **Python 3.12** — pinned via `.python-version` and `requires-python = ">=3.12,<3.13"`.
- **uv** — environment, dependencies, lockfile (`uv.lock` committed).
- **ruff** (lint/format), **pyright** (types), **pytest** (tests), **GitHub Actions** (CI).
- **pydantic v2** for the wire/primitives; FastAPI/uvicorn behind the `serve` extra.
- License: **Apache-2.0**.

## Commands (verified)

```bash
uv sync                        # create env from uv.lock (installs core + dev group)
uv run ruff check .            # lint
uv run ruff format --check .   # format check
uv run pyright                 # types
uv run pytest                  # full suite (includes the contract suite)

# run one test / one file
uv run pytest tests/test_package.py
uv run pytest tests/test_package.py::test_version_is_exposed
```

Ruff excludes `.opencode/`, `docs/`, `.specs/`, `docker/` (third-party skills and code
fences are not linted). Optional extras are installed on demand; the server, integration,
and e2e tests need `serve`:

```bash
uv sync --extra serve          # adds fastapi + uvicorn (+ httpx from the dev group)
uv run pytest -m "not e2e"     # skip the port-binding end-to-end tests
uv run jeba --predict --preset triage --backend fake "refund please"   # offline demo
```

The default backend is `llm` (needs `JEBA_LLM_*`); use `--backend fake` or `JEBA_BACKEND=fake`
for a model-free run, and `JEBA_BACKEND=encoder` for the local encoder. The encoder's real
weights need the `train` extra (`uv sync --extra train`); the decision math and all tests work
without it. Weights are fetched on demand (ADR-0010); tests use an injected fake encoder.

Training lives in `training/` (not installed): `uv run python -m training.generate_data --out data/train.jsonl`,
`uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json --dry-run`,
`uv run python -m training.fit_calibration --calibration data/preds.jsonl --out temperature.json`,
`uv run python -m training.evaluate --data data/eval.jsonl --out report.json`. Heavy modules
(torch/transformers/peft) are imported lazily behind the `train` extra.

Integration extras are similarly lazy: `JEBA_BACKEND=onnx` needs `--extra onnx`; `jeba-mcp-server`
needs `--extra mcp`; `jeba.integrations.langchain` needs `--extra langchain`; the fast path needs
`--extra fast`. Each raises a clear error naming the extra when it is missing.

## Directory boundaries (planned)

| Path | Belongs here | Does NOT belong here |
| --- | --- | --- |
| `src/jeba/primitives.py` | Primitive models + validators | I/O, model code |
| `src/jeba/wire.py` | `/v1/systemone` envelope + dispatch | Backend-specific logic |
| `src/jeba/backends/` | Backend implementations behind `base.py` | Wire changes |
| `src/jeba/agent.py`, `router.py`, `calibration.py` | Local inference internals | HTTP concerns |
| `src/jeba/serve.py`, `cli.py`, `client.py` | Transport + UX | Model internals |
| `training/` | Data gen, fine-tuning, calibration scripts | Runtime code |
| `tests/` | Contract, unit, integration, e2e, conditional | Fixtures with secrets |
| `benchmarks/` | Latency/throughput/ECE harnesses | Unit tests |
| `docs/` | Human-facing docs | Generated artifacts |
| `.specs/` | TLC project memory + feature specs | Source code |

## Hard rules

1. **Contract test is law.** Any public API change (wire fields, primitives, error statuses,
   extension surfaces) MUST update `tests/test_contract_wire.py` in the **same commit**.
2. **No hosted dependency in core.** Base install must run offline with no API key. Remote or
   heavy deps go behind extras: `serve`, `fast`, `onnx`, `langchain`, `mcp`, `train`.
3. **Additive extensions only.** Router control, hooks, `predict_batch` must never change the
   canonical `/v1/systemone` response shape.
4. **No secrets in the repo.** API keys come from env vars (`JEBA_API_KEY`, etc.).
5. **Conventional commits**, one logical change per commit; English docs/PRs.
6. **Tests are co-located** with the code they cover — never deferred to a later task.

## Pitfalls

- **Python 3.14 vs torch/transformers:** the system interpreter is 3.14, but torch/transformers
  do not support it yet. Always use the pinned 3.12 (see ADR-0003, blocker B-001).
- **Do not "fix" the wire to be nicer.** Field names, nesting, and error statuses are frozen
  for Jev compatibility (ADR-0001).
- **Do not add a required network call** to the default path (ADR-0004).
- **RTX 3060 12GB** is the training envelope: LoRA/QLoRA only, no full fine-tuning (ADR-0005).
- **Open decisions** (OD-1..OD-5) are listed in `.specs/project/STATE.md`; do not silently
  resolve them in code — raise them.

## Workflow

This repo follows the TLC spec-driven flow: Specify → Design → Tasks → Execute, with
`.specs/` as persistent memory. Before implementing, read the relevant
`.specs/features/<phase>/spec.md` and the matching tasks in `docs/tasks.md`. Update
`.specs/project/STATE.md` when decisions, blockers, or learnings change.