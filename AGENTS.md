# AGENTS.md

Compact, high-signal guide for agents and contributors working in this repository.

## Status

**Pre-implementation.** This repo contains **documentation only** — no `src/` code, no
`pyproject.toml`, no tests, no git history yet. Do not assume any command, module, or
dependency exists. When tooling lands (M0), replace the "planned" markers below with the
exact, verified commands.

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

## Commands (planned — not runnable yet)

These become authoritative in M0-T2. Until then, treat them as the intended interface:

```bash
uv sync                        # create env from uv.lock
uv run ruff check .            # lint
uv run ruff format --check .   # format check
uv run pyright                 # types
uv run pytest tests/           # tests (includes contract suite)
```

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