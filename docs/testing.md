# Testing & Quality Strategy

**Status:** Active. Gates are real as of M0; the contract suite is live as of M1
(`tests/test_contract_wire.py`, `tests/test_contract_errors.py`).

---

## Principles

1. **Contract-first.** The `/v1/systemone` contract test is the highest-priority suite. Any
   public API change MUST update it in the same commit (Laya convention, adopted in ADR-0001).
2. **Tests are co-located with code.** A task that creates a code layer writes its tests in the
   same task — never deferred to a later task.
3. **Backends are interchangeable.** Every backend passes the same contract suite; no backend
   gets a private contract.
4. **Optional deps are conditional.** Tests for extras (`serve`, `onnx`, `mcp`, `langchain`,
   `train`) skip cleanly when the extra is not installed (Laya-style conditional tests).
5. **No silent test deletion.** Gate checks record test counts; a decrease must be justified.

---

## Test types

| Type | Purpose | Location pattern | Runs when |
| --- | --- | --- | --- |
| **Contract** | Jev wire parity, primitives, error shapes | `tests/test_contract_*.py` | Always |
| **Unit** | One function/class in isolation | `tests/test_*.py` | Always |
| **Integration** | Server + backend + wire together | `tests/integration/` | Always (offline) |
| **E2E** | Repointed Jev client against a live server | `tests/e2e/` | Always |
| **Conditional** | Extra-specific behavior | `tests/test_onnx_backend.py`, `tests/test_mcp.py`, `tests/test_langchain.py` | Only if extra installed |
| **Benchmark** | Latency/throughput/ECE | `benchmarks/` | On demand / M4+ |

---

## Test Coverage Matrix

| Code Layer | Required Test Type | Location Pattern | Run Command |
| --- | --- | --- | --- |
| `primitives.py` | unit | `tests/test_primitives_*.py` | `uv run pytest tests/test_primitives_*.py` |
| `wire.py` | unit + contract | `tests/test_wire.py`, `tests/test_contract_wire.py` | `uv run pytest tests/test_contract_wire.py` |
| `backends/base.py` | unit | `tests/test_backends_base.py` | `uv run pytest tests/test_backends_base.py` |
| `backends/llm.py` | unit + integration | `tests/test_backends_llm.py` | `uv run pytest tests/test_backends_llm.py` |
| `backends/encoder.py` | contract + integration | `tests/test_contract_wire.py`, `tests/integration/` | `uv run pytest tests/` |
| `backends/onnx.py` | contract + integration (conditional) | `tests/test_onnx_backend.py` | `uv run pytest tests/test_onnx_backend.py` |
| `router.py` | unit | `tests/test_router.py` | `uv run pytest tests/test_router.py` |
| `calibration.py` | unit | `tests/test_calibration.py` | `uv run pytest tests/test_calibration.py` |
| `agent.py` | unit + benchmark | `tests/test_agent.py` | `uv run pytest tests/test_agent.py` |
| `hooks.py` | unit + hook-contract | `tests/test_hooks_api.py` | `uv run pytest tests/test_hooks_api.py` |
| `serve.py` | integration | `tests/integration/test_serve.py` | `uv run pytest tests/integration/` |
| `client.py` | unit + integration | `tests/test_client.py` | `uv run pytest tests/test_client.py` |
| `cli.py` | unit + smoke | `tests/test_cli.py` | `uv run pytest tests/test_cli.py` |
| `schemas.py` | unit | `tests/test_schemas.py` | `uv run pytest tests/test_schemas.py` |
| `training/*` | unit + evaluation | `tests/test_training_*.py` | `uv run pytest tests/test_training_*.py` |
| `mcp/`, `integrations/` | integration (conditional) | `tests/test_mcp.py`, `tests/test_langchain.py` | `uv run pytest tests/test_mcp.py tests/test_langchain.py` |

---

## Parallelism Assessment

| Test Type | Parallel-Safe? | Isolation Model | Evidence |
| --- | --- | --- | --- |
| Contract | Yes | Pure functions + fixtures; no shared state | `tests/test_contract_wire.py` uses in-memory fixtures |
| Unit | Yes | No I/O; deterministic | `FakeBackend` injected |
| Integration (server) | Yes | Ephemeral port per test; no shared DB | `httpx` ASGI transport or per-test server |
| E2E | No (serialize) | Live server + client; port binding | `tests/e2e/` binds a port |
| Conditional extras | Yes | Skipped unless extra present | `pytest.importorskip` |
| Benchmark | No (serialize) | GPU/CPU contention skews latency | `benchmarks/` runs alone |

**Rule:** tasks whose required test type is **not** parallel-safe must run sequentially even if
their code has no dependencies.

---

## Gate Check Commands

| Gate Level | When to Use | Command |
| --- | --- | --- |
| **Quick** | After tasks with unit tests only | `uv run pytest tests/ -q -x` |
| **Full** | After tasks with integration/e2e/contract tests | `uv run pytest tests/ && uv run ruff check . && uv run pyright` |
| **Build** | After phase completion | `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest tests/` |

> These commands are what CI runs (`.github/workflows/ci.yml`); the **Build** level is the
> documented phase gate in `docs/tasks.md`.

---

## Contract test rules

1. `tests/test_contract_wire.py` encodes `docs/protocol.md` exactly.
2. Golden fixtures live in `tests/fixtures/` and are captured from the documented Jev shapes.
3. Any change to a public field, type, or error status requires updating the contract test in
   the **same commit**; CI fails otherwise.
4. Every backend (LLM, encoder, ONNX) runs the same contract suite.
5. Hooks must not alter the canonical shape — verified by running the contract suite with hooks
   registered (`tests/test_hooks_api.py`).

### What the contract suite covers

- All three primitives: valid payloads, boundary limits (choice 255/256, score 2/10/1/11).
- Multi-question requests: N in → N out, ids echoed.
- Response invariants: probability keys, normalization, `usage` presence.
- Error shapes: 401, 422, 429, 529.
- Additive extensions do not change canonical fields.

---

## Conditional tests by extra

Pattern (Laya-inspired):

```python
# Illustrative only.
import pytest

pytest.importorskip("onnxruntime")  # skip cleanly when the extra is absent

def test_onnx_backend_contract():
    ...
```

This keeps the base test run green without optional dependencies while still testing extras
when installed.

---

## Benchmarks & reproduction

| Benchmark | Measures | Command |
| --- | --- | --- |
| Evaluation (accuracy / ECE) | per-primitive and per-language accuracy + ECE | `uv run python -m training.evaluate --data data/eval_multi.jsonl --out benchmarks/results/eval.json` |
| Fast path | p50/p95 and answer parity, stock vs `TACHYONE_FAST` | `uv run python -m benchmarks.fast_path --data data/eval_multi.jsonl --model-id jhu-clsp/mmBERT-base --adapter checkpoints/multi --out benchmarks/results/fast_path.json` |
| Report renderer | Markdown report from evaluation JSON artifacts | `uv run python -m benchmarks.report --entry encoder=<report.json> --out benchmarks/report.md` |
| Public probes | MASSIVE / XNLI / typed-decisions | **not yet delivered** (see `docs/benchmarks.md` → Known limitations) |

Reproduction rules:

- Every benchmark records hardware, versions, seed, and exact command in its output artifact.
- Benchmark runs are serialized (no parallel test execution) to avoid contention.
- Results are committed as artifacts under `benchmarks/results/` with a timestamp.

---

## CI

`.github/workflows/ci.yml`:

1. **quality** — checkout, install uv (Python 3.12), `uv sync --locked --extra serve`,
   `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest`.
2. **base-install** — `uv sync --locked --no-dev`, then import the core package offline
   (NFR-R01: no network, no API key).
3. **docs** — `uv sync --locked --group docs`, then `uv run mkdocs build --strict`.

Contract tests are not a separate step: they run inside `uv run pytest`, which blocks the
`quality` job. Any public wire or primitives change must update `tests/test_contract_*.py` in the
same commit (NFR-M03).

---

## Quality gates summary

| Gate | Blocks |
| --- | --- |
| ruff (`check` + `format --check`) | Merge |
| pyright | Merge |
| pytest (incl. contract) | Merge |
| Contract test updated with public API change | Merge (review rule; CI runs the suite, the pairing rule is enforced in review) |
| Test count not decreased without justification | Merge (review rule — not automated in CI) |