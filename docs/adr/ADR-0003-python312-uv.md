# ADR-0003: Python 3.12 + uv for environment and packaging

**Status:** Accepted
**Date:** 2026-09-24

## Context

The development machine's system Python is 3.14, but torch and transformers do not yet ship
wheels for 3.14. jeba's encoder/training layers depend on both. The project also needs fast,
reproducible environment and dependency management with a committed lockfile.

## Decision

Pin the project to Python 3.12:

- `pyproject.toml`: `requires-python = ">=3.12,<3.13"`
- `.python-version`: `3.12`
- Use **uv** for environment creation, dependency resolution, and the committed `uv.lock`.
- CI runs on Python 3.12 with the same uv-based install.

## Consequences

**Positive:**
- torch/transformers work; no wheel gaps.
- Reproducible installs via `uv.lock`; fast CI and local setup.
- Single tool for venv + install + lock + run.

**Negative:**
- Contributors must use 3.12, not the system 3.14 (see B-001).
- Slightly narrower dependency ceiling until 3.14 support lands upstream.

**Neutral / follow-ups:**
- Revisit the upper bound when torch/transformers support 3.14.
- Optional Docker images should pin the same interpreter.

## Alternatives Considered

- **System Python 3.14** — rejected: torch/transformers unavailable.
- **Poetry/pip-tools** — rejected: uv is faster and gives a single coherent workflow.
- **No pin (constraint-only)** — rejected: reproducibility and CI stability at risk.

## Related

- `.specs/project/STATE.md` (B-001)
- `CONTRIBUTING.md`
