# Contributing to jeba

Thanks for your interest. jeba is implemented and released (`v0.2.0`); see `README.md` and
`.specs/project/STATE.md`. This guide defines the conventions that govern code contributions.

---

## Ground rules

1. **English only** for code, comments, commit messages, docs, and PRs.
2. **Conventional commits**, one logical change per commit.
3. **Contract test is law.** Any change to a public API (wire fields, primitives, error
   statuses, extension surfaces) MUST update `tests/test_contract_wire.py` in the **same
   commit**. CI fails otherwise.
4. **No secrets in the repo.** API keys come from environment variables only.
5. **No hosted dependency in core.** The base install must run offline with no API key.
   Remote/heavy dependencies belong behind optional extras.
6. **Tests are co-located.** A change that creates a code layer writes its tests in the same
   change — never deferred.

---

## Development setup

> Requires Python 3.12 and `uv` (see ADR-0003).

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install dev dependencies
uv sync

# Run the quality gates
uv run ruff check .        # lint
uv run ruff format --check .  # format check
uv run pyright             # types
uv run pytest tests/       # tests (includes the contract suite)
```

Python is pinned to **3.12** (`.python-version`, `requires-python = ">=3.12,<3.13"`). Do not
use the system 3.14 — torch/transformers do not support it yet (see ADR-0003).

---

## Commit conventions

Format: `type(scope): description`

| Type | Use for |
| --- | --- |
| `feat` | New capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `test` | Tests only |
| `docs` | Documentation only |
| `build` | Packaging/dependencies |
| `ci` | CI configuration |
| `chore` | Maintenance |
| `refactor` | Behavior-preserving change |

Scopes follow the module: `primitives`, `wire`, `backends`, `router`, `calibration`, `agent`,
`serve`, `cli`, `sdk`, `training`, `mcp`, `langchain`, `docker`, `release`.

Examples:

```
feat(primitives): add choice question and answer
fix(router): fall back to multilingual on unknown script
perf(agent): sort batch by length before forward pass
test(contract): freeze systemone wire parity
docs(adr): add ADR-0007 multilingual mmbert
```

Rules:

- One logical change per commit; do not mix refactors with features.
- Keep the subject imperative and under ~72 characters.
- Reference requirement IDs in the body when applicable (e.g., `Refs: WIRE-06, PRIM-02`).

---

## Pull requests

1. Branch from `main`; keep PRs focused.
2. Ensure all gates pass locally before opening the PR.
3. PR description: what changed, why, requirement IDs, and how it was verified.
4. If the PR touches a public API, state explicitly that the contract test was updated.
5. PRs and reviews are in English.

### Definition of done

- [ ] Code implements the task's "Done when" criteria.
- [ ] Tests written in the same change; gate passes.
- [ ] `uv run ruff check .`, `uv run pyright`, `uv run pytest tests/` all green.
- [ ] Contract test updated if a public API changed.
- [ ] Docs updated when behavior or interfaces change.
- [ ] No secrets, no hosted dependency added to core.

---

## Quality gates

| Gate | Command | Blocks merge |
| --- | --- | --- |
| Lint | `uv run ruff check .` | Yes |
| Format | `uv run ruff format --check .` | Yes |
| Types | `uv run pyright` | Yes |
| Tests | `uv run pytest tests/` | Yes |
| Contract | included in `pytest` | Yes |

See [`docs/testing.md`](docs/testing.md) for the full strategy, coverage matrix, and benchmarks.

---

## Adding a backend

1. Implement the `Backend` protocol from `src/jeba/backends/base.py`.
2. Do **not** modify `wire.py` or `docs/protocol.md`.
3. Run the shared contract suite against your backend.
4. Put optional dependencies behind a new or existing extra with an import guard.
5. Add conditional tests that skip cleanly when the extra is absent.

---

## Documentation

- Architecture decisions go in `docs/adr/` using the template in `docs/adr/README.md`.
- Requirements get stable IDs (`WIRE-01`, `PRIM-02`, …) and are traced in
  `docs/requirements/traceability.md`.
- Keep `AGENTS.md` accurate when tooling or commands change.

---

## License

By contributing, you agree your contributions are licensed under [Apache-2.0](LICENSE).