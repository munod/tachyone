# ADR-0013: Rename the project from `jeba` to Tachyone

**Status:** Accepted
**Date:** 2026-09-26

## Context

`jeba` was picked as a placeholder while the project was being specified; it never grew into a
name the project wanted to keep. By v0.3.0 the project had shipped all milestones M0–M6, published
LoRA adapters to the Hugging Face Hub, and put a documentation site online — so the name had moved
from "harmless placeholder" to "public identity".

Renaming has a cost that grows with adoption, so the decision was made against measurements rather
than assumptions:

| Surface | Measured state on 2026-09-26 |
| --- | --- |
| GitHub repository | 0 stars, 0 forks |
| Hugging Face adapters | 0 downloads, 0 likes (both repos) |
| PyPI | never published (`/pypi/jeba/json` → 404) |
| In-repo footprint | 136 files, 1144 occurrences, 22 `JEBA_*` variables, 164 imports |

The candidate name was checked before anything was touched: PyPI, GitHub and both Hugging Face
repos were free; `tachyone` is a valid Python identifier; no internal collision.

## Decision

Rename the project to **Tachyone** everywhere:

- Python package: `src/jeba/` → `src/tachyone/`; every import and `__all__` entry.
- Console scripts: `jeba`, `jeba-serve`, `jeba-mcp-server` → `tachyone`, `tachyone-serve`,
  `tachyone-mcp-server`.
- Environment variables: the `JEBA_*` prefix (22 variables) → `TACHYONE_*`.
- Public SDK classes: `JebaClient`/`JebaError`/`JebaAPIError`/`JebaConnectionError` →
  `TachyoneClient`/`TachyoneError`/`TachyoneAPIError`/`TachyoneConnectionError`.
- Wire default: the `model` id `jeba-latest` → `tachyone-latest`, **with no legacy value kept** —
  the field is an opaque echoed string (ADR-0001), so clients that send any id keep working.
- Distribution: `munod/jeba-en` / `munod/jeba-multi` → `munod/tachyone-en` / `munod/tachyone-multi`;
  GitHub repository → `munod/tachyone`; site → `munod.github.io/tachyone/`.
- Local artefacts: cache `~/.cache/jeba/models` → `~/.cache/tachyone/models`, Docker image
  `tachyone:local`, `site_name: Tachyone`.

**Historical records keep the original name.** ADR-0001..ADR-0012 and the released sections of
`CHANGELOG.md` are records of what was decided and shipped under `jeba`; editing them would falsify
the record. Everything living — code, docs, `.specs/`, `CHANGELOG.md`'s `[Unreleased]` — uses the
new name.

## Consequences

**Positive:** one consistent identity across code, CLI, docs, Hub and site; the name can be
publicised without second thoughts; package, module, CLI and env prefix all agree.

**Negative:** any automation already written against `JEBA_*`, `jeba-serve` or `JebaClient` breaks
with no alias period. The Hugging Face repositories are renamed, so links to `huggingface.co/munod/jeba-*`
depend on the platform's redirect behaviour.

**Neutral / follow-ups:** the renamed adapters must be published under the new repository ids;
`DEFAULT_CHECKPOINTS` in `router.py` points at them; `uv.lock` records the new project name; the
wordmark SVG needed `font-size` 52 → 44 so `TACHYONE.` still fits the 450×120 viewBox.

## Alternatives Considered

- **Keep `jeba`** — zero cost now, but the name is the thing being rejected; the cost only grows.
- **Rename the product only**, keeping the package/CLI as `jeba` — rejected: two names forever is
  worse than one rename, and every install would show the old one.
- **Rename but keep `jeba-latest` as an accepted wire value** — rejected: the value is opaque and
  echoed, so there is nothing to stay compatible with, and a legacy alias would live forever for no
  benefit.
- **Keep `JEBA_*` as a deprecated alias for one release** — rejected for symmetry with the above;
  there are no existing users to protect (measured above).

## Related

- `docs/adr/ADR-0001-jev-drop-in-protocol.md` (the frozen wire this rename must not break)
- `docs/adr/ADR-0010-weights-distribution.md` (repo ids for the adapters)
- `.specs/project/STATE.md` AD-010 · `CHANGELOG.md` `[Unreleased]`
