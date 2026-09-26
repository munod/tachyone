# Architecture Decision Records

ADRs capture significant, hard-to-reverse decisions for jeba. Once **Accepted**, an ADR is
not edited in place to change its decision — supersede it with a new ADR instead.

## Index

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [ADR-0001](ADR-0001-jev-drop-in-protocol.md) | Use the Jev `/v1/systemone` protocol as a drop-in contract | Accepted | 2026-09-24 |
| [ADR-0002](ADR-0002-pluggable-backend-phasing.md) | Pluggable backend with phased LLM → encoder strategy | Accepted | 2026-09-24 |
| [ADR-0003](ADR-0003-python312-uv.md) | Python 3.12 + uv for environment and packaging | Accepted | 2026-09-24 |
| [ADR-0004](ADR-0004-local-first.md) | Local-first, offline-capable core | Accepted | 2026-09-24 |
| [ADR-0005](ADR-0005-encoder-rlcd.md) | Encoder backend trained with RLCD proper-scoring calibration | Accepted | 2026-09-24 |
| [ADR-0006](ADR-0006-license-telemetry.md) | Apache-2.0 license and opt-out telemetry | Accepted | 2026-09-24 |
| [ADR-0007](ADR-0007-multilingual-mmbert.md) | Multilingual from Phase 3 via mmBERT-base | Accepted | 2026-09-24 |
| [ADR-0008](ADR-0008-llm-provider-surface.md) | LLM backend: OpenAI-compatible surface with injectable transport | Accepted | 2026-09-24 |
| [ADR-0009](ADR-0009-extension-endpoints.md) | Extension endpoints mirror the canonical wire response | Accepted | 2026-09-24 |
| [ADR-0010](ADR-0010-weights-distribution.md) | Weights fetched from the Hugging Face Hub on demand and cached locally | Accepted | 2026-09-24 |
| [ADR-0011](ADR-0011-no-telemetry.md) | No telemetry; opt-out variables reserved | Accepted | 2026-09-24 |
| [ADR-0012](ADR-0012-onnx-before-fast-path.md) | Ship the ONNX backend before the TileLang fast path | Accepted | 2026-09-24 |

## Implementation notes (post-acceptance)

Accepted ADRs are not edited in place. Where the implementation and the original wording have
drifted, the divergence is recorded here.

- **ADR-0004 — default backend.** The decision says jeba would "default to the local encoder
  backend once available (M3)". That switch was **never made**: `src/jeba/config.py` still ships
  `DEFAULT_BACKEND = "llm"`, so the default path calls an OpenAI-compatible provider and needs
  `JEBA_LLM_*` credentials. For a local, key-free run use `--backend encoder` (or
  `JEBA_BACKEND=encoder`) or the model-free `--backend fake`. Flipping the default is an open
  behaviour change, not a documentation fix — raise it before doing it.
- **ADR-0010 — prefetch.** The decision names a `jeba download` subcommand. **It was never
  implemented**: `jeba` is a single flag-based command (`src/jeba/cli.py`) with no subcommands.
  The working equivalents today are `hf download <repo> --local-dir <JEBA_MODELS_DIR>` or one
  warm-up prediction (`uv run jeba --predict --backend encoder "<any text>"`), after which
  `JEBA_OFFLINE=1` serves from cache.
- **ADR-0002 — OD-1.** The "still open (OD-1)" note predates ADR-0008, which resolved it.
- **ADR-0010 — `HF_HUB_OFFLINE`.** The ADR describes offline mode as "honoring `HF_HUB_OFFLINE`".
  jeba itself never reads or sets it — `grep -r HF_HUB_OFFLINE src/` is empty. What
  `JEBA_OFFLINE=1` actually does is pass `local_files_only=True` to every `huggingface_hub` call.
  Exporting `HF_HUB_OFFLINE=1` still works, because the *library* reads that variable on its own
  (`huggingface_hub/constants.py`), but it is not a jeba-controlled switch. For a documented,
  jeba-level guarantee use `JEBA_OFFLINE=1`; see
  [`docs/huggingface.md`](../huggingface.md) for the prefetch step offline installs need first.

## Template

```markdown
# ADR-NNNN: Title

**Status:** Proposed | Accepted | Superseded by ADR-XXXX | Deprecated
**Date:** YYYY-MM-DD

## Context

What is the issue we are facing? What forces are at play?

## Decision

What we decided, stated as a fact.

## Consequences

**Positive:** ...
**Negative:** ...
**Neutral / follow-ups:** ...

## Alternatives Considered

- **Option A** — why rejected.
- **Option B** — why rejected.

## Related

- Links to requirements, docs, or other ADRs.
```
