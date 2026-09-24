# ADR-0010: Weights are fetched from the Hugging Face Hub on demand and cached locally

**Status:** Accepted
**Date:** 2026-09-24

## Context

M3 loads encoder checkpoints (ModernBERT-large for English, mmBERT-base for multilingual).
M4/M6 add jeba's own fine-tuned checkpoints. OD-3 asked how these weights are distributed.
Constraints: the base install must stay offline-capable with no API key and no hosted
dependency (ADR-0004), the repo/pip must stay light, and air-gapped operation must be possible.

## Decision

Weights are **not** bundled or committed. They are fetched from the Hugging Face Hub on first
use and cached locally:

- Cache directory: `JEBA_MODELS_DIR` if set, otherwise `~/.cache/jeba/models` (XDG-style).
- Prefetch explicitly with a `jeba download` command for offline/air-gapped installs.
- Offline operation uses the cache only (`local_files_only`, honoring `HF_HUB_OFFLINE`).
- Checkpoint revisions are pinned; loading never requires an API key (public repos in M3).
- M3 uses the public base encoders with untrained task heads; trained checkpoints arrive in
  M4/M6 and follow the same policy.

Hub access is an optional dependency (the `train` extra: `huggingface_hub`, `torch`,
`transformers`); the core stays dependency-free.

## Consequences

**Positive:** Light repo/pip; offline after cache or prefetch; no key required; matches the
Laya/Needle model users expect.
**Negative:** First use needs network unless prefetched; cache and revision management add code.
**Neutral / follow-ups:** A future mirror/local-registry can be added behind the same loader
seam without changing callers.

## Alternatives Considered

- **Bundle weights in the package/wheel** — hundreds of MB to GB per encoder; pip/CI heavy.
- **Local directory only (no auto-download)** — safest but poor first-run experience.

## Related

- ADR-0004 (local-first), ADR-0007 (multilingual mmBERT), `docs/architecture.md`
- Requirements: BACK-06, ROUTE-05, NFR-S03, NFR-S04
