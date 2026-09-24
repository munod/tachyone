# ADR-0006: Apache-2.0 license and opt-out telemetry

**Status:** Accepted
**Date:** 2026-09-24

## Context

jeba builds on open-source work (notably Laya, Apache-2.0) and targets local-first users who
value privacy and control. Licensing and any data collection must be consistent with that
position. The reference ecosystem (Needle) uses opt-out telemetry (`NEEDLE_TELEMETRY=0`,
`DO_NOT_TRACK=1`), which is a reasonable precedent if telemetry is ever needed.

## Decision

- License jeba under **Apache-2.0**.
- If telemetry is ever added, it must be **opt-out** (honoring `DO_NOT_TRACK=1` and a
  jeba-specific variable), **disabled by default**, and **never required** for functionality.
- Never commit secrets to the repository; API keys are read from environment variables only.

## Consequences

**Positive:**
- Permissive license maximizes adoption and contribution.
- Privacy-respecting defaults align with local-first positioning.
- Clear contribution terms (Apache-2.0 inbound = outbound).

**Negative:**
- No copyleft leverage over downstream forks.
- Opt-out telemetry yields weaker adoption metrics than opt-in/required schemes.

**Neutral / follow-ups:**
- Telemetry default and variable name remain open (OD-2).
- A `NOTICE` file may be needed if bundled third-party assets require attribution.

## Alternatives Considered

- **MIT** — viable, but Apache-2.0 adds an explicit patent grant, preferable for ML tooling.
- **AGPL** — rejected: discourages commercial adoption.
- **Required telemetry** — rejected: contradicts local-first/privacy goals.

## Related

- `LICENSE`
- `docs/requirements/non-functional.md` (NFR-S01, NFR-S05)
- ADR-0004