# ADR-0011: No telemetry; opt-out variables are reserved

**Status:** Accepted
**Date:** 2026-09-24

## Context

ADR-0006 allows telemetry only if it is opt-out and never required for function. NFR-S05
requires honoring `DO_NOT_TRACK=1`. jeba is local-first and offline-capable (ADR-0004); a
hosted telemetry endpoint would add a network dependency and a privacy surface for no
functional gain. OD-2 asked for the default and the exact variables.

## Decision

The project sends **no telemetry** and does not implement a telemetry endpoint or install id.
`jeba.telemetry` is a no-op guard: it always reports disabled and never performs network I/O.
The variables `JEBA_TELEMETRY` and `DO_NOT_TRACK` are reserved and documented so that any
future telemetry is opt-out by construction — `DO_NOT_TRACK=1` (or `JEBA_TELEMETRY=0`) must
disable it, and it must never block offline use.

## Consequences

**Positive:** Zero privacy surface; nothing to trust; no offline coupling.
**Negative:** No adoption metrics; any future telemetry needs a new ADR and endpoint work.
**Neutral / follow-ups:** If metrics are ever needed, a separate integration (not core) is the
right home.

## Alternatives Considered

- **Opt-out (on by default)** — contradicts the local-first posture and needs trust in a hash id.
- **Opt-in (`JEBA_TELEMETRY=1`)** — safer than opt-out but still unnecessary for M5 scope.

## Related

- ADR-0004 (local-first), ADR-0006 (license/telemetry), NFR-S05, OPS-08
