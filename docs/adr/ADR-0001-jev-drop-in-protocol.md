# ADR-0001: Use the Jev `/v1/systemone` protocol as a drop-in contract

**Status:** Accepted
**Date:** 2026-09-24

## Context

jeba needs a public contract for atomic decision primitives. TypeSafe Jev already defines a
clear `/v1/systemone` protocol (Bearer auth, `{state, model, questions}`, three primitives,
`{model, answers, usage}`, documented errors). Building a bespoke API would force every
adopter to write a new client and would lose an existing compatibility target. The core value
proposition is "local jeba, same contract you already use."

## Decision

jeba implements the Jev `/v1/systemone` protocol **exactly** and accepts existing Jev clients
unchanged. jeba-only capabilities (router control, hooks, `predict_batch`, integrations) are
**additive extensions** that never mutate the canonical request/response shape.

## Consequences

**Positive:**
- Immediate compatibility target; existing clients work by repointing the base URL.
- A frozen contract enables a strong golden-fixture test suite that gates every backend.
- Clear primitive semantics reduce design ambiguity.

**Negative:**
- We inherit Jev's primitives and error model; some ideas must live as extensions.
- Field parity constrains internal refactors (serialization must not drift).

**Neutral / follow-ups:**
- `docs/protocol.md` is the single source of truth.
- The contract test must be updated in the same commit as any public API change.

## Alternatives Considered

- **Bespoke API** — rejected: loses existing clients and offers no offsetting benefit.
- **Superset with renamed fields** — rejected: breaks drop-in compatibility.
- **Jev-compatible but with extensions in the canonical body** — rejected: risks breaking
  strict clients that reject unknown fields.

## Related

- `docs/protocol.md`
- `docs/requirements/functional.md` (WIRE-01..07, PRIM-01..06)
- `.specs/features/wire-contract/spec.md`
