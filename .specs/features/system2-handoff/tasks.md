# Confidence Thresholding & System-2 Handoff — Tasks

**Spec:** `.specs/features/system2-handoff/spec.md`
**Status:** B3-T1–T4 done; all gates green; contract suite unchanged.

---

## B3-T1: Uncertainty helpers in `calibration.py`

**What:** Add `normalized_entropy` and `margin` beside `confidence`; handle empty/degenerate and
unnormalized inputs consistently.
**Where:** `src/tachyone/calibration.py`, `tests/test_calibration.py`.
**Depends on:** — · **Requirement:** CAL-06.
**Done when:** entropy returns `[0,1]` (uniform→1, one-hot→0, n≤1→0), margin is the top-two gap in
`[0,1]`, and both normalize inputs like `confidence`.
**Tests:** `tests/test_calibration.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(calibration): add entropy and margin uncertainty helpers`.

## B3-T2: `handoff.py` with assess/assess_response

**What:** New module with frozen dataclasses (`Uncertainty`, `HandoffSignal`, `HandoffReport`),
`assess(probabilities, *, threshold)` and `assess_response(response, *, threshold)`; `noul` uses its
binary certainty `max(p, 1-p)`, `choice`/`score` use `confidence` plus their distribution; reject
thresholds outside `[0,1]`.
**Where:** `src/tachyone/handoff.py`, `tests/test_handoff.py`.
**Depends on:** B3-T1 · **Requirement:** CAL-03, EXT-02.
**Done when:** the aggregate `abstain` is true when any question abstains and every signal carries
`confidence`/`entropy`/`margin`/`threshold`/`abstain`.
**Tests:** `tests/test_handoff.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(handoff): add confidence threshold and handoff signal`.

## B3-T3: Export and CLI `--threshold`

**What:** Re-export `assess`/`assess_response` from `tachyone.__init__`; add `Tachyone --predict
--threshold τ` that appends a sibling `handoff` object while keeping `model`/`answers`/`usage`
untouched; unchanged output when the flag is absent.
**Where:** `src/tachyone/__init__.py`, `src/tachyone/cli.py`, `tests/test_cli.py`, `tests/test_package.py`.
**Depends on:** B3-T2 · **Requirement:** CAL-05, NFR-R01.
**Done when:** a base install can import the helpers and the CLI prints the handoff decision
offline with `--backend fake`.
**Tests:** `tests/test_cli.py`, `tests/test_package.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(cli): add handoff threshold option`.

## B3-T4: Document the pattern and trace `CAL-06`

**What:** Add the "abstain / hand off" pattern (when, suggested τ, composing with an LLM) to the
docs, a small cookbook page with nav, and register `CAL-06` in the functional requirements and
traceability matrix.
**Where:** `docs/overview.md`, `docs/cookbook-handoff.md`, `mkdocs.yml`,
`docs/requirements/functional.md`, `docs/requirements/traceability.md`.
**Depends on:** B3-T3 · **Requirement:** CAL-06.
**Done when:** a reader finds a tested `if confidence < τ: handoff()` example and `CAL-06` is in
the matrix.
**Tests:** `tests/test_docs_site.py` (nav resolves) · **Gate:** full.
**Commit:** `docs: document the System-2 handoff pattern`.

## B3-T5: Close B-3

**What:** Run all gates and update the backlog/state.
**Where:** `.specs/project/BACKLOG.md`, `.specs/project/STATE.md`.
**Depends on:** B3-T1–T4 · **Requirement:** —.
**Done when:** gates green and B-3 marked done with a pointer to the feature spec.
**Tests:** none (process) · **Gate:** build.
**Commit:** `chore(specs): close B-3 system-2 handoff`.
