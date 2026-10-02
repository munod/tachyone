# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-10-02

### Added

- **Five-domain multilingual coverage (B-5b).** The four new domain lexicons (`ecommerce`,
  `agent_tools`, `documents`, `voice`) now ship complete tables for all seven training languages
  (`support` already did); `test_every_committed_domain_file_is_complete` iterates
  `DEFAULT_LANGUAGES`, because a table that only ships English puts English text under a
  non-`en` `lang` tag — the L-008 artifact. Three review passes (read-aloud of generated
  records + a full template × filler combinatorial audit, 14,082 combinations) keep the
  composed sentences grammatical per language.
- **Multilingual five-domain datasets:** `training/configs/data_multi_domains.json` /
  `data_eval_multi_domains.json` → 30,000 train (support keeps its 18,000; four new domains
  3,000 each) and 7,500 eval records over `pt es fr de it nl`; both support halves are
  byte-identical to `train_multi`/`eval_multi` modulo the `domain` key (test-enforced), so the
  no-regression gate measures the published rows.
- **`report["per_domain_language"]`** — accuracy/ECE per `domain/language` cell, emitted only
  when the eval set carries more than one domain (legacy reports stay byte-identical) and
  rendered by `benchmarks/report.py` worst-cell-first with a `Worst cell (accuracy)` line.

### Changed

- **`munod/tachyone-multi` republished (commit [`9a3ef5a5`](https://huggingface.co/munod/tachyone-multi/commit/9a3ef5a5d38972ffe119cdfe6b4b6c7bb14da9d1)).**
  The artifact is the B-5b joint run (30k five-domain multilingual records, mmBERT, LoRA r=64,
  8 epochs) with the `choice` heads re-fitted on the **frozen** trunk
  (`training/configs/fit_bank_multi_domains.json`, recipe shipped as `choice_bank_fit.json`);
  the stale `finetune_config.json` was deleted and all six uploaded files were sha256-verified.
  On the five-domain split: **0.561 → 0.9975** overall (worst new domain 0.438 → 0.993, `choice`
  0.620 → 0.992, gate strict **1.000**, per-domain ECE 0.0005–0.0038 with zero exceptions, all
  six languages ≤ 0.004, held-out-text 0.996). On the support-only *routed* split the
  product-level number moves **0.743 → 0.895**; gates were fixed from the previous adapter's
  baseline (0.561 / support 0.8413) *before* training — table in `.specs/project/BACKLOG.md` B-5b.
- **`benchmarks/report.md` gains a fourth entry** (`multilingual five-domain — B-5b fitted
  choice-head bank`) with the gate row, the domain × language cross table and the held-out-text
  split; the `multilingual` entry was re-measured on the new weights (0.743 → 0.895) and the
  reproduction commands now follow the multi chain (generate → joint run → frozen-trunk fit →
  predict), including the previously stale render invocation.
- **Public probe MASSIVE re-run against the released B-5b bytes:** 0.011 → **0.039** (chance
  0.017; every language now above chance at 0.025–0.055) with `ECE raw` 0.354 → 0.518 and
  `Conf` 0.365 → 0.557 — sharp off-domain, the same trade English showed at B-5; the fit is the
  only probe off the grid ceiling (T=14.55), so `benchmarks/probes.py` now renders its
  temperature line from the artifacts and skips `*_pre*` backup files so the documented
  `probe_*.json` glob reproduces the page exactly. typed-decisions and XNLI are English-routed
  and unchanged.
- **Fast path re-checked on the multilingual checkpoint:** 3.66× p50 (13.97 → 3.82 ms), **0
  top-label flips** (`benchmarks/results/fast_path_multi.json`); English 2.59× / 0 flips
  (`fast_path_en.json`) and the replaced multilingual weights 3.76× / 0 flips (`fast_path.json`).
- README, `docs/model-card.md`, `docs/benchmarks.md` and `docs/huggingface.md` carry the new
  set; `docs/benchmarks.md`'s multilingual section is now the B-5b artifact with both splits.

## [0.5.0] - 2026-10-01

### Added

- **Per-domain `choice` heads behind a deterministic gate (ADR-0016).** `choice_head.json` grows
  to `{shared, domains: {name: {head, signatures}}}`; the legacy single-scorer shape loads as a
  shared-only bank, so every published adapter answers bit-for-bit as before. Exactly one head
  per `choice` question — caller hint → lexical signature match (pure Python, no encode, offline)
  → shared head — so a wrong gate can at worst reproduce today's numbers, and a corrupt entry
  warns by name (file + key) and drops only itself (B-8).
- **Optional `choice_head` request hint**, additive and contract-tested in the same commit
  (`docs/protocol.md` extension table, CLI `--choice-head`,
  SDK `system_one(..., choice_head=)`); the backend seam gained `choice_head=` as a keyword with
  a default, and the canonical response shape is untouched.
- **The bank trainer and the frozen-trunk fitter.** `training/finetune_rlcd.py` learns a shared
  head plus one per domain when the records carry `domain` (records without it — every legacy
  config — still write the legacy asset byte-identically); `training/fit_choice_bank.py` fits a
  shared-head **control** and the **bank** against a *frozen* trunk from a single encode pass and
  writes two loadable adapter dirs, each carrying its own `choice_bank_fit.json` recipe (L-011),
  with `--dry-run` to validate config and data without torch.
- **Gate and integrity rows in the evaluation report.** `report["choice_gate"]` (strict /
  fell-to-shared / wrong-domain, overall and per domain, plus `answers_routed_by`) beside
  `per_domain`; `training/predict --head-hint domain` runs the oracle arm; `--train-data` adds
  `report["text_seen"]`, splitting accuracy by whether the row's input text occurs in training;
  `benchmarks/report.py` renders both.
- **Label audit beside the accuracy:** `training/evaluate.py` now emits `report["noul_per_language"]`
  and `report["noul_labels"]` (per-language contradictory-label rate, judged against the same
  phrase banks; unreadable states count as `unknown`, never as contradictions), and
  `benchmarks/report.py` renders both in one table so label noise can no longer be read as a
  capability gap. Acceptance tests cover the generator, the shipped eval sets and the renderer.
- `docs/adr/ADR-0014-noul-label-from-text.md` (scope decision) and backlog **B-12** (the analogous
  `score` near-tie label, measured at a 0.922 text-consistent ceiling).

### Fixed

- **`noul` labels are derived from the text, not from the loop index (B-11 / ADR-0014).**
  `_noul_record` computed `target = 1 iff even index and request tone`, which contradicted **121
  of the 241 request-toned rows in `eval_en`** and — because language is
  `index % len(languages)` — left **every** `noul` label in `es`/`de`/`nl` at 0 while `pt`/`fr`/`it`
  came out tone-consistent by accident (lesson L-010: a constant label is a shortcut the model
  wins with). The rule is now `request` → 1, `neutral`/empty → 0, asserted against each state's
  own phrase bank across all five committed domains and seven languages. Only `noul.target` bytes
  move: states, questions and every `choice`/`score` record are byte-identical, the golden hashes
  were recaptured, and the seven shipped datasets were regenerated.
- **`score` and `choice` labels are derived from the text too (B-12 / ADR-0015).** `_score_record`
  lowered the level one step on `index % 13 == 0` as an "ambiguous near-tie", contradicting **7.8%
  of `score` rows** (39/500 in both eval sets) and capping the primitive at **0.888** — exactly
  where the released checkpoints sat (0.882 English). The empty boundary state (5.4% of *every*
  primitive) carried an index-derived label in `score` (the tone drawn before the state was
  emptied) and in `choice` (`options[index % 4]`). Now `score` takes the tone's level with the
  empty state at the middle level, `choice` takes the option the state names with the empty state
  at the catch-all, and **`ecommerce` gained the `other` option** every other domain already
  shipped (re-cycling its four options into five). Acceptance tests check each primitive against
  its own phrase/option banks across five domains and seven languages; golden hashes recaptured;
  seven datasets regenerated (`noul` untouched).

### Changed

- **The B-5 / ADR-0016 isolate was run and every gate now passes with margin** (four arms on one
  harness; `benchmarks/results/en_domains_*.json`, numbers in `.specs/project/BACKLOG.md` B-5):
  five-domain accuracy **0.879 → 0.964**, `support` **0.964 ≥ 0.85**, worst new domain
  **0.963 ≥ 0.70**, gate strict **1.000** (0 to shared, 0 wrong domain) over 2,500 `choice` rows,
  per-domain ECE **0.021–0.027** — all five domains under the 0.05 target for the first time — and
  the gap to the released adapter's `support` cell **0.086 → 0.008**. On the 473 `choice` rows
  whose text never occurs in training, 0.892 → 0.998. The shared-head control sits **3 of 2,500**
  rows behind the bank, so refitting the head on the corrected labels — not per-domain capacity —
  closed the gap (lesson L-012). **Released as `munod/tachyone-en`** (revision recorded in
  `docs/huggingface.md` §3): on the support split `choice` 0.946 → **1.000** while `noul`/`score`
  move 0.992/0.978 → 0.946/0.946 and overall 0.972 → 0.964, and the five-domain split goes
  0.511 → 0.964 (worst domain 0.328 → 0.963). Latency was re-measured on the L4 reference box and
  is labelled as such wherever it appears — it is not comparable with the earlier columns. The
  three public probes were re-run against the new revision (`benchmarks/probes.md`):
  typed-decisions 0.330 → **0.269**, XNLI 0.333 → **0.341**, MASSIVE unchanged at 0.011, and the
  off-domain confidence the new head carries is published beside them (`Conf` up to 0.991,
  `ECE raw` 0.268 → 0.650 on XNLI). The head-to-head was re-scored with the same discipline —
  Tachyone only, peers and gold labels untouched: their turf **0.236 → 0.233**, our turf **0.974
  → 0.958** (`choice` **1.000**, `noul`/`score` 1.000/0.984 → 0.938/0.938).
- **Both adapters retrained and the whole measured surface republished as one set** (AD-009
  pattern; pre-B-11 numbers stay valid for the data that produced them and are not comparable
  line-by-line):

  | checkpoint | pre-B-11 labels | on the corrected labels | published |
  | --- | ---: | ---: | ---: |
  | English | 0.859 (`noul` 0.718) | 0.935 (old weights) | **0.945** (`noul` **0.992**) |
  | Multilingual | 0.853 (`noul` 0.960) | 0.714 (old weights) | **0.718** (`noul` **0.832**, 8 epochs) |

  The multilingual figure went through a four-run sweep (control proved the 4-epoch drop was
  systematic, not L-006 variance; 8 epochs recovered `score` 0.648 → 0.854; `choice_rank` 128
  overfit) and ships `choice` 0.468 against 0.674 for the old weights — a real cost of the
  balanced `noul` task, published in `docs/benchmarks.md` and `BACKLOG.md` **B-11** rather than
  hidden. Fast path re-measured on the new weights (9.19 → 3.37 ms, **2.72×**, parity 0.000771,
  0 top-label flips); the three public probes re-run (typed-decisions 0.323 → **0.330**, XNLI
  0.334 → **0.333**, MASSIVE 0.033 → **0.013**, below its 0.017 chance and tracking that same
  `choice` regression); and the head-to-head re-run with **all four engines on table B**, since
  the gold labels under it changed (Tachyone 0.854 → **0.953** at home, 0.229 → **0.241** on the
  peer's probe).
- **Everything re-measured again for B-12, and the published set re-chosen by merit.** Training on
  the corrected labels makes `noul` and `score` trivially learnable (both hit **1.000** — they are
  the same tone detector once the near-tie noise is gone) and the shared trunk pays for it in
  `choice`, which is what the cycle measured: an English **control of the identical published
  recipe landed at 0.841** (L-006), the five-domain retrain's `choice` collapsed to **0.303**
  (train loss 0.027, eval at chance), while the multilingual retrain *won*. So the published
  artifacts are the **best measured per adapter**:

  | checkpoint | published | overall | on pre-fix labels |
  | --- | --- | ---: | ---: |
  | English | B-11 weights | **0.972** (`choice` 0.946, `score` 0.978) | 0.859 / 0.945 |
  | Multilingual | B-12 retrain (8 epochs) | **0.743** (`choice` 0.468) | 0.853 / 0.718 |
  | five-domain (B-5a) | run 5 weights | **0.879** — **support 0.886 ✓, worst new 0.815 ✓** | 0.785 ✗ / 0.861 ✓ |

  Re-runs on the published set: fast path **2.65×** (8.97 → 3.39 ms, parity 0.000559, 0 flips),
  probes 0.330 / **0.011** / 0.333, head-to-head table A **0.236** (peers untouched) and table B
  **0.974** at home with all four engines re-scored on the B-12 gold labels. The B-5 gates now
  pass on both label generations, and the "retrain run 5 on corrected labels" option was tried and
  **lost to the B-11 weights by 0.111** — recorded in `docs/benchmarks.md` and `BACKLOG.md` **B-5**.

### Documentation

- **The `B-5a` multi-domain experiment passes both gates, and the "retrain it" option was tried
  and measured (2026-09-29).** On the pre-B-11 labels run 5 scored 0.785 on `support` ✗; on the
  B-11 labels the same weights reached **0.861 ✓**; on the **B-12 labels** they reach **0.879
  overall with support 0.886 ✓ and the worst new domain `voice` 0.815 ✓**, per-domain ECE
  0.036–0.080. Retraining run 5's recipe on the corrected labels — the "cheap untried option" the
  previous entry named — **was run and lost**: `noul` and `score` both hit 1.000 while `choice`
  collapsed to 0.303 and `support` fell to 0.763 ✗, so the designated artifact keeps its B-11
  weights. **Nothing was released** — the published adapters are the support-only ones (English
  0.972). `docs/benchmarks.md` carries both tables and the gate verdicts; the six-run curve and
  the three structural options live in `.specs/project/BACKLOG.md` **B-5**.

## [0.4.0] - 2026-09-27

### Added

- **`docs/compare.md` — the three questions a prospective user asks first:** why not a small
  model on Ollama/llama.cpp, why not Jev, why not another open System One scorer. Measured
  figures where they exist, explicit notes where they do not, and the composition answer
  (Tachyone first, an LLM when `confidence < τ`) linking the System-2 cookbook. The tagline
  *"LLMs generate text. Tachyone produces calibrated decisions"* now sits in the landing hero,
  the README blockquote, the overview value proposition and the mkdocs site description.
- **`docs/use-cases.md` — five recipes with real JSON in and JSON out:** support-ticket
  classification, ticket routing, incident prioritization, risk assessment and document triage.
  The payloads live in `docs/assets/examples/*.json` and are embedded with `pymdownx.snippets`;
  `tests/test_docs_examples.py` validates every pair against the wire (question/answer key parity,
  probabilities over the declared options, `confidence` = selected mass, `noul` without
  `confidence`) and fails if an example stops being embedded.
- **Integration hub and recipes:** `docs/integrations.md` plus FastAPI, n8n, Power Automate,
  Azure Functions and LangGraph pages. Every HTTP payload shown was executed against a local
  `tachyone-serve` (200 with the documented shape; 422/401/429/529 rows come from
  `test_serve.py`), and each recipe carries a verification label distinguishing *shipped +
  tested* from *payload verified* — the third-party runtimes are not run in CI. The LangGraph
  flow is a real script (`docs/assets/examples/langgraph_flow.py`) executed by the test suite.
- **Head-to-head comparison harness (`benchmarks/compare.py`)** answering the same rows with the
  local encoder, the open peer scorer `pngwn/system-one-qwen3.5-4b-scorer` (loaded from a local
  download through its own `system_one.py` — nothing third-party is vendored) and any
  OpenAI-compatible server behind Tachyone's `llm` backend. One metric implementation, both
  evaluation distributions, temperature fitted per engine on the capped validation split, peak
  RSS/VRAM of the process that did the inference, and contract compliance counted per question.
  `tests/test_benchmark_compare.py` covers the metrics, the temperature fit, the row → wire
  mapping and the renderer. Results, method and limitations published in `docs/compare.md` §3;
  the run reproduces the peer's own model card (T 1.75, accuracy 0.705, ECE 0.046).
- **The three public probes of `B-7` (MASSIVE / XNLI / typed-decisions), published as
  `benchmarks/probes.md`.** `benchmarks/probes.py` maps each dataset onto the harness row shape —
  typed-decisions keeps its native `noul`/`choice`/`score` wire with multi-question rows flattened
  one-per-question, MASSIVE becomes 60-way `choice` with `task` = language (so the per-language
  accuracy/ECE table `B-1` needed falls out of one run), XNLI becomes 3-way `choice` — and
  `python -m benchmarks.compare run --format probe --probe <name>` evaluates them through the same
  engines, metrics and per-dataset temperature fit. `python -m benchmarks.probes render`
  regenerates the report with licences, citations, chance levels and the reproduction commands.
  Measured on the RTX 3060 with the released adapters, evaluation only: **typed-decisions 0.323**
  (chance 0.20–0.50), **MASSIVE 0.033** over seven locales (chance 0.017; per language
  0.016–0.047), **XNLI 0.334** (chance 0.333) — all three temperature fits hit the grid ceiling, so
  `Conf` and `Brier` are published next to `ECE cal`. XNLI's licence was verified against
  `facebookresearch/XNLI`: **CC BY-NC 4.0**, and only derived metrics are published. 14 tests in
  `tests/test_benchmark_probes.py`, one per probe's label mapping.

- **Multi-domain synthetic data (B-5a refactor, in progress).** `training/generate_data.py` now
  reads committed per-domain content from `training/data/domains/<domain>.json` — `lexicon.json`,
  `phrases.json` and `team_descriptions.json` merge into `domains/support.json` — and
  `DataConfig.domains` / `--domains` choose which domains to emit (default `support`, so every
  existing config regenerates **byte-for-byte**: golden hashes plus a config-vs-dataset test pin
  it). Four new English domains ship (`ecommerce`, `agent_tools`, `documents`, `voice`), records
  carry a `domain` field for non-default runs, `training/evaluate.py` reports `per_domain`
  (rendered by `benchmarks/report.py` together with the worst-domain line), `DataConfig.per_domain`
  keeps an incumbent domain's record volume while new domains are added smaller, and four presets
  follow the data (`orders`, `tools`, `docs`, `voice`). **The published adapter is unchanged** —
  the multi-domain checkpoint is still in training (gates: support ≥ 0.85, worst new domain ≥ 0.70).

### Changed

- **Renamed `jeba` → Tachyone (ADR-0013).** Package `src/tachyone/`, console scripts `tachyone`,
  `tachyone-serve`, `tachyone-mcp-server`, env prefix `TACHYONE_*`, SDK classes `TachyoneClient` /
  `Tachyone*Error`, wire default model id `tachyone-latest`, Hub repos `munod/tachyone-en` /
  `munod/tachyone-multi`, site `munod.github.io/tachyone/`. No alias period: `JEBA_*`, `jeba-serve`
  and `JebaClient` are gone. ADR-0001..ADR-0012 and the released sections below keep the original
  name as historical record.

- **English adapter reseeded (B-9).** The published `training/configs/finetune_en.json` recipe run
  again from `seed: 2` produced a checkpoint that beats the previously published one on seven of
  eight metrics: overall accuracy **0.763 → 0.859**, overall ECE **0.061 → 0.023**, `choice`
  0.834 → 0.948, `score` 0.712 → 0.910 and their ECEs 0.074 → 0.020 / 0.042 → 0.039. Only `noul`
  accuracy moved against it (0.744 → 0.718), while `noul` ECE improved 0.101 → 0.020. The config
  now pins `seed: 2`, `munod/tachyone-en` was republished from it, and `benchmarks/report.md` was
  regenerated. Background: `.specs/project/BACKLOG.md` B-9.

### Fixed

- **The `llm` backend's prompt under-specified the response wrapper (B-10).** `_SYSTEM_PROMPT`
  documented `{"answers": ...}` only inside its `noul` example, so a model that took the `choice`
  example literally answered with the *inner* object and every failure cost
  `TACHYONE_LLM_RETRIES + 1` autoregressive attempts. The prompt now states the wrapper once for
  all three primitives with a complete example and bans the value errors actually observed
  (boolean `noul`, the criteria text echoed back, a bare string instead of an answer object, an
  option label used as the question id); `build_answers` names the missing `answers` key and echoes
  the top-level keys it saw. **Parsing stays strict** — the inner shape is still rejected, so the
  `JSON ok` column keeps measuring the model rather than the parser. Re-run of the four LLM rows
  with the same servers, flags and rows: probe `JSON ok` **0.083 → 0.889** (`ling-tiny`) and
  **0.889 → 1.000** (`ornith-9b`), home **0.417 → 0.917** and **0.833 → 1.000**; accuracy/ECE are
  recorded before *and* after in `docs/compare.md` §3 because the prompt moves the distribution.
- **A calibration asset that cannot be loaded no longer disappears without a trace (B-8).**
  `load_temperatures()` / `load_choice_head()` caught every exception and returned the empty
  baseline, so a partially populated cache under `TACHYONE_OFFLINE=1` — or a corrupt
  `temperature_calibration.json` / `choice_head.json` — left the encoder answering *uncalibrated*
  with no indication. They now log a WARNING naming the asset and the source (stderr) for anything
  other than "this adapter does not ship that file": `LocalEntryNotFoundError` (a partial prefetch
  looks exactly like this), a mis-typed adapter id, a network error, and files that are unreadable,
  not a JSON object, hold a non-numeric temperature or an invalid scorer. A corrupt temperature
  value, which used to raise out of the loader, now degrades with the same warning. Nothing raises
  — the change is log-only by design. Message documented in `docs/huggingface.md` §4.
- `benchmarks/report.py` split `--entry` on the **first** `=`, which broke on the report's own
  entry names (`english (... r=16 ...)`) with a misleading `FileNotFoundError`; it now splits on
  the last one.
- The report's hardcoded Reproduce block pointed at files that do not exist (`data/eval.jsonl`)
  and omitted the predict / fit-calibration / evaluate / report steps, so a regenerated report
  could not be reproduced from its own instructions.
- **Landing hero title was nearly invisible on the light theme.** The hero rules were bare
  classes (specificity 0-1-0), so Material's `.md-typeset h1` (0-1-1) won despite `extra.css`
  loading last and painted the title with `--md-default-fg-color--light` (`#0000008a`) over the
  dark hero gradient — it also silently replaced the title's size, weight and margins. Every
  `.tachyone-hero__*` rule is now scoped under `.tachyone-hero` (0-2-0, no `!important`), guarded by
  `tests/test_docs_site.py::test_hero_styles_are_scoped_against_theme_overrides`.

- **Two training-loop bugs behind the multi-domain collapses (B-5a).** `val_split` took the *tail*
  of the file, so on domain-ordered data the validation set was the tail of the last domain (the
  published `val_loss` 0.326 turned out to be `support`/`score` alone); it now shuffles with the
  config seed (`split_records()`). Every optimizer step was also 32 consecutive records of one
  primitive and one domain, with every epoch ending on `score` — textbook multi-domain
  interference — so epochs now permute each primitive's records (`shuffled()`, seeded by
  `seed:epoch:kind`). Both fixes moved the five-domain run from 0.630 to 0.804 accuracy; 3 tests,
  plus a per-epoch per-primitive train-loss line so a plateau is visible without a second run.
- **Two committed data configs did not reproduce the data they document.** `data_multi.json` and
  `data_noisy.json` claimed `seed: 42` while the shipped datasets were generated with seed 1
  (hash-verified); `data_en.json`, `data_eval_en.json` and `data_eval_multi.json` now document the
  datasets that had no config at all.

### Documentation

- **Continuity record for the open backlog**, so the work resumes from the repository rather than
  from a conversation: `BACKLOG.md` carries full execution detail for `B-7` (probe inventory with
  HF ids, licenses and sizes plus the `benchmarks/probes.py` loader design), `B-5` (generator
  refactor with a byte-identical guarantee for existing configs, the five-domain table,
  English-first scope, worst-domain gates and the 8–12 h training estimate) and `B-10` (decision:
  stay strict, no inner-shape fallback, so the compliance metric keeps meaning something).
  `STATE.md` opens **OD-6 / OD-7 / OD-8** for the three scope questions still unanswered.
  `benchmarks/README.md` warns that the peer scorer's default `--option-batch 16` OOMs a 12 GB
  card (use `4`) and records the exact eight commands behind the published tables. Roadmap,
  overview, testing and training pages now read *partially delivered* where a public probe exists.
- Corrected the per-language ECE claim (**2/6** languages meet ECE ≤ 0.05, not 5/6), the
  requirement counts (60 functional / 43 non-functional / 104 traced), stale `v0.2.0` status
  headers, references to files and `(planned)` markers that no longer exist, and the documented
  default backend (`llm`). Added CLI, MCP, LangChain and Docker guides.

## [0.3.0] - 2026-09-26

### Added

- **Input-noise robustness (B-4):** `training/generate_data.py` gains a seeded, opt-in
  `noise_rate` (char swap/delete, accent strip, casing flip, terminal-punctuation drop) applied to
  the `state` only; `training/evaluate.py --noise-rate` reports a clean vs noisy split
  (`report["noisy"]`); `benchmarks/report.py` renders it. `TRAIN-08`.
- **Confidence thresholding / System-2 handoff (B-3):** `normalized_entropy` and `margin` in
  `calibration.py`; `handoff.py` with `assess`/`assess_response`; CLI `--threshold` appends a
  sibling `handoff` object. `CAL-06`.

### Fixed

- `training/evaluate.py` now honors `JEBA_ADAPTERS` (and other env settings) when building the
  encoder backend, so a local adapter can be evaluated.

### Changed

- **Multilingual LoRA rank (NFR-C06):** raised `lora_rank` 16 → 64 / `lora_alpha` 128 in
  `training/configs/finetune_multi.json`. Multilingual overall accuracy 0.702 → 0.853 and `es` ECE
  0.170 → 0.038 (`es` accuracy 0.472 → 0.956); 2/6 languages now meet ECE ≤ 0.05 (`es` 0.038,
  `pt` 0.024 — `de` 0.063, `fr` 0.051, `it` 0.059 and `nl` 0.104 remain). See
  `.specs/features/lora-rank-experiment/`.
- Retrained end-to-end on the RTX 3060; refreshed `benchmarks/report.md` (English 0.763 / ECE
  0.061; multilingual r=64 0.853 / ECE 0.038) and the model card. The released adapters are already
  robust to the injected noise (EN 0.763→0.760, multi 0.853→0.847); a noise-augmented adapter
  (r=16) scored 0.719/0.719 but calibrated worse and is not released.
- Republished the multilingual adapter at LoRA r=64 to the Hugging Face Hub
  (`munod/jeba-multi`, commit `599df58`). The English adapter (`munod/jeba-en`) is unchanged.

## [0.2.0] - 2026-09-24

### Added

- **Multilingual quality (B-1):** fully localized synthetic data (`training/data/lexicon.json`),
  per-`(primitive, language)` temperature fitting with a widened grid, runtime `kind:lang` →
  `kind` → global temperature selection with heuristic language detection, and per-language
  accuracy/ECE reporting (worst-language gate). Retrained on the RTX 3060: multilingual `choice`
  0.40 → 0.734 and overall 0.609 → 0.711; English overall 0.721 → 0.781.
- **Fast path (B-2):** `JEBA_FAST` wires the encoder to a per-shape CUDA-graph forward with
  bf16-resident weights and graceful fallback; `benchmarks/fast_path.py` measures 2.68× p50
  (10.27 → 3.83 ms) with 0 top-label flips, meeting NFR-P01/NFR-P07.

### Changed

- `maybe_accelerate` now takes an injected accelerator builder; TileLang fused kernels remain
  optional.
- `training/generate_data.py` seeds an RNG per record (independent draws) and lowers the
  hard-negative rate to 1/6 (see L-003).

## [0.1.1] - 2026-09-24

### Added

- Dedicated low-rank `choice` head, trained alongside the LoRA adapter with a near-identity
  initialization (so training starts at the cosine baseline). English `choice` rises from
  ~0.25 (chance) to 0.78 and multilingual to 0.40 (L-002).
- Rich, localized option descriptions — including a genuinely learnable `other` class — and
  distractor clauses in the synthetic data generator.
- `choice_head.json` is trained, packaged, and loaded by the runtime (local directory or Hub).

### Changed

- Refreshed full-scale numbers: English overall 0.721 / ECE 0.052; multilingual 0.609 / ECE 0.086
  (see [`benchmarks/report.md`](benchmarks/report.md)).
- README rewritten for the released state; markdown files excluded from ruff.

## [0.1.0] - 2026-09-24

First tagged release: a local-first, multilingual System One decision engine that speaks the
TypeSafe Jev `/v1/systemone` wire protocol as a drop-in.

### Added

- **M0 — Bootstrap:** uv project on Python 3.12, ruff/pyright/pytest gates, CI, Apache-2.0.
- **M1 — Wire Contract:** `choice` / `score` / `noul` primitives, the Jev `/v1/systemone`
  envelope, error shapes, and a golden contract suite.
- **M2 — LLM Backend + Serve:** OpenAI-compatible structured-output backend with an injectable
  transport, FastAPI server (`/v1/systemone`, `/predict`, `/predict/batch`, `/health`), Python
  SDK with 429/529 backoff, CLI, presets, and `decide()` from JSON Schema.
- **M3 — Local Encoder:** single-pass encoder backend, script/language router with checkpoint
  lifecycle, confidence/calibration, batching, and prediction/lifecycle hooks. Offline.
- **M4 — Training & Calibration:** deterministic data generation, batched LoRA/QLoRA fine-tuning
  with an RLCD proper-scoring objective, temperature/ECE fitting, an evaluation harness, and
  committed reproducible configs.
- **M5 — Ecosystem & Acceleration:** ONNX backend, optional fast path with graceful fallback,
  MCP stdio server, LangChain adapter, Docker/compose, and a no-op opt-out telemetry guard.
- **M6 — Proof & Release:** reproducible benchmark report, mkdocs documentation site, model card,
  and release process.

### Models

- LoRA adapters published on the Hugging Face Hub:
  [`munod/jeba-en`](https://huggingface.co/munod/jeba-en) (ModernBERT-large) and
  [`munod/jeba-multi`](https://huggingface.co/munod/jeba-multi) (mmBERT-base). The runtime loads
  them by default and caches weights locally (ADR-0010).

### Notes

- Measured on a single RTX 3060 12GB: English 0.613 overall accuracy / 0.059 ECE; multilingual
  0.493 / 0.034 (see [`benchmarks/report.md`](benchmarks/report.md)). `choice` remains near
  chance and needs a dedicated head (L-002).
- `JEBA_BACKEND=llm` is the default backend; the local encoder requires the `train` extra.

## [0.0.1] - 2026-09-24

Specification baseline (documentation only, no code).

[Unreleased]: https://github.com/munod/tachyone/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/munod/tachyone/releases/tag/v0.6.0
[0.5.0]: https://github.com/munod/tachyone/releases/tag/v0.5.0
[0.4.0]: https://github.com/munod/tachyone/releases/tag/v0.4.0
[0.3.0]: https://github.com/munod/tachyone/releases/tag/v0.3.0
[0.2.0]: https://github.com/munod/tachyone/releases/tag/v0.2.0
[0.1.1]: https://github.com/munod/tachyone/releases/tag/v0.1.1
[0.1.0]: https://github.com/munod/tachyone/releases/tag/v0.1.0
[0.0.1]: https://github.com/munod/tachyone/releases/tag/v0.0.1
