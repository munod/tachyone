"""Tests for deterministic synthetic data generation (M4-T1, TRAIN-01/TRAIN-07)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from tachyone.primitives import Question
from training.generate_data import (
    _LEVEL_BY_TONE,
    DEFAULT_DOMAIN,
    DEFAULT_LANGUAGES,
    DOMAINS,
    DataConfig,
    DomainData,
    _holdout_strings,
    _line,
    _split_bank,
    domain_signatures,
    generate,
    iter_records,
    main,
)

_QUESTION = TypeAdapter(Question)
_ROOT = Path(__file__).resolve().parents[1]


def _records(count: int = 12) -> list[dict[str, Any]]:
    return list(iter_records(DataConfig(seed=7, per_type=count, languages=("en", "pt"))))


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    config = DataConfig(seed=123, per_type=30)
    generate(config, first)
    generate(config, second)
    assert first.read_bytes() == second.read_bytes()


def test_different_seeds_differ(tmp_path: Path) -> None:
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    generate(DataConfig(seed=1, per_type=20), one)
    generate(DataConfig(seed=2, per_type=20), two)
    assert one.read_bytes() != two.read_bytes()


def test_counts_and_ids_are_unique() -> None:
    records = _records(5)
    assert len(records) == 15
    ids = [record["id"] for record in records]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "noul-000000"


def test_records_validate_as_primitives() -> None:
    for record in _records(6):
        assert record["type"] in {"noul", "choice", "score"}
        _QUESTION.validate_python(
            {
                "type": record["type"],
                "instructions": record["instructions"],
                "criteria": record["criteria"],
            }
        )
        if record["type"] == "noul":
            assert record["target"] in (0, 1)
        elif record["type"] == "choice":
            assert record["target"] in record["criteria"]
        else:
            assert isinstance(record["target"], int)
            assert 0 <= record["target"] < len(record["criteria"])


def test_languages_are_interleaved() -> None:
    records = _records(4)
    langs = {record["lang"] for record in records}
    assert langs <= {"en", "pt"}


def test_choice_targets_cover_all_teams() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=40, languages=("en", "pt"))))
    teams = {r["target"] for r in records if r["type"] == "choice"}
    assert {"billing", "technical", "sales", "other"} <= teams


def test_choice_uses_localized_terms_and_distractors() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=40, languages=("pt",))))
    states = " ".join(r["state"] for r in records if r["type"] == "choice").lower()
    assert any(word in states for word in ("reembolso", "fatura", "cobrança", "pagamento"))
    # Every pt distractor clause contains "também"; ensure hard negatives are present.
    assert "também" in states


def test_choice_criteria_have_rich_descriptions_including_other() -> None:
    records = list(iter_records(DataConfig(seed=3, per_type=8, languages=("en",))))
    choice = next(r for r in records if r["type"] == "choice")
    criteria = choice["criteria"]
    assert all(isinstance(value, str) and value for value in criteria.values())
    # "other" must be a semantically rich, learnable option (not a bare label).
    assert len(criteria["other"]) > 60
    assert "outside" in criteria["other"]


def test_instructions_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    choice = next(r for r in records if r["type"] == "choice")
    score = next(r for r in records if r["type"] == "score")
    noul = next(r for r in records if r["type"] == "noul")
    assert "equipe" in choice["instructions"].lower()
    assert "urgência" in score["instructions"].lower()
    assert "solicita" in noul["instructions"].lower()


def test_score_levels_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    score = next(r for r in records if r["type"] == "score")
    assert score["criteria"] == ["nenhuma", "baixa", "média", "alta"]


def test_noul_entities_are_localized() -> None:
    records = list(iter_records(DataConfig(seed=5, per_type=12, languages=("pt",))))
    noul = next(r for r in records if r["type"] == "noul")
    assert isinstance(noul["criteria"], dict)
    # The true-criterion uses the localized template, not the English one.
    assert noul["criteria"]["true"].startswith("solicita um ")
    assert "asks for" not in noul["criteria"]["true"]


def _state_tone(domain: DomainData, lang: str, state: str, tones: Sequence[str]) -> set[str]:
    """Phrase banks the emitted ``state`` was built from, recovered from the text alone.

    ``_boundary_state`` keeps the phrase, repeats it (long input) or empties it, so the bank that
    produced the state is a prefix question — no RNG replay, which is the point of B-11/B-12: the
    label must be checkable against the text a reader would actually see.
    """
    if not state:
        return set()
    phrases = domain.phrases(lang)
    entities = domain.entities(lang)
    return {
        tone
        for tone in tones
        if any(
            state.startswith(phrase.format(entity=entity))
            for phrase in phrases[tone]
            for entity in entities
        )
    }


def test_noul_target_follows_the_text_not_the_loop_index() -> None:
    """B-11 acceptance: request-toned -> 1, neutral-toned -> 0, empty state -> 0.

    Checked across every committed domain and language, which is where the old index-parity rule
    showed up as 40-55% contradictory labels in `de`/`es`/`nl` (L-008).
    """
    langs = ("en", "pt", "es", "fr", "de", "it", "nl")
    domains = tuple(sorted(DOMAINS))
    records = [
        record
        for record in iter_records(
            DataConfig(seed=9, per_type=42, languages=langs, domains=domains)
        )
        if record["type"] == "noul"
    ]
    assert records
    positives = 0
    for record in records:
        domain = DomainData.load(record["domain"])
        if not record["state"]:
            assert record["target"] == 0, record["id"]  # "" reads as no request
            continue
        tones = _state_tone(domain, record["lang"], record["state"], ("request", "neutral"))
        assert len(tones) == 1, (record["id"], tones)  # the text is never ambiguous
        expected = 1 if tones == {"request"} else 0
        assert record["target"] == expected, (record["id"], record["state"], record["target"])
        positives += record["target"]
    # Text-consistent labels are balanced (~47%: the empty-boundary states are all negative),
    # where the index-parity rule produced ~25% positives.
    assert 0.40 < positives / len(records) < 0.55


def test_score_level_comes_from_the_text_not_the_index() -> None:
    """B-12 acceptance: the level is the tone's; an empty state takes the middle level.

    The old ``index % 13`` "near-tie" downgrade contradicted 7.8% of rows and capped `score` at
    ~0.888 whatever the model learned; the empty boundary state inherited a tone it cannot read.
    Both are gone — the index draws remain, they simply no longer move the label.
    """
    langs = ("en", "pt", "es", "fr", "de", "it", "nl")
    domains = tuple(sorted(DOMAINS))
    records = [
        record
        for record in iter_records(
            DataConfig(seed=11, per_type=42, languages=langs, domains=domains)
        )
        if record["type"] == "score"
    ]
    assert records
    tone_banks = ("calm", "neutral", "request", "urgent")
    index_draws = 0
    for record in records:
        domain = DomainData.load(record["domain"])
        if not record["state"]:
            assert record["target"] == 1, record["id"]  # no signal -> the middle level
            continue
        banks = _state_tone(domain, record["lang"], record["state"], tone_banks)
        assert len(banks) == 1, (record["id"], banks)  # the text names exactly one level
        tone = banks.pop()
        assert record["target"] == _LEVEL_BY_TONE[tone], (record["id"], tone, record["target"])
        if int(record["id"].split("-")[1]) % 13 == 0:
            index_draws += 1
    assert index_draws > 0  # the near-tie draws still happen; they just carry the tone's level


def test_choice_empty_states_answer_to_the_catch_all() -> None:
    """B-12 acceptance: an empty `choice` state never carries an index-derived option."""
    langs = ("en", "pt", "es", "fr", "de", "it", "nl")
    domains = tuple(sorted(DOMAINS))
    for name in domains:
        assert "other" in DomainData.load(name).options, name  # every domain ships the catch-all
    empties = 0
    for record in iter_records(DataConfig(seed=11, per_type=42, languages=langs, domains=domains)):
        if record["type"] != "choice" or record["state"]:
            continue
        empties += 1
        assert record["target"] == "other", (record["id"], record["target"])
    assert empties > 0  # the empty boundary case is still exercised


def test_every_language_gets_equal_support() -> None:
    langs = ("en", "pt", "es", "fr", "de", "it", "nl")
    records = list(iter_records(DataConfig(seed=5, per_type=len(langs) * 4, languages=langs)))
    for kind in ("noul", "choice", "score"):
        counts: dict[str, int] = {}
        for record in records:
            if record["type"] == kind:
                counts[record["lang"]] = counts.get(record["lang"], 0) + 1
        assert set(counts) == set(langs)
        assert len(set(counts.values())) == 1


def test_every_language_covers_every_option() -> None:
    """Every committed recipe must emit **every** option of every domain as a ``choice`` target
    in **every** ``(domain, language)`` cell (data audit 2026-10-05, Finding 1).

    The old ``options[index % len(options)]`` stride was aliased with
    ``languages[index % len(languages)]`` by ``gcd``: with six languages and four options each
    multilingual cell trained only a subset of the options as labels — ``de``/``es``/``nl`` never
    saw ``billing``/``sales`` in ``support`` — and the eval recipes carried the same defect, so
    the eval could not see it (0 of 2,500 eval ``choice`` rows named an untrained label).
    ``other`` counts as an option like any other.
    """
    from training.config import load_data_config

    config_paths = sorted((_ROOT / "training" / "configs").glob("data*.json"))
    assert config_paths, "committed data configs exist to audit"
    for config_path in config_paths:
        cells: dict[tuple[str, str], set[str]] = {}
        for record in iter_records(load_data_config(config_path)):
            if record["type"] == "choice":
                key = (record.get("domain", DEFAULT_DOMAIN), record["lang"])
                cells.setdefault(key, set()).add(record["target"])
        for (domain, lang), targets in cells.items():
            missing = set(DOMAINS[domain]["options"]) - targets
            assert not missing, (
                f"{config_path.name}: ({domain}, {lang}) never trains the label(s) {sorted(missing)}"
            )


def test_hard_negative_distractors_reach_every_language() -> None:
    """The hard-negative cycle runs over the record's block within its own language, so every
    language sees the distractor clause at the same ~1/``_HARD_NEGATIVE_RATE`` rate (data audit
    2026-10-05, the stride sibling of Finding 1).

    On the old ``index % _HARD_NEGATIVE_RATE`` with six languages the two strides collided
    exactly: every distractor row landed in the first language alone (``pt`` 94.7%, the other
    five 0%), so five of six languages never trained distractor rejection — and 94.7% of ``pt``
    rows carried the ambiguous clause the generator keeps rare on purpose.
    """
    langs = ("pt", "es", "fr", "de", "it", "nl")
    domain = DomainData.load(DEFAULT_DOMAIN)
    clauses = {
        lang: [
            phrase.format(distractor=term)
            for phrase in domain.phrases(lang)["distractor"]
            for option in domain.options
            for term in domain.option_terms(lang, option)
        ]
        for lang in langs
    }
    seen = dict.fromkeys(langs, 0)
    totals = dict.fromkeys(langs, 0)
    records = iter_records(DataConfig(seed=5, per_type=len(langs) * 60, languages=langs))
    for record in records:
        if record["type"] != "choice":
            continue
        totals[record["lang"]] += 1
        if any(clause in record["state"] for clause in clauses[record["lang"]]):
            seen[record["lang"]] += 1
    for lang in langs:
        assert totals[lang] == 60
        # 10 of 60 blocks are flagged; at most one is emptied by the boundary state.
        assert seen[lang] >= 7, f"{lang} sees only {seen[lang]} hard negatives"
    assert max(seen.values()) - min(seen.values()) <= 3, seen


# --- Fase 1: leave-one-template-out eval (data audit 2026-10-05, Finding 3) --------------

#: holdout eval config → the training config whose ``train`` split it must not overlap.
_SPLIT_PAIRS = [
    ("data_eval_en_domains_holdout.json", "data_en_domains.json"),
    ("data_eval_multi_domains_holdout.json", "data_multi_domains.json"),
]

#: (domain, lang) → entities + option terms, longest first, for template masking.
_TERM_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}


def _template(state: str, domain: str, lang: str) -> str:
    """Mask a state down to its phrasing: entities and option terms become ``{X}`` and a
    boundary long state (the base repeated 40x) collapses to one copy of the base."""
    if not state:
        return ""
    key = (domain, lang)
    terms = _TERM_CACHE.get(key)
    if terms is None:
        spec = DOMAINS[domain]["languages"].get(lang, DOMAINS[domain]["languages"]["en"])
        collected = set(spec["entities"])
        for table in spec["option_terms"].values():
            collected.update(table)
        terms = tuple(sorted(collected, key=len, reverse=True))
        _TERM_CACHE[key] = terms
    masked = state
    for term in terms:
        masked = masked.replace(term, "{X}")
    words = masked.split()
    for unit in range(1, len(words) // 2 + 1):
        head = words[:unit]
        if len(words) % unit == 0 and words == head * (len(words) // unit):
            return " ".join(head)
    return masked


def test_template_split_slices_are_disjoint() -> None:
    """``train`` and ``holdout`` never emit the same sentence.

    The hold-out of a bank is its last phrase, and ``train`` additionally drops any sentence
    that is a hold-out in **another** bank of the same language — identical phrasing is
    authored across domains (``Sem pressa com {entity}.`` sits in three pt banks), and without
    the drop it reaches training through a foreign non-hold-out slot while its twin evaluates
    as unseen (measured: 21 leaking rows, all of them that case).
    """
    for domain, spec in DOMAINS.items():
        for lang, bank in spec["languages"].items():
            held = _holdout_strings(lang)
            for tone, phrases in bank["phrases"].items():
                assert len(phrases) >= 2, (domain, lang, tone)
                train = _split_bank(phrases, "train", lang)
                holdout = _split_bank(phrases, "holdout", lang)
                assert list(holdout) == [phrases[-1]]
                assert set(train).isdisjoint(holdout), (domain, lang, tone)
                dropped = set(phrases) - set(train)
                assert dropped == {phrase for phrase in phrases if phrase in held}, (
                    domain,
                    lang,
                    tone,
                )


def test_holdout_eval_shares_almost_no_template_with_training() -> None:
    """Fase 1 gate: at most 5% of non-empty holdout eval rows may carry a ``(kind, template)``
    that training emits — the seed-drawn evals sit at ~99% (89.5%/61.0% identical rows), which
    is exactly why no gain below the ceiling was measurable on them.

    Training is generated from the same configs in ``train`` mode — the split the training
    configs adopt when the data is next recomposed. Empty boundary states (5.3% of rows) share
    one template by construction and are excluded; every other row must be phrasing the split
    never emits. Exact ``(kind, state)`` rows must not coincide at all.
    """
    from training.config import load_data_config

    for eval_name, train_name in _SPLIT_PAIRS:
        train_records = [
            record
            for record in iter_records(
                replace(
                    load_data_config(_ROOT / "training" / "configs" / train_name),
                    template_split="train",
                )
            )
            if record["state"]
        ]
        templates_by_lang: dict[str, set[tuple[str, str]]] = {}
        states: set[tuple[str, str]] = set()
        for record in train_records:
            domain = record.get("domain", DEFAULT_DOMAIN)
            templates_by_lang.setdefault(record["lang"], set()).add(
                (record["type"], _template(record["state"], domain, record["lang"]))
            )
            states.add((record["type"], record["state"]))

        eval_records = list(
            iter_records(load_data_config(_ROOT / "training" / "configs" / eval_name))
        )
        nonempty = [record for record in eval_records if record["state"]]
        assert nonempty, eval_name
        per_lang: dict[str, list[int]] = {}
        exact = 0
        for record in nonempty:
            domain = record.get("domain", DEFAULT_DOMAIN)
            bucket = per_lang.setdefault(record["lang"], [0, 0])
            bucket[1] += 1
            key = (record["type"], _template(record["state"], domain, record["lang"]))
            if key in templates_by_lang[record["lang"]]:
                bucket[0] += 1
            if (record["type"], record["state"]) in states:
                exact += 1
        for lang, (hit, total) in sorted(per_lang.items()):
            assert hit / total <= 0.05, f"{eval_name}: {lang} leaks {100 * hit / total:.1f}%"
        hit = sum(count for count, _ in per_lang.values())
        total = sum(count for _, count in per_lang.values())
        assert hit / total <= 0.05, f"{eval_name}: {100 * hit / total:.2f}% template overlap"
        assert exact == 0, f"{eval_name}: {exact} non-empty rows are byte-identical to training"


def test_boundary_cases_present() -> None:
    records = _records(40)
    states = [record["state"] for record in records if record["type"] == "noul"]
    assert "" in states  # empty input injected at specific indices
    assert any(len(state) > 500 for state in states)  # very long input


def test_streaming_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "data.jsonl"
    written = generate(DataConfig(seed=3, per_type=5), out)
    assert written == 15
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15
    assert json.loads(lines[0])["id"] == "noul-000000"


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.jsonl"
    code = main(["--seed", "9", "--per-type", "3", "--out", str(out)])
    assert code == 0
    assert out.exists()
    assert "wrote 9 records" in capsys.readouterr().out


def _noisy_records(rate: float, count: int = 40) -> list[dict[str, Any]]:
    return list(
        iter_records(DataConfig(seed=11, per_type=count, languages=("en", "pt"), noise_rate=rate))
    )


def test_noise_disabled_is_byte_identical_to_clean(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    noisy = tmp_path / "noisy.jsonl"
    generate(DataConfig(seed=5, per_type=20, noise_rate=0.0), clean)
    generate(DataConfig(seed=5, per_type=20), noisy)
    assert clean.read_bytes() == noisy.read_bytes()


def test_noise_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    config = DataConfig(seed=123, per_type=30, noise_rate=0.5)
    generate(config, first)
    generate(config, second)
    assert first.read_bytes() == second.read_bytes()


def test_noise_rate_zero_perturbs_nothing() -> None:
    explicit = _noisy_records(0.0)
    default = list(iter_records(DataConfig(seed=11, per_type=40, languages=("en", "pt"))))
    assert [r["state"] for r in explicit] == [r["state"] for r in default]


def test_noise_rate_one_perturbs_most_records() -> None:
    clean = {r["id"]: r["state"] for r in _noisy_records(0.0)}
    noisy = {r["id"]: r["state"] for r in _noisy_records(1.0)}
    changed = [rid for rid in clean if clean[rid] != noisy[rid]]
    # Empty states and edit draws that happen to be no-ops (e.g. accent strip on ASCII) pass
    # through unchanged, so require a clear majority rather than every record.
    assert len(changed) > len(clean) // 2


def test_noise_changes_only_the_state() -> None:
    clean = {r["id"]: r for r in _noisy_records(0.0)}
    noisy = {r["id"]: r for r in _noisy_records(1.0)}
    for rid, record in noisy.items():
        for key in ("type", "instructions", "criteria", "target", "lang"):
            assert record[key] == clean[rid][key], key


def test_noise_preserves_empty_and_long_boundary_states() -> None:
    # indices 0,19,38,... are empty; 23,46,... are long (see _boundary_state).
    records = _noisy_records(1.0, count=60)
    for record in records:
        assert isinstance(record["state"], str)  # never raises, always a string


def test_noise_differs_across_seeds() -> None:
    one = {
        r["id"]: r["state"] for r in iter_records(DataConfig(seed=1, per_type=30, noise_rate=0.6))
    }
    two = {
        r["id"]: r["state"] for r in iter_records(DataConfig(seed=2, per_type=30, noise_rate=0.6))
    }
    assert one != two


def test_noise_cli_rejects_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--per-type", "3", "--noise-rate", "1.5", "--out", str(tmp_path / "x.jsonl")])


def test_noise_cli_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "noisy.jsonl"
    code = main(["--seed", "9", "--per-type", "3", "--noise-rate", "0.5", "--out", str(out)])
    assert code == 0
    assert "wrote 9 records" in capsys.readouterr().out


# ------------------------------------------------------------------ B-5: domains
#: sha256[:16] of ``iter_records`` for these configs. Captured **before** the B-5 refactor that
#: moved every byte of content into ``training/data/domains/`` (which changed no byte, the hard
#: guarantee from BACKLOG B-5: L-003/B-4 already cost days to a determinism assumption that did
#: not hold), and **recaptured after B-11** (only ``noul.target`` bytes moved) and again **after
#: B-12** (``score`` loses the ``index % 13`` near-tie, ``choice``/``score`` empty states take
#: their defaults, and ``ecommerce`` gains the ``other`` catch-all, which re-cycles its options),
#: again **after the stride decoupling** (data audit 2026-10-05, Finding 1 and its distractor
#: sibling — ``en`` was byte-identical there, one language so ``block == index``), and once more
#: **after the content expansion** (Fase 2a: entities 20 → 60 and phrase banks → 11/9/9/9/8 per
#: language in all five domains — domain content feeds every record, so all three move).
_GOLDEN_SINGLE_DOMAIN = {
    "en": ("43dc51504a396239", DataConfig(seed=42, per_type=50, languages=("en",))),
    "en_pt": ("7bc8e173f745b2f4", DataConfig(seed=42, per_type=50, languages=("en", "pt"))),
    "noisy": (
        "4a8d0795dcfbd26f",
        DataConfig(seed=7, per_type=30, languages=("en", "pt", "de"), noise_rate=0.15),
    ),
}

#: Committed data configs and the dataset each one is supposed to reproduce (hash-checked in
#: CI-agnostic form: skipped when ``data/`` is absent, since it is gitignored).
_SHIPPED = [
    ("data_en.json", "data/train_en.jsonl"),
    ("data_eval_en.json", "data/eval_en.jsonl"),
    ("data_multi.json", "data/train_multi.jsonl"),
    ("data_noisy.json", "data/train_multi_noisy.jsonl"),
    ("data_eval_multi.json", "data/eval_multi.jsonl"),
    ("data_en_domains.json", "data/train_en_domains.jsonl"),
    ("data_eval_en_domains.json", "data/eval_en_domains.jsonl"),
    ("data_multi_domains.json", "data/train_multi_domains.jsonl"),
    ("data_eval_multi_domains.json", "data/eval_multi_domains.jsonl"),
    ("data_eval_en_domains_holdout.json", "data/eval_en_domains_holdout.jsonl"),
    ("data_eval_multi_domains_holdout.json", "data/eval_multi_domains_holdout.jsonl"),
]

_FIVE = ("support", "ecommerce", "agent_tools", "documents", "voice")


def _digest(config: DataConfig) -> str:
    digest = hashlib.sha256()
    for record in iter_records(config):
        digest.update(_line(record).encode())
    return digest.hexdigest()[:16]


@pytest.mark.parametrize("case", sorted(_GOLDEN_SINGLE_DOMAIN))
def test_single_domain_output_is_byte_identical(case: str) -> None:
    """The config → bytes mapping is pinned: the B-5 refactor moved no byte, B-11 moved only
    ``noul.target``, and B-12 moved ``score``/``choice`` labels (plus `ecommerce`'s re-cycling) —
    see the recapture note above."""
    expected, config = _GOLDEN_SINGLE_DOMAIN[case]
    assert _digest(config) == expected


@pytest.mark.parametrize(("config_name", "dataset"), _SHIPPED)
def test_committed_config_reproduces_its_shipped_dataset(config_name: str, dataset: str) -> None:
    """``training/configs/*.json`` documents the data it generated, seed and all (TRAIN-06)."""
    from training.config import load_data_config

    config_path = _ROOT / "training" / "configs" / config_name
    data_path = _ROOT / dataset
    if not data_path.exists():
        pytest.skip(f"{dataset} is gitignored and not present")
    config = load_data_config(config_path)
    shipped = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]
    assert _digest(config) == shipped, f"{config_name} does not reproduce {dataset}"


def test_multi_domain_run_labels_every_record_and_keeps_counts() -> None:
    per_type = 4
    records = list(
        iter_records(DataConfig(seed=1, per_type=per_type, languages=("en",), domains=_FIVE))
    )
    assert len(records) == len(_FIVE) * 3 * per_type
    assert {record["domain"] for record in records} == set(_FIVE)
    for domain in _FIVE:
        for kind in ("noul", "choice", "score"):
            count = sum(1 for r in records if r["domain"] == domain and r["type"] == kind)
            assert count == per_type


def test_support_records_are_identical_inside_and_outside_a_multi_domain_run() -> None:
    """``support`` keeps the legacy seed, so a support measurement stays comparable (B-5)."""
    solo = list(iter_records(DataConfig(seed=1, per_type=6, languages=("en",))))
    multi = [
        {k: v for k, v in record.items() if k != "domain"}
        for record in iter_records(DataConfig(seed=1, per_type=6, languages=("en",), domains=_FIVE))
        if record["domain"] == DEFAULT_DOMAIN
    ]
    assert multi == solo


_MULTI_LANGS = ("pt", "es", "fr", "de", "it", "nl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multilingual_five_domain_datasets_preserve_the_incumbent() -> None:
    """B5B-02: the no-regression gate needs the incumbent rows byte-identical (modulo ``domain``).

    ``train_multi_domains``' support half must equal ``train_multi`` and its eval half must
    equal ``eval_multi`` once the ``domain`` key is removed — the same invariant the English
    pair guarantees, so the released multilingual adapter's support cell stays comparable on
    identical rows (AD-009 / B-5a pattern).
    """
    paths = [
        _ROOT / "data/train_multi_domains.jsonl",
        _ROOT / "data/train_multi.jsonl",
        _ROOT / "data/eval_multi_domains.jsonl",
        _ROOT / "data/eval_multi.jsonl",
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("gitignored datasets not present")
    train_multi, train_solo, eval_multi, eval_solo = (_load_jsonl(path) for path in paths)

    # Counts: support keeps its 23.4k incumbent volume (7,800 per primitive), four new domains
    # add 7.2k each (52.2k total); eval mirrors the English split at 1,500 per domain (7,500
    # total). Fase 2b sizes: per-domain counts stay divisible by len(languages) x len(options)
    # (24 for four-option domains, 30 for ecommerce) so every cell balances exactly.
    train_domains: dict[str, int] = {}
    for record in train_multi:
        train_domains[record["domain"]] = train_domains.get(record["domain"], 0) + 1
    assert train_domains == {"support": 23400, **dict.fromkeys(_FIVE[1:], 7200)}
    eval_domains: dict[str, int] = {}
    for record in eval_multi:
        eval_domains[record["domain"]] = eval_domains.get(record["domain"], 0) + 1
    assert eval_domains == dict.fromkeys(_FIVE, 1500)

    # Every (domain, primitive) slice interleaves all six training languages.
    langs = {record["lang"] for record in train_multi}
    assert langs == set(_MULTI_LANGS)
    for domain in _FIVE:
        for kind in ("noul", "choice", "score"):
            slice_langs = {
                record["lang"]
                for record in train_multi
                if record["domain"] == domain and record["type"] == kind
            }
            assert slice_langs == set(_MULTI_LANGS), (domain, kind)

    # The incumbent's rows are preserved exactly, only the ``domain`` key is added.
    strip = lambda records: [  # noqa: E731 - one-line helper mirroring the en-pair test
        {k: v for k, v in record.items() if k != "domain"} for record in records
    ]
    train_support = strip([r for r in train_multi if r["domain"] == DEFAULT_DOMAIN])
    eval_support = strip([r for r in eval_multi if r["domain"] == DEFAULT_DOMAIN])
    assert train_support == train_solo
    assert eval_support == eval_solo


def test_single_domain_run_writes_no_domain_field() -> None:
    """The default config must stay byte-identical: no new keys (BACKLOG B-5)."""
    records = list(iter_records(DataConfig(seed=4, per_type=3, languages=("en",))))
    assert all("domain" not in record for record in records)
    # ... while a non-default single domain is labelled, so evaluation can attribute it.
    records = list(
        iter_records(DataConfig(seed=4, per_type=3, languages=("en",), domains=("voice",)))
    )
    assert all(record["domain"] == "voice" for record in records)


def test_unknown_domain_is_rejected_with_the_committed_list() -> None:
    with pytest.raises(SystemExit, match="committed domains"):
        list(iter_records(DataConfig(domains=("telepathy",), per_type=1, languages=("en",))))


def test_domain_signatures_carry_labels_terms_and_descriptions() -> None:
    """The ADR-0016 gate matches on exactly the committed vocabulary, nothing learned."""
    signatures = domain_signatures("support", ("en",))
    spec = DOMAINS["support"]
    options = tuple(spec["options"])
    english = spec["languages"]["en"]
    assert set(options) <= set(signatures)  # option labels
    assert "refund" in signatures  # an option term
    assert english["option_descriptions"]["billing"] in signatures  # a description
    # Deterministic and duplicate-free: the shipped asset must be reproducible.
    assert signatures == domain_signatures("support", ("en",))
    assert len(signatures) == len(set(signatures))


def test_domain_signatures_accumulate_languages_and_reject_unknown_domains() -> None:
    english = set(domain_signatures("support", ("en",)))
    both = set(domain_signatures("support", ("en", "de")))
    assert english <= both  # adding a language only adds terms
    assert len(both) > len(english)  # support ships a German table (measured, not assumed)
    with pytest.raises(SystemExit, match="committed domains"):
        domain_signatures("telepathy", ("en",))


def test_every_committed_domain_file_is_complete() -> None:
    """A domain needs options, terms, descriptions, entities, phrases, levels and instructions.

    Every committed language in ``DEFAULT_LANGUAGES`` gets the full check — a table that only
    ships English would silently fall back to English text under a non-English ``lang`` tag,
    which is the L-008 label artifact (B-5b).
    """
    assert set(DOMAINS) == set(_FIVE)
    for name, spec in DOMAINS.items():
        options = spec["options"]
        assert len(options) >= 2
        assert len(set(options)) == len(options)
        assert set(spec["languages"]) >= set(DEFAULT_LANGUAGES), name
        for lang in DEFAULT_LANGUAGES:
            table = spec["languages"][lang]
            assert len(table["entities"]) >= 8, (name, lang)
            assert len(table["levels"]) == 4, (name, lang)
            assert set(table["instructions"]) == {"noul", "choice", "score"}, (name, lang)
            assert set(table["noul_criteria"]) == {"true", "false"}, (name, lang)
            assert set(table["phrases"]) == {
                "request",
                "neutral",
                "urgent",
                "calm",
                "distractor",
            }, (
                name,
                lang,
            )
            assert set(table["option_terms"]) == set(options), (name, lang)
            assert set(table["option_descriptions"]) == set(options), (name, lang)
            for option in options:
                assert len(table["option_terms"][option]) >= 4, (name, lang, option)
                assert len(table["option_descriptions"][option]) >= 20, (name, lang, option)
            for tone in ("request", "neutral", "urgent", "calm"):
                assert len(table["phrases"][tone]) >= 3, (name, lang, tone)
            assert all("{entity}" in p for p in table["phrases"]["request"]), (name, lang)
            assert all("{distractor}" in p for p in table["phrases"]["distractor"]), (name, lang)
            assert "{entity}" in table["instructions"]["noul"], (name, lang)


def test_multi_domain_records_validate_as_primitives() -> None:
    for record in iter_records(DataConfig(seed=6, per_type=5, languages=("en",), domains=_FIVE)):
        _QUESTION.validate_python(
            {
                "type": record["type"],
                "instructions": record["instructions"],
                "criteria": record["criteria"],
            }
        )
        if record["type"] == "choice":
            assert record["target"] in record["criteria"]


def test_per_domain_overrides_the_count_for_named_domains() -> None:
    """Keeping the incumbent's volume while new domains are added (B-5, see DataConfig)."""
    config = DataConfig(
        seed=8, per_type=3, languages=("en",), domains=_FIVE, per_domain={"support": 7}
    )
    records = list(iter_records(config))
    for domain in _FIVE:
        expected = 7 if domain == "support" else 3
        count = sum(1 for record in records if record["domain"] == domain)
        assert count == expected * 3, domain


def test_per_domain_cli_parses_and_rejects_malformed_overrides(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    with pytest.raises(SystemExit, match="name=COUNT"):
        main(["--per-domain", "support", "--out", str(out)])
    with pytest.raises(SystemExit, match="name=COUNT"):
        main(["--per-domain", "support=many", "--out", str(out)])
    code = main(
        [
            "--per-type",
            "2",
            "--domains",
            "support,voice",
            "--per-domain",
            "support=5",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in records if r["domain"] == "support") == 15
    assert sum(1 for r in records if r["domain"] == "voice") == 6
