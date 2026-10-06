"""Tests for the evaluation harness (M4-T5, TRAIN-05)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tachyone.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from training.evaluate import (
    EvalExample,
    add_state_noise,
    choice_gate_report,
    evaluate,
    input_signature,
    load_examples,
    noul_label_audit,
    record_hint,
    record_to_example,
    save_report,
)
from training.generate_data import DataConfig, generate


def _examples(tmp_path: Path, per_type: int = 8) -> list[EvalExample]:
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=11, per_type=per_type, languages=("en", "pt")), path)
    return load_examples(path)


def _never_called(state: object, question: Question) -> Answer:
    raise AssertionError("predictor should not be called")


def _perfect(examples: list[EvalExample]):
    """Predict the known target with full confidence, keyed by question identity."""
    by_question = {id(example.question): example.target for example in examples}

    def predictor(state: object, question: Question, *, head_hint: str | None = None) -> Answer:
        del state, head_hint  # a perfect predictor needs no head, hint or not (B6)
        target = by_question[id(question)]
        if isinstance(question, NoulQuestion):
            return NoulAnswer(noul=1.0 if target == 1 else 0.0)
        keys = list(question.criteria)
        if isinstance(question, ChoiceQuestion):
            return ChoiceAnswer(
                choice=str(target),
                probabilities={key: (1.0 if key == target else 0.0) for key in keys},
                confidence=1.0,
            )
        assert isinstance(question, ScoreQuestion)
        return ScoreAnswer(
            score=float(target),
            legend=dict(enumerate(keys)),
            probabilities={index: (1.0 if index == target else 0.0) for index in range(len(keys))},
            confidence=1.0,
        )

    return predictor


def _biased(state: object, question: Question) -> Answer:
    del state
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=0.5)
    keys = list(question.criteria)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=str(keys[0]),
            probabilities={key: (1.0 if key == keys[0] else 0.0) for key in keys},
            confidence=1.0,
        )
    assert isinstance(question, ScoreQuestion)
    return ScoreAnswer(
        score=0.0,
        legend=dict(enumerate(keys)),
        probabilities={index: (1.0 if index == 0 else 0.0) for index in range(len(keys))},
        confidence=1.0,
    )


def test_record_to_example_types() -> None:
    noul = record_to_example(
        {
            "id": "n",
            "type": "noul",
            "state": "x",
            "instructions": "q?",
            "criteria": {"true": "y", "false": "n"},
            "target": 1,
            "lang": "en",
        }
    )
    assert isinstance(noul.question, NoulQuestion)
    choice = record_to_example(
        {
            "id": "c",
            "type": "choice",
            "state": "x",
            "instructions": "q?",
            "criteria": {"a": None, "b": None},
            "target": "a",
            "lang": "en",
        }
    )
    assert isinstance(choice.question, ChoiceQuestion)
    score = record_to_example(
        {
            "id": "s",
            "type": "score",
            "state": "x",
            "instructions": "q?",
            "criteria": ["low", "high"],
            "target": 1,
            "lang": "en",
        }
    )
    assert isinstance(score.question, ScoreQuestion)


def test_evaluate_with_perfect_predictor(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _perfect(examples))
    assert report["overall"]["accuracy"] == pytest.approx(1.0)
    assert report["overall"]["ece"] == pytest.approx(0.0, abs=1e-9)
    assert set(report["per_primitive"]) == {"noul", "choice", "score"}
    assert report["overall"]["latency_ms"]["p50"] <= report["overall"]["latency_ms"]["p95"]


def test_evaluate_reports_per_language(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _biased)
    assert set(report["per_language"]) <= {"en", "pt"}
    assert report["overall"]["accuracy"] < 1.0


def test_save_report(tmp_path: Path) -> None:
    examples = _examples(tmp_path, per_type=3)
    out = tmp_path / "report.json"
    save_report(evaluate(examples, _biased), out)
    assert '"overall"' in out.read_text(encoding="utf-8")


def test_evaluate_empty() -> None:
    report = evaluate([], _never_called)
    assert report["overall"]["n"] == 0


def test_evaluate_noise_rate_adds_noisy_view(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _biased, noise_rate=0.3)
    assert report["noise_rate"] == 0.3
    assert "noisy" in report
    assert set(report["noisy"]) == {
        "bins",
        "overall",
        "per_primitive",
        "per_language",
        "per_domain",
        "noul_per_language",
    }
    assert report["noisy"]["overall"]["n"] == report["overall"]["n"]


def test_evaluate_default_has_no_noisy_view(tmp_path: Path) -> None:
    report = evaluate(_examples(tmp_path), _biased)
    assert "noisy" not in report


def test_add_state_noise_is_deterministic_and_scoped() -> None:
    examples = [
        EvalExample(
            id="a",
            type="noul",
            state="café com açúcar",
            question=NoulQuestion(instructions="q?"),
            target=1,
            lang="pt",
        )
    ]
    one = add_state_noise(examples, rate=1.0, seed=7)
    two = add_state_noise(examples, rate=1.0, seed=7)
    assert [e.state for e in one] == [e.state for e in two]


def test_add_state_noise_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        add_state_noise([], rate=1.5)


def test_backend_predictor_honors_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the predictor used to pass a literal env dict to Config.from_env, dropping
    # TACHYONE_ADAPTERS. It must merge the process env so a local adapter can be evaluated.
    from tachyone.config import Config

    monkeypatch.setenv("TACHYONE_ADAPTERS", "tachyone-multi=checkpoints/multi_noisy")
    merged = {
        **__import__("os").environ,
        "TACHYONE_BACKEND": "encoder",
        "TACHYONE_MODELS_DIR": ".cache/tachyone/models",
    }
    assert Config.from_env(merged).adapters == {"tachyone-multi": "checkpoints/multi_noisy"}


def test_evaluate_reports_per_domain(tmp_path: Path) -> None:
    """B-5 gates on the worst domain, so the report must break down by domain."""
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=13, per_type=5, languages=("en",), domains=("support", "voice")), path)
    examples = load_examples(path)
    report = evaluate(examples, _perfect(examples))
    assert set(report["per_domain"]) == {"support", "voice"}
    assert report["per_domain"]["voice"]["n"] == 15  # 5 records per primitive
    assert report["per_domain"]["support"]["accuracy"] == 1.0


def test_records_without_a_domain_are_attributed_to_support(tmp_path: Path) -> None:
    """Pre-B-5 datasets carry no ``domain`` field; they are the support domain by definition."""
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=13, per_type=3, languages=("en",)), path)
    examples = load_examples(path)
    assert {example.domain for example in examples} == {"support"}
    report = evaluate(examples, _perfect(examples))
    assert set(report["per_domain"]) == {"support"}
    assert report["per_domain"]["support"] == report["overall"]


def test_evaluate_reports_domain_language_cells_for_multi_domain_sets(tmp_path: Path) -> None:
    """B5B-03: the worst-cell gate needs domain x language in one cell (B-5b)."""
    path = tmp_path / "eval.jsonl"
    generate(
        DataConfig(seed=13, per_type=4, languages=("en", "pt"), domains=("support", "voice")),
        path,
    )
    examples = load_examples(path)
    report = evaluate(examples, _perfect(examples))
    cross = report["per_domain_language"]
    assert set(cross) == {
        f"{domain}/{lang}" for domain in ("support", "voice") for lang in ("en", "pt")
    }
    assert all(cell["n"] == 6 for cell in cross.values())  # 12 records/domain ÷ 2 languages
    assert all(cell["accuracy"] == 1.0 for cell in cross.values())
    # Sorted, deterministic key order so a re-render diffs cleanly.
    assert list(cross) == sorted(cross)


def test_single_domain_reports_carry_no_domain_language_cells(tmp_path: Path) -> None:
    """A legacy (one-domain) report stays byte-identical — the cross cell never appears."""
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=13, per_type=3, languages=("en", "pt")), path)
    examples = load_examples(path)
    report = evaluate(examples, _perfect(examples))
    assert "per_domain_language" not in report
    assert set(report["per_domain"]) == {"support"}


# ------------------------------------------------------------------ B-11: the `noul` label audit


def test_noul_labels_audit_of_generated_data_is_clean(tmp_path: Path) -> None:
    """B-11 acceptance: no request-toned record carries 0, no neutral-toned one carries 1.

    Checked over two domains and four languages — the per-language spread (`de`/`nl`) is what
    L-008 traced to label noise rather than model quality.
    """
    path = tmp_path / "eval.jsonl"
    generate(
        DataConfig(
            seed=3,
            per_type=30,
            languages=("en", "pt", "de", "nl"),
            domains=("support", "voice"),
        ),
        path,
    )
    examples = load_examples(path)
    report = evaluate(examples, _perfect(examples))
    audit = report["noul_labels"]
    assert audit["n"] == sum(row["n"] for row in audit["per_language"].values())
    assert audit["unknown"] == 0  # no surface noise in a clean dataset
    assert audit["contradictory"] == 0
    assert audit["contradictory_rate"] == 0.0
    assert 0.4 < audit["positive_rate"] < 0.55  # text-consistent labels are balanced
    # Accuracy for exactly those rows, per language, beside the audit (the acceptance asks for
    # the two to be reported together so label noise cannot read as a capability gap).
    assert set(report["noul_per_language"]) == set(audit["per_language"])
    for lang, row in report["noul_per_language"].items():
        assert row["n"] == audit["per_language"][lang]["n"]
        assert row["accuracy"] == 1.0  # the perfect predictor nails the text-consistent labels


def test_noul_label_audit_catches_a_flipped_target() -> None:
    """The audit must be able to fail: it caught 50% of ``eval_en.jsonl`` before B-11."""
    request = EvalExample(
        id="a",
        type="noul",
        state="Please help me with my invoice.",
        question=NoulQuestion(instructions="q?"),
        target=0,  # contradicts the text
        lang="en",
    )
    neutral = EvalExample(
        id="b",
        type="noul",
        state="Just checking in about the invoice.",
        question=NoulQuestion(instructions="q?"),
        target=1,  # contradicts the text
        lang="en",
    )
    empty = EvalExample(
        id="c",
        type="noul",
        state="",
        question=NoulQuestion(instructions="q?"),
        target=1,  # an empty state reads as no request
        lang="en",
    )
    audit = noul_label_audit([request, neutral, empty])
    assert (audit["request"], audit["neutral"], audit["empty"]) == (1, 1, 1)
    assert audit["contradictory"] == 3
    assert audit["contradictory_rate"] == 1.0
    # The corrected labels pass.
    fixed = [
        replace(request, target=1),
        replace(neutral, target=0),
        replace(empty, target=0),
    ]
    assert noul_label_audit(fixed)["contradictory"] == 0


def test_noul_label_audit_does_not_judge_states_it_cannot_read() -> None:
    """B-4's surface-noise edits are ``unknown``, never counted as contradictions."""
    perturbed = EvalExample(
        id="a",
        type="noul",
        state="Plese hlp me with my invoice",  # not any phrase in the bank
        question=NoulQuestion(instructions="q?"),
        target=0,
        lang="en",
    )
    audit = noul_label_audit([perturbed])
    assert audit["unknown"] == 1
    assert audit["judged"] == 0
    assert audit["contradictory"] == 0


@pytest.mark.parametrize(
    "dataset",
    [
        "data/eval_en.jsonl",
        "data/eval_multi.jsonl",
        "data/eval_en_domains.jsonl",
        "data/eval_en_domains_holdout.jsonl",
        "data/eval_multi_domains_holdout.jsonl",
    ],
)
def test_shipped_eval_sets_carry_no_contradictory_noul_labels(dataset: str) -> None:
    """The B-11 acceptance, on the shipped eval sets themselves (they are gitignored: skipped
    when absent, like the config-vs-dataset hash check)."""
    path = Path(__file__).resolve().parents[1] / dataset
    if not path.exists():
        pytest.skip(f"{dataset} is gitignored and not present")
    audit = noul_label_audit(load_examples(path))
    assert audit["unknown"] == 0, "eval sets are clean; no state should be unreadable"
    assert audit["contradictory"] == 0, audit
    assert 0.40 < audit["positive_rate"] < 0.55, "text-consistent labels are balanced"


# --- B6: the choice gate beside the per-domain numbers it routes (ADR-0016) ---------------


def _choice(
    example_id: str,
    domain: str,
    *,
    target: str = "billing",
    state: str = "my card was charged twice",
) -> EvalExample:
    return EvalExample(
        id=example_id,
        type="choice",
        state=state,
        question=ChoiceQuestion.model_validate(
            {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"},
            }
        ),
        target=target,
        lang="en",
        domain=domain,
    )


def test_choice_gate_report_separates_the_fallback_from_misrouting() -> None:
    """Falling back to shared is today's behaviour; only a wrong domain is harmful."""
    examples = [
        _choice("a", "support"),
        _choice("b", "support"),
        _choice("c", "voice"),
        _choice("d", "voice"),
    ]
    selected = {"a": "support", "b": None, "c": "support", "d": "voice"}
    report = choice_gate_report(examples, lambda example: selected[example.id])

    assert (report["n"], report["strict"], report["fell_to_shared"], report["wrong_domain"]) == (
        4,
        2,
        1,
        1,
    )
    assert report["strict_accuracy"] == pytest.approx(0.5)
    assert report["per_domain"]["support"] == {
        "n": 2,
        "strict": 1,
        "fell_to_shared": 1,
        "wrong_domain": 0,
        "strict_accuracy": pytest.approx(0.5),
    }
    assert report["per_domain"]["voice"]["wrong_domain"] == 1


def test_choice_gate_report_judges_choice_records_only() -> None:
    examples = [
        _choice("a", "support"),
        EvalExample(
            id="n",
            type="noul",
            state="please refund this",
            question=NoulQuestion(instructions="q?"),
            target=1,
            lang="en",
            domain="voice",
        ),
    ]
    report = choice_gate_report(examples, lambda example: example.domain)
    assert report["n"] == 1
    assert report["strict_accuracy"] == 1.0
    assert set(report["per_domain"]) == {"support"}


def test_evaluate_attaches_the_gate_and_names_who_routed(tmp_path: Path) -> None:
    examples = [_choice("a", "support"), _choice("b", "voice")]

    def gate(example: EvalExample) -> str | None:
        return example.domain  # the perfect gate

    with_gate = evaluate(examples, _perfect(examples), choice_gate=gate)
    assert with_gate["choice_gate"]["strict_accuracy"] == 1.0
    assert with_gate["choice_gate"]["answers_routed_by"] == "gate"

    oracle = evaluate(examples, _perfect(examples), choice_gate=gate, head_hint="domain")
    assert oracle["choice_gate"]["answers_routed_by"] == "domain"
    assert oracle["overall"]["accuracy"] == pytest.approx(1.0)

    assert "choice_gate" not in evaluate(examples, _perfect(examples))  # no bank: no row


def test_head_hint_reaches_the_predictor_per_record() -> None:
    """The oracle resolves to each record's own domain; a literal key is passed through."""
    examples = [_choice("a", "support"), _choice("b", "voice")]
    seen: list[str | None] = []

    def predictor(state: object, question: Question, *, head_hint: str | None = None) -> Answer:
        seen.append(head_hint)
        return _perfect(examples)(state, question)

    evaluate(examples, predictor, head_hint="domain")
    assert seen == [example.domain for example in examples]

    seen.clear()
    evaluate(examples, predictor, head_hint="voice")
    assert seen == ["voice", "voice"]

    seen.clear()
    evaluate(examples, predictor)  # no hint: the gate decides, nothing is forced
    assert seen == [None, None]


def test_record_hint_is_one_rule_for_the_oracle() -> None:
    example = _choice("a", "support")
    assert record_hint(example, "domain") == example.domain
    assert record_hint(example, "voice") == "voice"


# --- B7: accuracy split by whether the row's input text occurred in training --------------


def test_input_signature_ignores_the_label_but_not_the_text() -> None:
    """The label is excluded on purpose: same evidence = same fingerprint (B7)."""
    base = _choice("a", "support")
    same_text = replace(base, id="b", target="technical")  # same input, different label
    other_text = _choice("c", "support", state="the parcel never arrived")
    assert input_signature(base) == input_signature(same_text)
    assert input_signature(base) != input_signature(other_text)
    # The domain is metadata, not text: nothing a model sees distinguishes these two.
    assert input_signature(base) == input_signature(_choice("d", "voice"))


def test_evaluate_splits_accuracy_by_seen_input_text() -> None:
    examples = [
        _choice("a", "support"),
        _choice("b", "voice", state="the thermostat never turns on"),
    ]
    seen = {input_signature(examples[0])}
    report = evaluate(examples, _perfect(examples), seen_inputs=seen)

    split = report["text_seen"]
    assert split["seen"]["n"] == 1
    assert split["seen"]["accuracy"] == pytest.approx(1.0)
    assert split["unseen"]["n"] == 1
    assert split["unseen"]["accuracy"] == pytest.approx(1.0)

    # Without a training set to compare against, no row is claimed to be memorized.
    assert "text_seen" not in evaluate(examples, _perfect(examples))
