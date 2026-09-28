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
    evaluate,
    load_examples,
    noul_label_audit,
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

    def predictor(state: object, question: Question) -> Answer:
        del state
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
