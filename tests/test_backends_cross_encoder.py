"""Tests for the S2-v V6 seam: the cross-encoder as an opt-in scoring override.

The load-bearing properties: the override answers ``choice``/``noul`` from the scorer's own
per-option logits (the P3 knobs bypassed — the teacher's softmax is the calibration),
``score`` never leaves the single-pass path, and with the knob unset the legacy path is
byte-identical (additive extension). GPU loading is exercised by the cycle, not here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from tachyone.backends.cross_encoder import CrossEncoderEnsemble, maybe_cross_encoder
from tachyone.backends.encoder import EncoderModel
from tachyone.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


def _encode(texts: list[str]) -> list[list[float]]:
    return [
        [float(sum(map(ord, text)) % 97) / 97 + index * 0.01 for index in range(8)]
        for text in texts
    ]


class _FakeScorer:
    """Deterministic logits per call; records what the seam hands it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def temperature(self, kind: str, lang: str | None = None) -> float:
        return 1.0

    @property
    def calibration(self):
        return None

    def score(
        self, state_text: str, question_text: str, criterion_texts: Sequence[str]
    ) -> list[float]:
        self.calls.append((state_text, question_text, tuple(criterion_texts)))
        # strongest logit on the criterion whose text holds "yes"/"true"/first
        weights = [float(len(text)) for text in criterion_texts]
        return weights


def test_choice_and_noul_route_to_the_scorer_and_score_does_not() -> None:
    scorer = _FakeScorer()
    model = EncoderModel(_encode, cross_encoder=scorer)
    questions = {
        "route": ChoiceQuestion(
            instructions="Which specialist?",
            criteria={"math_agent": "Numbers.", "coding_agent": "Code."},
        ),
        "ok": NoulQuestion(
            instructions="Is it permitted?", criteria=NoulCriteria(false="No.", true="Yes.")
        ),
        "level": ScoreQuestion(instructions="How urgent?", criteria=["low", "high"]),
    }
    answers = model.answer_state("the request", questions)
    assert isinstance(answers["route"], ChoiceAnswer)
    assert isinstance(answers["ok"], NoulAnswer)
    assert isinstance(answers["level"], ScoreAnswer)
    assert len(scorer.calls) == 2  # score never reaches the scorer
    asked = {call[1] for call in scorer.calls}
    assert asked == {"Which specialist?", "Is it permitted?"}


def test_choice_probabilities_are_the_scorers_softmax() -> None:
    scorer = _FakeScorer()
    model = EncoderModel(_encode, cross_encoder=scorer)
    answer = model.answer_state(
        "s",
        {"route": ChoiceQuestion(instructions="Pick.", criteria={"a": "aaaaaa", "b": "bb"})},
    )["route"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "a"  # the longer description scores higher in the fake
    assert set(answer.probabilities) == {"a", "b"}
    assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-9
    assert answer.confidence == answer.probabilities[answer.choice]


def test_noul_maps_the_true_criterion() -> None:
    scorer = _FakeScorer()
    model = EncoderModel(_encode, cross_encoder=scorer)
    answer = model.answer_state(
        "s",
        {
            "ok": NoulQuestion(
                instructions="Permitted?", criteria=NoulCriteria(false="no", true="yes indeed")
            )
        },
    )["ok"]
    assert isinstance(answer, NoulAnswer)
    # "yes indeed" is the longer text -> the stronger logit -> noul > 0.5
    assert answer.noul > 0.5
    # without criteria the canonical pair is scored and "true" leads (position 0)
    fallback = model.answer_state("s", {"ok": NoulQuestion(instructions="Permitted?")})["ok"]
    assert isinstance(fallback, NoulAnswer)
    assert 0.0 <= fallback.noul <= 1.0


def test_legacy_path_is_byte_identical_without_the_knob() -> None:
    questions = {
        "is_urgent": NoulQuestion(instructions="Does this express urgency?"),
        "department": ChoiceQuestion(
            instructions="Which team?", criteria={"billing": "Money.", "tech": "Code."}
        ),
    }
    plain = EncoderModel(_encode).answer_state("please refund now", questions)
    explicit_off = EncoderModel(_encode, cross_encoder=None).answer_state(
        "please refund now", questions
    )
    assert plain == explicit_off


def test_maybe_cross_encoder_is_off_without_the_env(monkeypatch) -> None:
    monkeypatch.delenv("TACHYONE_CROSS_ENCODER", raising=False)
    assert maybe_cross_encoder() is None


def test_maybe_cross_encoder_scopes_per_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("TACHYONE_CROSS_ENCODER", "tachyone-en=/nonexistent-teacher")
    # scoped specs match only the named checkpoint: the multi arm keeps the legacy path
    assert maybe_cross_encoder(checkpoint_id="tachyone-multi") is None
    # the named checkpoint resolves to a scorer (loading fails here only because the
    # teacher directory is fake or the train extra is absent - the routing decision
    # itself is what this asserts)
    try:
        scorer = maybe_cross_encoder(checkpoint_id="tachyone-en")
    except (FileNotFoundError, RuntimeError):
        scorer = "resolved"
    assert scorer is not None


def test_ensemble_averages_the_teachers_logits() -> None:
    class Fixed:
        def __init__(self, logits):
            self.logits = logits

        def temperature(self, kind, lang=None):
            return 1.0

        @property
        def calibration(self):
            return None

        def score(self, state_text, question_text, criterion_texts):
            return list(self.logits)

    ensemble = CrossEncoderEnsemble([Fixed([1.0, 3.0]), Fixed([3.0, 1.0])])
    assert ensemble.score("s", "q", ["a", "b"]) == [2.0, 2.0]
    with pytest.raises(ValueError, match="at least one scorer"):
        CrossEncoderEnsemble([])


def test_calibration_flattens_the_fit_and_the_argmax_cannot_move(tmp_path) -> None:
    from tachyone.backends.cross_encoder import _load_calibration

    assert _load_calibration(tmp_path) == {}  # no file -> raw softmax
    (tmp_path / "temperature_calibration.json").write_text(
        json.dumps(
            {
                "per_primitive": {
                    "choice": {"temperature": 1.4, "by_language": {"en": {"temperature": 1.2}}},
                    "noul": {"temperature": 0.8, "by_language": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    flat = _load_calibration(tmp_path)
    assert flat == {"choice": 1.4, "choice:en": 1.2, "noul": 0.8}
    scorer = _FakeScorer()
    scorer.temperatures = flat  # type: ignore[attr-defined]

    class Calibrated(_FakeScorer):
        def temperature(self, kind, lang=None):
            if lang and f"{kind}:{lang}" in flat:
                return flat[f"{kind}:{lang}"]
            return flat.get(kind, 1.0)

    from tachyone.primitives import Question

    model = EncoderModel(_encode, cross_encoder=Calibrated())
    question: dict[str, Question] = {
        "q": ChoiceQuestion(instructions="Pick.", criteria={"a": "aaaaaa", "b": "bb"})
    }
    hot = EncoderModel(_encode, cross_encoder=_FakeScorer()).answer_state("s", question)["q"]
    warm = model.answer_state("s", question)["q"]
    assert isinstance(hot, ChoiceAnswer) and isinstance(warm, ChoiceAnswer)
    assert warm.choice == hot.choice  # the temperature moves probabilities, never the answer
    assert warm.probabilities != hot.probabilities


def test_choice_probabilities_take_the_cells_reliability() -> None:
    """S2-v4 R2: the fitted cells set the peak and the tail rescales (argmax preserved)."""

    class _Cal:
        def __init__(self, value):
            self.value = value
            self.noul_knots = None

        def strength(self, embedding):
            return 0.9

        def cell_confidence(self, kind, lang, distribution, strength):
            return self.value

    class Mapped:
        cal = _Cal(0.6)

        def temperature(self, kind, lang=None):
            return 1.0

        @property
        def calibration(self):
            return self.cal

        def score(self, state_text, question_text, criterion_texts):
            return [float(len(t)) for t in criterion_texts]

    model = EncoderModel(_encode, cross_encoder=Mapped())
    answer = model.answer_state(
        "s", {"q": ChoiceQuestion(instructions="Pick.", criteria={"a": "aaaaaa", "b": "bb"})}
    )["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "a"
    assert answer.probabilities["a"] == pytest.approx(0.6)
    assert answer.confidence == pytest.approx(0.6)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
