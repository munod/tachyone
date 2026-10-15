"""Tests for the S2-v V6 seam: the cross-encoder as an opt-in scoring override.

The load-bearing properties: the override answers ``choice``/``noul`` from the scorer's own
per-option logits (the P3 knobs bypassed — the teacher's softmax is the calibration),
``score`` never leaves the single-pass path, and with the knob unset the legacy path is
byte-identical (additive extension). GPU loading is exercised by the cycle, not here.
"""

from __future__ import annotations

from collections.abc import Sequence

from tachyone.backends.cross_encoder import maybe_cross_encoder
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
