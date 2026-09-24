"""Unit tests for the encoder backend (M3-T5, BACK-03 / BACK-06 / BACK-07).

The decision math is exercised with an injected deterministic encoder, so these tests need
no torch and no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jeba.backends.encoder import (
    ChoiceScorer,
    EncoderBackend,
    EncoderCheckpoint,
    EncoderModel,
    apply_adapters,
    load_choice_head,
    load_encoder,
    load_temperatures,
)
from jeba.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from jeba.router import DEFAULT_CHECKPOINTS, ENGLISH, MULTILINGUAL, CheckpointInfo, Router
from jeba.wire import SystemOneRequest, answer

pytestmark = pytest.mark.contract


def _encode(texts: list[str]) -> list[list[float]]:
    return [
        [float(sum(map(ord, text)) % 97) / 97 + index * 0.01 for index in range(8)]
        for text in texts
    ]


_QUESTIONS: dict[str, Question] = {
    "is_urgent": NoulQuestion(instructions="Does this express urgency?"),
    "department": ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "refunds", "technical": "bugs", "sales": "pricing"},
    ),
    "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"]),
}


class _Loader:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def __call__(self, info: CheckpointInfo) -> EncoderCheckpoint:
        self.loaded.append(info.id)
        return EncoderCheckpoint(info, _encode)


def _backend(loader: _Loader | None = None) -> tuple[EncoderBackend, _Loader]:
    used = loader or _Loader()
    return EncoderBackend(Router(loader=used, max_loaded=2)), used


@pytest.mark.asyncio
async def test_all_primitives_answered() -> None:
    backend, _ = _backend()
    result = await backend.predict(_QUESTIONS, state="please refund", model="jeba-latest")
    assert isinstance(result.answers["is_urgent"], NoulAnswer)
    assert isinstance(result.answers["department"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)


@pytest.mark.asyncio
async def test_distributions_are_normalized_and_confident() -> None:
    backend, _ = _backend()
    result = await backend.predict(_QUESTIONS, state="please refund", model="jeba-latest")
    department = result.answers["department"]
    urgency = result.answers["urgency"]
    assert isinstance(department, ChoiceAnswer)
    assert isinstance(urgency, ScoreAnswer)
    assert abs(sum(department.probabilities.values()) - 1.0) < 1e-9
    assert abs(sum(urgency.probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= department.confidence <= 1.0
    assert 0.0 <= urgency.confidence <= 1.0
    assert urgency.legend == {0: "low", 1: "medium", 2: "high"}
    noul = result.answers["is_urgent"]
    assert isinstance(noul, NoulAnswer)
    assert 0.0 <= noul.noul <= 1.0


@pytest.mark.asyncio
async def test_empty_questions_short_circuits() -> None:
    backend, loader = _backend()
    result = await backend.predict({}, state="x", model="jeba-latest")
    assert result.answers == {}
    assert result.usage.input_tokens == 0
    assert loader.loaded == []


@pytest.mark.asyncio
async def test_routes_multilingual_for_non_latin() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model="jeba-latest")
    assert loader.loaded == [MULTILINGUAL]


@pytest.mark.asyncio
async def test_explicit_model_overrides_routing() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model=ENGLISH)
    assert loader.loaded == [ENGLISH]


@pytest.mark.asyncio
async def test_encoder_backend_passes_the_contract() -> None:
    backend, _ = _backend()
    request = SystemOneRequest(state="help", model="jeba-en", questions=_QUESTIONS)
    response = await answer(request, backend)
    assert set(response.answers) == set(_QUESTIONS)
    dumped = response.model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}


def test_from_config_injects_encoder() -> None:
    from jeba.config import Config

    backend = EncoderBackend.from_config(
        Config.from_env({"JEBA_BACKEND": "encoder"}), encode=_encode
    )
    assert backend.name == "encoder"


def test_load_encoder_requires_train_extra() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the real encoder path is exercised elsewhere")
    with pytest.raises(RuntimeError, match="train extra"):
        load_encoder(DEFAULT_CHECKPOINTS[ENGLISH], models_dir="/tmp/jeba-models", device="cpu")


def test_default_checkpoints_declare_base_and_adapter() -> None:
    english = DEFAULT_CHECKPOINTS[ENGLISH]
    multilingual = DEFAULT_CHECKPOINTS[MULTILINGUAL]
    assert english.base_model == "answerdotai/ModernBERT-large"
    assert english.adapter == "munod/jeba-en"
    assert multilingual.base_model == "jhu-clsp/mmBERT-base"
    assert multilingual.adapter == "munod/jeba-multi"


def test_apply_adapters_overrides_and_disables() -> None:
    router = Router()
    apply_adapters(router, {ENGLISH: "acme/tuned-en", MULTILINGUAL: None})
    assert router.checkpoints[ENGLISH].adapter == "acme/tuned-en"
    assert router.checkpoints[MULTILINGUAL].adapter is None


def test_load_temperatures_from_local_adapter_dir(tmp_path: Path) -> None:
    (tmp_path / "temperature_calibration.json").write_text(
        json.dumps(
            {"per_primitive": {"noul": {"temperature": 4.0}, "choice": {"temperature": 1.5}}}
        ),
        encoding="utf-8",
    )
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=str(tmp_path))
    assert load_temperatures(info, models_dir=str(tmp_path)) == {"noul": 4.0, "choice": 1.5}


def test_load_temperatures_without_adapter_is_empty() -> None:
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=None)
    assert load_temperatures(info, models_dir="/tmp") == {}


def test_per_primitive_temperature_sharpens_noul() -> None:
    question = NoulQuestion(instructions="Does this request urgency?")
    base = EncoderModel(_encode).answer_state("please refund now", {"q": question})["q"]
    tuned = EncoderModel(_encode, temperatures={"noul": 0.25}).answer_state(
        "please refund now", {"q": question}
    )["q"]
    assert isinstance(base, NoulAnswer) and isinstance(tuned, NoulAnswer)
    assert abs(tuned.noul - 0.5) >= abs(base.noul - 0.5)


def test_from_config_applies_adapter_overrides() -> None:
    from jeba.config import Config

    backend = EncoderBackend.from_config(
        Config.from_env({"JEBA_BACKEND": "encoder", "JEBA_ADAPTERS": "jeba-en=acme/tuned"}),
        encode=_encode,
    )
    assert backend._router.checkpoints[ENGLISH].adapter == "acme/tuned"
    assert backend._router.checkpoints[MULTILINGUAL].adapter == "munod/jeba-multi"


def _choice_question() -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "invoices", "technical": "bugs", "sales": "pricing"},
    )


def _zero_scorer(rank: int = 1, hidden: int = 16) -> ChoiceScorer:
    return ChoiceScorer([[0.0] * hidden] * rank, [[0.0] * (hidden // 2)] * rank)


def _random_scorer(rank: int = 2, hidden: int = 16, seed: int = 0) -> ChoiceScorer:
    import random

    rng = random.Random(seed)
    w1 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(rank)]
    w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden // 2)] for _ in range(rank)]
    return ChoiceScorer(w1, w2)


def test_zero_choice_scorer_matches_baseline() -> None:
    question = _choice_question()
    base = EncoderModel(_encode).answer_state("please refund", {"q": question})["q"]
    scored = EncoderModel(_encode, choice_scorer=_zero_scorer()).answer_state(
        "please refund", {"q": question}
    )["q"]
    assert isinstance(base, ChoiceAnswer) and isinstance(scored, ChoiceAnswer)
    assert scored.probabilities == base.probabilities  # residual is exactly zero


def test_choice_scorer_changes_and_is_permutation_equivariant() -> None:
    forward = _choice_question()
    reversed_question = ChoiceQuestion(
        instructions="Which team?",
        criteria={"sales": "pricing", "technical": "bugs", "billing": "invoices"},
    )
    model = EncoderModel(_encode, choice_scorer=_random_scorer())
    scored = model.answer_state("outage now", {"q": forward})["q"]
    permuted = model.answer_state("outage now", {"q": reversed_question})["q"]
    baseline = EncoderModel(_encode).answer_state("outage now", {"q": forward})["q"]
    assert isinstance(scored, ChoiceAnswer)
    assert isinstance(permuted, ChoiceAnswer)
    assert isinstance(baseline, ChoiceAnswer)
    assert scored.probabilities != baseline.probabilities
    for option, probability in scored.probabilities.items():
        assert probability == pytest.approx(permuted.probabilities[option])


def test_load_choice_head_from_local_dir(tmp_path: Path) -> None:
    (tmp_path / "choice_head.json").write_text(
        json.dumps({"rank": 1, "w1": [[0.0] * 16], "w2": [[0.0] * 8]}), encoding="utf-8"
    )
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=str(tmp_path))
    assert isinstance(load_choice_head(info, models_dir=str(tmp_path)), ChoiceScorer)


def test_load_choice_head_without_adapter_is_none() -> None:
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=None)
    assert load_choice_head(info, models_dir="/tmp") is None
