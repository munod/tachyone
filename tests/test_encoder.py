"""Unit tests for the encoder backend (M3-T5, BACK-03 / BACK-06 / BACK-07).

The decision math is exercised with an injected deterministic encoder, so these tests need
no torch and no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tachyone.backends.encoder import (
    ChoiceScorer,
    EncoderBackend,
    EncoderCheckpoint,
    EncoderModel,
    apply_adapters,
    load_choice_head,
    load_encoder,
    load_temperatures,
)
from tachyone.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from tachyone.router import DEFAULT_CHECKPOINTS, ENGLISH, MULTILINGUAL, CheckpointInfo, Router
from tachyone.wire import SystemOneRequest, answer

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
    result = await backend.predict(_QUESTIONS, state="please refund", model="tachyone-latest")
    assert isinstance(result.answers["is_urgent"], NoulAnswer)
    assert isinstance(result.answers["department"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)


@pytest.mark.asyncio
async def test_distributions_are_normalized_and_confident() -> None:
    backend, _ = _backend()
    result = await backend.predict(_QUESTIONS, state="please refund", model="tachyone-latest")
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
    result = await backend.predict({}, state="x", model="tachyone-latest")
    assert result.answers == {}
    assert result.usage.input_tokens == 0
    assert loader.loaded == []


@pytest.mark.asyncio
async def test_routes_multilingual_for_non_latin() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model="tachyone-latest")
    assert loader.loaded == [MULTILINGUAL]


@pytest.mark.asyncio
async def test_explicit_model_overrides_routing() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model=ENGLISH)
    assert loader.loaded == [ENGLISH]


@pytest.mark.asyncio
async def test_encoder_backend_passes_the_contract() -> None:
    backend, _ = _backend()
    request = SystemOneRequest(state="help", model="tachyone-en", questions=_QUESTIONS)
    response = await answer(request, backend)
    assert set(response.answers) == set(_QUESTIONS)
    dumped = response.model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}


def test_from_config_injects_encoder() -> None:
    from tachyone.config import Config

    backend = EncoderBackend.from_config(
        Config.from_env({"TACHYONE_BACKEND": "encoder"}), encode=_encode
    )
    assert backend.name == "encoder"


def test_load_encoder_requires_train_extra() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the real encoder path is exercised elsewhere")
    with pytest.raises(RuntimeError, match="train extra"):
        load_encoder(DEFAULT_CHECKPOINTS[ENGLISH], models_dir="/tmp/tachyone-models", device="cpu")


def test_default_checkpoints_declare_base_and_adapter() -> None:
    english = DEFAULT_CHECKPOINTS[ENGLISH]
    multilingual = DEFAULT_CHECKPOINTS[MULTILINGUAL]
    assert english.base_model == "answerdotai/ModernBERT-large"
    assert english.adapter == "munod/tachyone-en"
    assert multilingual.base_model == "jhu-clsp/mmBERT-base"
    assert multilingual.adapter == "munod/tachyone-multi"


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


def test_load_temperatures_reads_per_language(tmp_path: Path) -> None:
    (tmp_path / "temperature_calibration.json").write_text(
        json.dumps(
            {
                "per_primitive": {
                    "choice": {"temperature": 2.0, "by_language": {"pt": {"temperature": 3.0}}}
                }
            }
        ),
        encoding="utf-8",
    )
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=str(tmp_path))
    assert load_temperatures(info, models_dir=str(tmp_path)) == {"choice": 2.0, "choice:pt": 3.0}


def test_per_language_temperature_overrides_per_primitive() -> None:
    question = NoulQuestion(instructions="Does this request urgency?")
    temperatures = {"noul": 1.0, "noul:pt": 0.05}
    tuned = EncoderModel(_encode, temperatures=temperatures).answer_state(
        "please refund now", {"q": question}, lang="pt"
    )["q"]
    generic = EncoderModel(_encode, temperatures=temperatures).answer_state(
        "please refund now", {"q": question}, lang="en"
    )["q"]
    assert isinstance(tuned, NoulAnswer) and isinstance(generic, NoulAnswer)
    assert abs(tuned.noul - 0.5) >= abs(generic.noul - 0.5)


def test_runtime_detects_language_for_temperature() -> None:
    question = NoulQuestion(instructions="Does this request urgency?")
    text = "Preciso de um reembolso hoje."
    temperatures = {"noul": 1.0, "noul:pt": 0.05, "noul:es": 4.0}
    auto = EncoderModel(_encode, temperatures=temperatures).answer_state(text, {"q": question})["q"]
    forced_pt = EncoderModel(_encode, temperatures=temperatures).answer_state(
        text, {"q": question}, lang="pt"
    )["q"]
    forced_es = EncoderModel(_encode, temperatures=temperatures).answer_state(
        text, {"q": question}, lang="es"
    )["q"]
    assert isinstance(auto, NoulAnswer) and isinstance(forced_pt, NoulAnswer)
    assert isinstance(forced_es, NoulAnswer)
    assert auto.noul == pytest.approx(forced_pt.noul)  # auto-detection chose Portuguese
    assert auto.noul != pytest.approx(forced_es.noul)


def test_from_config_applies_adapter_overrides() -> None:
    from tachyone.config import Config

    backend = EncoderBackend.from_config(
        Config.from_env(
            {"TACHYONE_BACKEND": "encoder", "TACHYONE_ADAPTERS": "tachyone-en=acme/tuned"}
        ),
        encode=_encode,
    )
    assert backend._router.checkpoints[ENGLISH].adapter == "acme/tuned"
    assert backend._router.checkpoints[MULTILINGUAL].adapter == "munod/tachyone-multi"


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


def test_cuda_graph_encode_matches_eager_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tachyone.backends.encoder import _cuda_graph_encode

    class _Out:
        def __init__(self, last_hidden_state: object) -> None:
            self.last_hidden_state = last_hidden_state

    class _Tiny(torch.nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(64, dim)

        def forward(self, input_ids, attention_mask):
            return _Out(self.emb(input_ids))

    torch.manual_seed(0)
    model = _Tiny().cuda().eval()

    def tokenize(texts: list[str]):
        width = max(len(text) for text in texts)
        ids = torch.zeros((len(texts), width), dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            length = len(text)
            ids[row, :length] = torch.arange(1, length + 1, device="cuda")
            mask[row, :length] = 1
        return ids, mask

    def eager_pool(texts: list[str]) -> list[list[float]]:
        ids, mask = tokenize(texts)
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        return pooled.cpu().tolist()

    encode = _cuda_graph_encode(model, tokenize, device="cuda")
    batch = ["aa", "bbb", "c"]
    first = encode(batch)  # captures and replays
    second = encode(batch)  # replays the captured graph
    expected = eager_pool(batch)
    for produced in (first, second):
        assert len(produced) == len(batch)
        for row, reference in zip(produced, expected, strict=True):
            assert row == pytest.approx(reference, abs=1e-4)
    # A different shape captures a second graph and still matches the eager forward.
    other = eager_pool(["dddd", "ee"])
    for row, reference in zip(encode(["dddd", "ee"]), other, strict=True):
        assert row == pytest.approx(reference, abs=1e-4)
