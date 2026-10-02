"""Unit tests for the encoder backend (M3-T5, BACK-03 / BACK-06 / BACK-07).

The decision math is exercised with an injected deterministic encoder, so these tests need
no torch and no network.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from tachyone.backends.encoder import (
    ChoiceHeadBank,
    ChoiceScorer,
    DomainChoiceHead,
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
    JsonValue,
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


def test_english_context_reads_long_states_without_truncating_them() -> None:
    """B-13 P1b: runtime truncation must cover the long-policy cases that motivated it.

    The JevBench hard tier's states average 1,079 tokens (max 3,746); at the previous
    512 the deciding facts never reached the encoder. 4096 covers that tier with margin
    and stays inside ModernBERT-large's trained 8192 positions.
    """
    english = DEFAULT_CHECKPOINTS[ENGLISH]
    assert english.context == 4096
    assert english.context <= 8192
    assert DEFAULT_CHECKPOINTS[MULTILINGUAL].context == 1024


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
    bank = ChoiceHeadBank(shared=_zero_scorer())
    base = EncoderModel(_encode).answer_state("please refund", {"q": question})["q"]
    scored = EncoderModel(_encode, choice_bank=bank).answer_state("please refund", {"q": question})[
        "q"
    ]
    assert isinstance(base, ChoiceAnswer) and isinstance(scored, ChoiceAnswer)
    assert scored.probabilities == base.probabilities  # residual is exactly zero


def test_choice_scorer_changes_and_is_permutation_equivariant() -> None:
    forward = _choice_question()
    reversed_question = ChoiceQuestion(
        instructions="Which team?",
        criteria={"sales": "pricing", "technical": "bugs", "billing": "invoices"},
    )
    model = EncoderModel(_encode, choice_bank=ChoiceHeadBank(shared=_random_scorer()))
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
    """The legacy single-scorer asset loads as a shared-only bank (ADR-0016 §2)."""
    (tmp_path / "choice_head.json").write_text(
        json.dumps({"rank": 1, "w1": [[0.0] * 16], "w2": [[0.0] * 8]}), encoding="utf-8"
    )
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=str(tmp_path))
    bank = load_choice_head(info, models_dir=str(tmp_path))
    assert isinstance(bank, ChoiceHeadBank)
    assert isinstance(bank.shared, ChoiceScorer)
    assert bank.domains == {}  # no gate keys: every question falls to shared, as before


def test_load_choice_head_without_adapter_is_none() -> None:
    info = CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=None)
    assert load_choice_head(info, models_dir="/tmp") is None


def test_load_choice_head_keyed_asset(tmp_path: Path) -> None:
    """The keyed `{shared, domains}` shape loads every shipped head and its signatures."""
    scorer = {"rank": 1, "w1": [[0.5] * 16], "w2": [[0.5] * 8]}
    (tmp_path / "choice_head.json").write_text(
        json.dumps(
            {
                "shared": scorer,
                "domains": {
                    "support": {"head": scorer, "signatures": ["billing", "invoices"]},
                    "voice": {"head": scorer, "signatures": ["thermostat", "lights"]},
                },
            }
        ),
        encoding="utf-8",
    )
    info = _info(str(tmp_path))
    bank = load_choice_head(info, models_dir=str(tmp_path))
    assert isinstance(bank, ChoiceHeadBank)
    assert set(bank.domains) == {"support", "voice"}
    assert bank.domains["support"].signatures == ("billing", "invoices")
    assert isinstance(bank.shared, ChoiceScorer)


# --- ADR-0016 gate: hint → lexical signature match → shared ----------------------------------


def _domain_question(
    instructions: str = "Where does this belong?",
    criteria: dict[str, JsonValue | None] | None = None,
) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=criteria or {"a": "x", "b": "y"})


def _bank(**domains: tuple[str, ...]) -> ChoiceHeadBank:
    """A bank whose domain heads are all the zero scorer (residual-free) with given signatures."""
    return ChoiceHeadBank(
        shared=_zero_scorer(),
        domains={
            name: DomainChoiceHead(head=_zero_scorer(), signatures=signatures)
            for name, signatures in domains.items()
        },
    )


def test_gate_hint_wins_when_the_asset_ships_that_head() -> None:
    bank = _bank(support=("billing",), voice=("thermostat",))
    question = _domain_question("Which team handles this? invoices billing")
    assert bank.select_key(question) == "support"
    assert bank.select_key(question, hint="voice") == "voice"  # caller overrides the gate


def test_gate_unknown_hint_falls_through_to_the_lexical_rule() -> None:
    bank = _bank(support=("billing",), voice=("thermostat",))
    question = _domain_question("Which device? thermostat temperature")
    assert bank.select_key(question, hint="nonexistent") == "voice"


def test_gate_strictly_highest_signature_count_wins() -> None:
    # "billing" + "invoices" + "refunds" all appear: support scores 3, voice 0.
    bank = _bank(support=("billing", "invoices", "refunds"), voice=("thermostat",))
    question = _domain_question("billing invoices refunds")
    assert bank.select_key(question) == "support"
    assert bank.select_key(question) is not None


def test_gate_tie_or_no_match_falls_to_shared() -> None:
    bank = _bank(support=("billing",), voice=("thermostat",))
    # Both domains score 0 → ambiguity → shared.
    assert bank.select_key(_domain_question("nothing relevant here")) is None
    # Tie 1-1 → shared.
    assert bank.select_key(_domain_question("billing thermostat")) is None
    # Shared-only bank (legacy asset): always shared, no gate.
    legacy = ChoiceHeadBank(shared=_zero_scorer())
    assert legacy.select_key(_domain_question("billing invoices")) is None


def test_gate_signature_matching_ignores_case_accents_and_punctuation() -> None:
    bank = _bank(support=("duplicate charge",), voice=())
    question = _domain_question("About my DUPLICATE-Charge… please")
    assert bank.select_key(question) == "support"


def test_gate_matches_on_labels_and_descriptions_of_the_question() -> None:
    """The gate sees instructions + option labels + descriptions (ADR-0016 §2.2)."""
    question = ChoiceQuestion(
        instructions="Which queue?",
        criteria={"shipping": "tracking, delivery dates", "returns": None},
    )
    bank = _bank(voice=("thermostat",), ecommerce=("shipping", "tracking"))
    assert bank.select_key(question) == "ecommerce"


def test_model_selects_one_head_per_choice_question() -> None:
    """Two choice questions in one request each get their own domain head."""
    support_question = ChoiceQuestion(
        instructions="Which team handles this request?", criteria={"billing": "invoices"}
    )
    voice_question = ChoiceQuestion(
        instructions="Which device group?", criteria={"climate": "thermostats"}
    )
    support_head = _random_scorer(seed=1)
    voice_head = _random_scorer(seed=2)
    bank = ChoiceHeadBank(
        shared=_zero_scorer(),
        domains={
            "support": DomainChoiceHead(head=support_head, signatures=("billing", "invoices")),
            "voice": DomainChoiceHead(head=voice_head, signatures=("thermostat", "climate")),
        },
    )
    model = EncoderModel(_encode, choice_bank=bank)
    answers = model.answer_state(
        "state", {"a": support_question, "b": voice_question}, choice_head=None
    )
    # Compare against each head applied alone: the gate picked the matching head per question.
    support_only = EncoderModel(
        _encode, choice_bank=ChoiceHeadBank(shared=support_head)
    ).answer_state("state", {"a": support_question})["a"]
    voice_only = EncoderModel(_encode, choice_bank=ChoiceHeadBank(shared=voice_head)).answer_state(
        "state", {"b": voice_question}
    )["b"]
    assert isinstance(answers["a"], ChoiceAnswer) and isinstance(support_only, ChoiceAnswer)
    assert isinstance(answers["b"], ChoiceAnswer) and isinstance(voice_only, ChoiceAnswer)
    assert answers["a"].probabilities == support_only.probabilities
    assert answers["b"].probabilities == voice_only.probabilities


def test_noul_and_score_ignore_the_head_bank() -> None:
    bank = _bank(support=("billing",))
    question = NoulQuestion(instructions="Is this a request?")
    base = EncoderModel(_encode).answer_state("please refund", {"q": question})["q"]
    with_bank = EncoderModel(_encode, choice_bank=bank).answer_state(
        "please refund", {"q": question}
    )["q"]
    assert isinstance(base, NoulAnswer) and isinstance(with_bank, NoulAnswer)
    assert with_bank.noul == base.noul


# --- B-8: an unusable calibration asset must be loud, an absent one must stay silent -------


def _info(adapter: str | None) -> CheckpointInfo:
    return CheckpointInfo(id="x", languages=["*"], context=8, size_params=0, adapter=adapter)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and record.name == "tachyone.backends.encoder"
    ]


def test_clean_assets_load_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "temperature_calibration.json").write_text(
        json.dumps({"per_primitive": {"noul": {"temperature": 4.0}}}), encoding="utf-8"
    )
    (tmp_path / "choice_head.json").write_text(
        json.dumps({"rank": 1, "w1": [[0.0] * 16], "w2": [[0.0] * 8]}), encoding="utf-8"
    )
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path)) == {"noul": 4.0}
        assert isinstance(load_choice_head(info, models_dir=str(tmp_path)), ChoiceHeadBank)
    assert _warnings(caplog) == []


def test_local_adapter_without_assets_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An adapter that simply does not ship calibration is a documented, normal case."""
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path)) == {}
        assert load_choice_head(info, models_dir=str(tmp_path)) is None
    assert _warnings(caplog) == []


def test_unreadable_temperature_file_warns_naming_file_and_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "temperature_calibration.json"
    path.write_text("{not json", encoding="utf-8")
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path)) == {}  # degraded, not fatal
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "temperature_calibration.json" in messages[0]
    assert str(path) in messages[0]
    assert "continuing without it" in messages[0]


def test_temperature_file_with_a_non_numeric_value_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A structurally valid file with a corrupt value used to raise out of the loader."""
    (tmp_path / "temperature_calibration.json").write_text(
        json.dumps({"per_primitive": {"noul": {"temperature": "hot"}}}), encoding="utf-8"
    )
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path)) == {}
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "temperature_calibration.json" in messages[0]


def test_temperature_file_that_is_not_an_object_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "temperature_calibration.json").write_text("[1, 2, 3]", encoding="utf-8")
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path)) == {}
    assert any("expected a JSON object" in message for message in _warnings(caplog))


def test_corrupt_choice_head_warns_and_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "choice_head.json").write_text(json.dumps({"w1": "oops"}), encoding="utf-8")
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_choice_head(info, models_dir=str(tmp_path)) is None
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "choice_head.json" in messages[0]
    assert str(tmp_path) in messages[0]


def test_corrupt_keyed_entry_degrades_only_itself(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """B-8 extended to the keyed form: one broken head must not take the bank down."""
    good = {"rank": 1, "w1": [[0.0] * 16], "w2": [[0.0] * 8]}
    (tmp_path / "choice_head.json").write_text(
        json.dumps(
            {
                "shared": good,
                "domains": {
                    "support": {"head": good, "signatures": ["billing"]},
                    "broken": {"head": {"w1": "oops"}, "signatures": ["x"]},
                    "no_head": {"signatures": ["y"]},
                },
            }
        ),
        encoding="utf-8",
    )
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        bank = load_choice_head(info, models_dir=str(tmp_path))
    assert isinstance(bank, ChoiceHeadBank)
    assert isinstance(bank.shared, ChoiceScorer)  # shared survives a broken domain
    assert set(bank.domains) == {"support"}  # the two broken entries dropped, one warned each
    messages = _warnings(caplog)
    assert len(messages) == 2
    assert all("choice_head.json" in message for message in messages)
    assert any("'broken'" in message for message in messages)
    assert any("'no_head'" in message for message in messages)


def test_corrupt_keyed_shared_keeps_the_domain_heads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = {"rank": 1, "w1": [[0.0] * 16], "w2": [[0.0] * 8]}
    (tmp_path / "choice_head.json").write_text(
        json.dumps(
            {"shared": {"w1": "oops"}, "domains": {"voice": {"head": good, "signatures": ["x"]}}}
        ),
        encoding="utf-8",
    )
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        bank = load_choice_head(info, models_dir=str(tmp_path))
    assert isinstance(bank, ChoiceHeadBank)
    assert bank.shared is None  # unmatched questions degrade to the baseline…
    assert set(bank.domains) == {"voice"}  # …while the usable heads keep working
    assert any("invalid shared head" in message for message in _warnings(caplog))


def test_keyed_asset_with_nothing_usable_warns_and_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "choice_head.json").write_text(
        json.dumps({"shared": {"w1": "oops"}, "domains": {"x": {"head": {"w1": "oops"}}}}),
        encoding="utf-8",
    )
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_choice_head(info, models_dir=str(tmp_path)) is None
    assert len(_warnings(caplog)) == 2


def test_offline_local_adapter_without_assets_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """TACHYONE_OFFLINE=1 plus an incomplete cache is the B-8 scenario: say it out loud."""
    info = _info(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir=str(tmp_path), offline=True) == {}
        assert load_choice_head(info, models_dir=str(tmp_path), offline=True) is None
    messages = _warnings(caplog)
    assert len(messages) == 2
    assert "temperature_calibration.json" in messages[0]
    assert "choice_head.json" in messages[1]
    assert "TACHYONE_OFFLINE=1" in messages[0]


@pytest.mark.parametrize(
    ("error_name", "expects_warning"),
    [
        ("EntryNotFoundError", False),  # the repo does not ship the asset: silence is correct
        ("LocalEntryNotFoundError", True),  # not cached: a partial prefetch looks like this
        ("RepositoryNotFoundError", True),  # a mis-typed adapter id, not a missing file
        ("HfHubHTTPError", True),  # network/500: degrading silently hides it
    ],
)
def test_hub_download_failures_are_classified(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_name: str,
    expects_warning: bool,
) -> None:
    import sys
    import types

    module = types.ModuleType("huggingface_hub")
    error_type = type(error_name, (Exception,), {})

    def _download(**kwargs: object) -> str:
        raise error_type(f"{error_name} for {kwargs['filename']}")

    module.hf_hub_download = _download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    info = _info("acme/tuned")  # a repo id, so the hub branch is taken
    with caplog.at_level(logging.WARNING, logger="tachyone.backends.encoder"):
        assert load_temperatures(info, models_dir="/tmp") == {}

    messages = _warnings(caplog)
    assert bool(messages) is expects_warning
    if expects_warning:
        assert "temperature_calibration.json" in messages[0]
        assert "acme/tuned" in messages[0]


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
