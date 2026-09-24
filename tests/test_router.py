"""Unit tests for script/language routing (M3-T1, ROUTE-01/02/03/05)."""

from __future__ import annotations

import time

import pytest

from jeba.router import (
    ENGLISH,
    MULTILINGUAL,
    SUPPORTED_LANGUAGES,
    Router,
    detect_language,
    detect_script,
    state_text,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("Please refund the duplicate charge.", "latin"),
        ("Пожалуйста, верните деньги.", "cyrillic"),
        ("من فضلك استرد المبلغ.", "arabic"),
        ("请退款这笔重复收费。", "han"),
        ("कृपया धन वापस करें।", "devanagari"),
        ("환불해 주세요.", "hangul"),
        ("กรุณาคืนเงินด้วย", "thai"),
        ("12345 !!!", "unknown"),
    ],
)
def test_detect_script(text: str, script: str) -> None:
    assert detect_script(text) == script


def test_english_text_routes_to_english() -> None:
    decision = Router().route("Please refund the duplicate charge today.")
    assert decision.checkpoint_id == ENGLISH
    assert decision.detected_script == "latin"


@pytest.mark.parametrize(
    "text",
    [
        "Пожалуйста, верните деньги.",
        "من فضلك استرد المبلغ.",
        "请退款这笔重复收费。",
        "환불해 주세요.",
    ],
)
def test_non_latin_routes_to_multilingual(text: str) -> None:
    decision = Router().route(text)
    assert decision.checkpoint_id == MULTILINGUAL
    assert decision.detected_script != "latin"


def test_non_english_latin_routes_to_multilingual() -> None:
    assert Router().route("hola gracias por el reembolso").checkpoint_id == MULTILINGUAL
    assert Router().route("devolva a cobrança duplicada").checkpoint_id == MULTILINGUAL


@pytest.mark.parametrize(
    "text",
    [
        "Quero cancelar minha assinatura agora",
        "meu pagamento falhou duas vezes",
        "esqueci minha senha",
        "preciso de ajuda com uma cobrança duplicada",
        "gostaria de falar sobre o problema no pedido",
    ],
)
def test_portuguese_without_accents_routes_to_multilingual(text: str) -> None:
    decision = Router().route(text)
    assert decision.checkpoint_id == MULTILINGUAL
    assert decision.detected_language == "pt"


def test_detect_language_identifies_portuguese() -> None:
    assert detect_language("quero cancelar agora", "latin") == "pt"
    assert detect_language("please refund the charge", "latin") == "en"


def test_unknown_script_uses_default() -> None:
    router = Router(default=MULTILINGUAL)
    assert router.route("12345 !!!").checkpoint_id == MULTILINGUAL


def test_state_object_is_serialized_for_routing() -> None:
    decision = Router().route({"body": "请退款"})
    assert decision.checkpoint_id == MULTILINGUAL


def test_lang_guess_code_overrides_heuristic() -> None:
    assert Router().route("ok").checkpoint_id == ENGLISH
    assert Router(lang_guess=lambda _s: "pt").route("ok").checkpoint_id == MULTILINGUAL
    assert (
        Router(lang_guess=lambda _s: "en_US.UTF-8").route("hola gracias").checkpoint_id == ENGLISH
    )


def test_lang_guess_abstaining_falls_back_to_detection() -> None:
    assert Router(lang_guess=lambda _s: None).route("Пожалуйста").checkpoint_id == MULTILINGUAL


def test_supported_languages_covers_100_plus() -> None:
    assert len(set(SUPPORTED_LANGUAGES)) >= 100
    assert "en" in SUPPORTED_LANGUAGES
    assert "pt" in SUPPORTED_LANGUAGES


def test_state_text_handles_objects() -> None:
    assert "refund" in state_text({"body": "refund"})
    assert state_text("plain") == "plain"


def test_routing_overhead_is_small() -> None:
    router = Router()
    samples = ["Please refund", "Пожалуйста верните", "请退款", "hola gracias"]
    start = time.perf_counter()
    for index in range(500):
        router.route(samples[index % len(samples)])
    mean_ms = (time.perf_counter() - start) / 500 * 1000
    assert mean_ms < 5.0  # target is <0.5 ms; loose bound to avoid CI flakiness
