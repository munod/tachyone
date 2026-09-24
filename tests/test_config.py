"""Unit tests for environment configuration (M2-T1, SERVE-06 / NFR-S02)."""

from __future__ import annotations

import pytest

from jeba.config import Config


def test_defaults_are_local_first() -> None:
    config = Config.from_env({})
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.device == "auto"
    assert config.backend == "llm"
    assert config.api_key is None
    assert config.llm_api_key is None
    assert config.models == ()
    assert config.preload == ()


def test_env_overrides() -> None:
    config = Config.from_env(
        {
            "JEBA_HOST": "0.0.0.0",
            "JEBA_PORT": "9001",
            "JEBA_DEVICE": "cuda",
            "JEBA_BACKEND": "fake",
            "JEBA_MODELS": "jeba-en, jeba-multi",
            "JEBA_PRELOAD": "jeba-en",
            "JEBA_THREADS": "8",
            "JEBA_API_KEY": "secret-server",
            "JEBA_LLM_BASE_URL": "http://localhost:11434/v1",
            "JEBA_LLM_MODEL": "llama3",
            "JEBA_LLM_RETRIES": "4",
        }
    )
    assert config.host == "0.0.0.0"
    assert config.port == 9001
    assert config.device == "cuda"
    assert config.backend == "fake"
    assert config.models == ("jeba-en", "jeba-multi")
    assert config.preload == ("jeba-en",)
    assert config.threads == 8
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_model == "llama3"
    assert config.llm_retries == 4


def test_openai_api_key_is_a_fallback_for_llm_key() -> None:
    assert Config.from_env({"OPENAI_API_KEY": "sk-x"}).llm_api_key == "sk-x"
    assert Config.from_env({"JEBA_LLM_API_KEY": "a", "OPENAI_API_KEY": "b"}).llm_api_key == "a"


def test_secrets_are_not_in_repr() -> None:
    config = Config.from_env({"JEBA_API_KEY": "top-secret", "JEBA_LLM_API_KEY": "llm-secret"})
    text = repr(config)
    assert "top-secret" not in text
    assert "llm-secret" not in text


@pytest.mark.parametrize(
    "env",
    [
        {"JEBA_PORT": "0"},
        {"JEBA_PORT": "70000"},
        {"JEBA_PORT": "not-a-number"},
        {"JEBA_BACKEND": "quantum"},
        {"JEBA_DEVICE": "tpu"},
        {"JEBA_THREADS": "-1"},
        {"JEBA_LLM_TIMEOUT": "0"},
        {"JEBA_LLM_RETRIES": "99"},
    ],
)
def test_invalid_values_fail_fast(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        Config.from_env(env)


def test_empty_strings_fall_back_to_defaults() -> None:
    config = Config.from_env({"JEBA_HOST": "", "JEBA_API_KEY": "", "JEBA_LLM_API_KEY": ""})
    assert config.host == "127.0.0.1"
    assert config.api_key is None
    assert config.llm_api_key is None


def test_models_dir_defaults_and_override() -> None:
    assert Config.from_env({}).models_dir.endswith(".cache/jeba/models")
    assert Config.from_env({"JEBA_MODELS_DIR": "/models"}).models_dir == "/models"
