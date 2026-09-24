"""Environment configuration: the single source of truth for server/backend settings.

All settings come from ``JEBA_*`` environment variables with safe local-first defaults.
Secrets (``JEBA_API_KEY``, ``JEBA_LLM_API_KEY``) are never included in ``repr`` so they
cannot be logged accidentally (NFR-S02). Invalid values fail fast at startup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

BACKENDS = ("llm", "encoder", "onnx", "fake")
DEVICES = ("auto", "cpu", "cuda", "mps")

#: Default backend for M2. It flips to ``encoder`` in M3 once the local engine lands.
DEFAULT_BACKEND = "llm"


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or value == "" else value


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, *, lo: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if value < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _text(env, name, default)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in env.get(name, "").split(",") if item.strip())


def _default_models_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "jeba", "models")


@dataclass(frozen=True, slots=True)
class Config:
    """Validated runtime configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "auto"
    backend: str = DEFAULT_BACKEND
    models: tuple[str, ...] = ()
    models_dir: str = field(default_factory=_default_models_dir)
    preload: tuple[str, ...] = ()
    threads: int = 0  # 0 = let the runtime decide
    api_key: str | None = field(default=None, repr=False)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = field(default=None, repr=False)
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 30.0
    llm_retries: int = 2

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a config from ``env`` (defaults to ``os.environ``)."""
        source = os.environ if env is None else env
        return cls(
            host=_text(source, "JEBA_HOST", "127.0.0.1"),
            port=_int(source, "JEBA_PORT", 8000, 1, 65535),
            device=_choice(source, "JEBA_DEVICE", "auto", DEVICES),
            backend=_choice(source, "JEBA_BACKEND", DEFAULT_BACKEND, BACKENDS),
            models=_csv(source, "JEBA_MODELS"),
            models_dir=_text(source, "JEBA_MODELS_DIR", _default_models_dir()),
            preload=_csv(source, "JEBA_PRELOAD"),
            threads=_int(source, "JEBA_THREADS", 0, 0, 4096),
            api_key=_text(source, "JEBA_API_KEY", "") or None,
            llm_base_url=_text(source, "JEBA_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=(
                _text(source, "JEBA_LLM_API_KEY", "") or _text(source, "OPENAI_API_KEY", "") or None
            ),
            llm_model=_text(source, "JEBA_LLM_MODEL", "gpt-4o-mini"),
            llm_timeout=_float(source, "JEBA_LLM_TIMEOUT", 30.0, lo=0.1),
            llm_retries=_int(source, "JEBA_LLM_RETRIES", 2, 0, 10),
        )


__all__ = ["BACKENDS", "DEFAULT_BACKEND", "DEVICES", "Config"]
