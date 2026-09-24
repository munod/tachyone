"""Static checks for the Docker assets (M5-T5, OPS-04).

A real container build is environment-dependent; these assertions keep the image and compose
file honest without requiring a Docker daemon in CI.
"""

from __future__ import annotations

from pathlib import Path

_DOCKER = Path(__file__).resolve().parents[1] / "docker"


def test_dockerfile_uses_python_and_serve_extra() -> None:
    text = (_DOCKER / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12" in text
    assert "uv sync --locked --no-dev --extra serve" in text
    assert "jeba-serve" in text
    assert "USER jeba" in text


def test_compose_defines_the_service() -> None:
    text = (_DOCKER / "compose.yaml").read_text(encoding="utf-8")
    assert "jeba:" in text
    assert "context: .." in text
    assert "dockerfile: docker/Dockerfile" in text
    assert "/health" in text
    assert "JEBA_BACKEND" in text


def test_dockerignore_excludes_heavy_paths() -> None:
    text = (_DOCKER / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".venv", ".opencode", "training", "*.safetensors"):
        assert entry in text
