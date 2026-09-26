"""Tests for release assets: model card, changelog, release process (M6-T3, OPS-07)."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_model_card_has_front_matter_and_sections() -> None:
    text = (_ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "license: apache-2.0" in text
    assert "library_name: tachyone" in text
    for section in ("## Model details", "## Uses", "## Training", "## Evaluation", "## Citation"):
        assert section in text


def test_changelog_follows_keep_a_changelog() -> None:
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Keep a Changelog" in text
    assert "## [Unreleased]" in text
    assert "## [0.0.1]" in text


def test_release_checklist_exists() -> None:
    text = (_ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    assert "uv run pytest" in text
    assert "mkdocs build --strict" in text
    assert "training.finetune_rlcd" in text
    assert "benchmarks/report.md" in text
