"""Tests for the documentation site config (M6-T2, NFR-D01 / NFR-D02).

Static checks always run; the actual ``mkdocs`` build runs only when the ``docs`` group is
installed (CI docs job).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MKDOCS = _ROOT / "mkdocs.yml"


def _nav_docs() -> list[str]:
    text = _MKDOCS.read_text(encoding="utf-8")
    return re.findall(r":\s*([\w./-]+\.md)\s*$", text, flags=re.MULTILINE)


def test_mkdocs_config_and_nav_resolve() -> None:
    assert _MKDOCS.exists()
    docs = _nav_docs()
    assert "index.md" in docs
    assert "protocol.md" in docs
    for relative in docs:
        assert (_ROOT / "docs" / relative).exists(), f"missing docs page: {relative}"


def test_docs_index_and_requirements() -> None:
    assert (_ROOT / "docs" / "index.md").exists()
    requirements = (_ROOT / "requirements-docs.txt").read_text(encoding="utf-8")
    assert "mkdocs" in requirements
    assert "mkdocs-material" in requirements


def test_site_builds(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    pytest.importorskip("material")
    pytest.importorskip("cairosvg")  # social cards need the mkdocs-material[imaging] extra
    from mkdocs.commands.build import build  # pyright: ignore[reportMissingImports]
    from mkdocs.config import load_config  # pyright: ignore[reportMissingImports]

    config = load_config(config_file=str(_MKDOCS))
    config["site_dir"] = str(tmp_path / "site")
    build(config)
    assert (tmp_path / "site" / "index.html").exists()


def test_brand_assets_exist() -> None:
    assert (_ROOT / "docs" / "assets" / "logo.svg").exists()
    assert (_ROOT / "docs" / "assets" / "favicon.svg").exists()
    assert (_ROOT / "docs" / "stylesheets" / "extra.css").exists()
