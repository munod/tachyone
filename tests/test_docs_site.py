"""Tests for the documentation site config (M6-T2, NFR-D01 / NFR-D02).

Static checks always run; the actual ``mkdocs`` build runs only when the ``docs`` group is
installed (CI docs job).
"""

from __future__ import annotations

import re
import subprocess
import sys
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

    site_dir = tmp_path / "site"
    # Use the same CLI entry point as CI (`mkdocs build`); it materializes plugins and runs the
    # `config`/`pre_build` hooks that the social plugin needs (load_config alone leaves the
    # plugin collection unresolved and trips "'SocialPlugin' object has no attribute 'card_pool'").
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "-f",
            str(_MKDOCS),
            "-d",
            str(site_dir),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (site_dir / "index.html").exists()


def test_brand_assets_exist() -> None:
    assert (_ROOT / "docs" / "assets" / "logo.svg").exists()
    assert (_ROOT / "docs" / "assets" / "logo-wordmark.svg").exists()
    assert (_ROOT / "docs" / "stylesheets" / "extra.css").exists()
