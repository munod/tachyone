"""M0 skeleton smoke tests: package imports and entry points resolve offline."""

from __future__ import annotations

import importlib

import pytest

import jeba


def test_version_is_exposed() -> None:
    assert jeba.__version__ == "0.0.1"


@pytest.mark.parametrize(
    "module_name",
    [
        "jeba",
        "jeba.cli",
        "jeba.serve",
        "jeba.mcp",
        "jeba.mcp.server",
        "jeba.backends",
        "jeba.integrations",
    ],
)
def test_modules_import_without_optional_deps(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    ("module_name", "func_name"),
    [
        ("jeba.cli", "main"),
        ("jeba.serve", "main"),
        ("jeba.mcp.server", "main"),
    ],
)
def test_entry_points_are_callable(module_name: str, func_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))
