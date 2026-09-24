"""M0/M2 skeleton smoke tests: package imports and entry points resolve offline.

Core modules must import without any optional extra (NFR-R01). The ``serve`` module is
covered only when the ``serve`` extra is installed.
"""

from __future__ import annotations

import importlib

import pytest

import jeba

_CORE_MODULES = [
    "jeba",
    "jeba.cli",
    "jeba.config",
    "jeba.primitives",
    "jeba.wire",
    "jeba.backends",
    "jeba.backends.base",
    "jeba.backends.fake",
    "jeba.backends.llm",
    "jeba.integrations",
    "jeba.mcp",
    "jeba.mcp.server",
]


def test_version_is_exposed() -> None:
    assert jeba.__version__ == "0.0.1"


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_core_modules_import_without_optional_deps(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    ("module_name", "func_name"),
    [
        ("jeba.cli", "main"),
        ("jeba.mcp.server", "main"),
    ],
)
def test_core_entry_points_are_callable(module_name: str, func_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


def test_serve_module_requires_extra() -> None:
    pytest.importorskip("fastapi")
    import jeba.serve

    assert callable(jeba.serve.main)
    assert callable(jeba.serve.create_app)
