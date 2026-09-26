"""M0/M2 skeleton smoke tests: package imports and entry points resolve offline.

Core modules must import without any optional extra (NFR-R01). The ``serve`` module is
covered only when the ``serve`` extra is installed.
"""

from __future__ import annotations

import importlib

import pytest

import tachyone

_CORE_MODULES = [
    "tachyone",
    "tachyone.calibration",
    "tachyone.cli",
    "tachyone.config",
    "tachyone.fast",
    "tachyone.handoff",
    "tachyone.primitives",
    "tachyone.telemetry",
    "tachyone.wire",
    "tachyone.backends",
    "tachyone.backends.base",
    "tachyone.backends.fake",
    "tachyone.backends.llm",
    "tachyone.backends.onnx",
    "tachyone.integrations",
    "tachyone.integrations.langchain",
    "tachyone.mcp",
    "tachyone.mcp.server",
]


def test_version_is_exposed() -> None:
    assert tachyone.__version__ == "0.3.0"


def test_handoff_helpers_are_public() -> None:
    for name in ("assess", "assess_response", "HandoffReport", "HandoffSignal", "Uncertainty"):
        assert callable(getattr(tachyone, name)), name


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_core_modules_import_without_optional_deps(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    ("module_name", "func_name"),
    [
        ("tachyone.cli", "main"),
        ("tachyone.mcp.server", "main"),
    ],
)
def test_core_entry_points_are_callable(module_name: str, func_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


def test_serve_module_requires_extra() -> None:
    pytest.importorskip("fastapi")
    import tachyone.serve

    assert callable(tachyone.serve.main)
    assert callable(tachyone.serve.create_app)
