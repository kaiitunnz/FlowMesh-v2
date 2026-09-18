"""Every worker module imports on its own.

A package that is only ever reached through one entry point hides an import cycle: the
suite imports `worker.runner` first, which happens to initialize the packages further
down, so a module that cannot be imported directly still looks fine everywhere. The
cycle then surfaces wherever something imports the other module first.
"""

import importlib
import subprocess  # nosec B404 - each import runs in its own interpreter, argv list
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_MODULES = [
    "worker.lifecycle",
    "worker.runner",
    "worker.egress",
    "worker.egress.sidecar",
    "worker.executors",
    "worker.executors.base_executor",
    "worker.telemetry.otel",
    "worker.resident",
    "worker.sandbox.agent_runtime",
]


@pytest.mark.parametrize("module", _MODULES)
def test_the_module_imports_first(module: str) -> None:
    """Import it in a fresh interpreter, so nothing else has primed sys.modules."""
    result = subprocess.run(  # nosec B603 - argv list, no shell, interpreter by path
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        cwd=_SRC.parent,
        env={"PYTHONPATH": _SRC.as_posix(), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert (
        result.returncode == 0
    ), f"importing {module} first fails:\n{result.stderr.strip()}"


def test_importing_one_module_does_not_depend_on_import_order() -> None:
    """The in-process check, for the modules this suite already loaded."""
    for module in _MODULES:
        importlib.import_module(module)
