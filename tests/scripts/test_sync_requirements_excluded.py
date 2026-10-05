"""The packages each image resolves for itself are picked out of its compiled set."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "dev" / "sync_requirements.py"
)


@pytest.fixture(scope="module")
def sync_requirements() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_requirements", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMPILED = """\
--index-url https://pypi.org/simple
fastapi==0.120.0
nvidia-ml-py==13.580.82
nvidia-cublas==12.9.1.4
setuptools==84.0.0
torch==2.13.0+cpu
torchcodec==0.9.0
"""


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("server", ["setuptools==84.0.0"]),
        (
            "worker",
            [
                "nvidia-cublas==12.9.1.4",
                "setuptools==84.0.0",
                "torchcodec==0.9.0",
            ],
        ),
    ],
)
def test_only_the_packages_the_constraints_leave_are_picked(
    sync_requirements: ModuleType, image: str, expected: list[str]
) -> None:
    assert sync_requirements.excluded_pins(image, _COMPILED) == expected
