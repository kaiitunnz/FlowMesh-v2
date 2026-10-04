"""The runtime groups' grpcio and protobuf floors admit the generated proto stubs.

A stub refuses to import on a runtime older than the one it was generated for, so
every group that imports the stubs declares at least that version.
"""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[2]
_STUBS = _ROOT / "src" / "shared" / "grpc" / "supervisor" / "v1"
_GROUPS = ("runtime-server", "runtime-worker-core")


def _generated_versions() -> dict[str, Version]:
    grpc_stub = (_STUBS / "supervisor_pb2_grpc.py").read_text()
    pb2_stub = (_STUBS / "supervisor_pb2.py").read_text()
    grpc_version = re.search(r'GRPC_GENERATED_VERSION = "([^"]+)"', grpc_stub)
    pb2_version = re.search(r"Domain\.PUBLIC,\s*(\d+),\s*(\d+),\s*(\d+)", pb2_stub)
    assert grpc_version is not None and pb2_version is not None
    return {
        "grpcio": Version(grpc_version.group(1)),
        "protobuf": Version(".".join(pb2_version.groups())),
    }


def _floor(group: str, package: str) -> Version:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    for entry in pyproject["dependency-groups"][group]:
        if isinstance(entry, str) and (req := Requirement(entry)).name == package:
            floors = [s.version for s in req.specifier if s.operator == ">="]
            assert floors, f"{group} declares no floor for {package}"
            return Version(floors[0])
    raise AssertionError(f"{group} does not declare {package}")


@pytest.mark.parametrize("group", _GROUPS)
@pytest.mark.parametrize("package", ["grpcio", "protobuf"])
def test_a_runtime_floor_admits_the_generated_stubs(group: str, package: str) -> None:
    assert _floor(group, package) >= _generated_versions()[package]
