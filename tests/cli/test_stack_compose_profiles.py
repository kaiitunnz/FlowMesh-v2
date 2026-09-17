"""A node's own compose profile never discards the operator's.

Compose's ``--profile`` flag replaces ``COMPOSE_PROFILES`` rather than adding to it, so
a root node passing only its own role profile silently deploys none of the optional
services an operator asked for. Nothing about that failure is visible at deploy time:
the services are simply absent.
"""

from pathlib import Path

from flowmesh_cli_stack.stack import _profiles
from flowmesh_stack.docker import profile_args


def _env_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(body)
    return path


def test_the_operators_profiles_join_the_nodes_own(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=root\nCOMPOSE_PROFILES=telemetry\n")

    assert _profiles(env_file, "root") == ["root", "telemetry"]


def test_several_operator_profiles_are_all_kept(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "COMPOSE_PROFILES= telemetry , extra \n")

    assert _profiles(env_file, "root") == ["root", "telemetry", "extra"]


def test_a_node_with_no_operator_profiles_passes_only_its_own(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=root\n")

    assert _profiles(env_file, "root") == ["root"]


def test_a_worker_node_still_gets_the_operators_profiles(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=worker\nCOMPOSE_PROFILES=telemetry\n")

    assert _profiles(env_file, None) == ["telemetry"]


def test_a_duplicate_is_not_passed_twice(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "COMPOSE_PROFILES=root,telemetry\n")

    assert _profiles(env_file, "root") == ["root", "telemetry"]


def test_each_profile_becomes_its_own_flag() -> None:
    # One flag carrying a comma-joined list would be read as a single profile name.
    assert profile_args(["root", "telemetry"]) == [
        "--profile",
        "root",
        "--profile",
        "telemetry",
    ]
    assert profile_args("root") == ["--profile", "root"]
    assert profile_args(None) == []
