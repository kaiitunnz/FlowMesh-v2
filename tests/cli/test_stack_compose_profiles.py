"""Which compose profiles a node deploys.

Compose's ``--profile`` flag replaces ``COMPOSE_PROFILES`` rather than adding to it, so
a root node passing only its own role profile silently deploys none of the optional
services an operator asked for. Nothing about that failure is visible at deploy time:
the services are simply absent.

A root node also brings up the content store the fabric writes to, unless the deployment
names a store of its own — so a fresh cluster stores content unconfigured, and one
pointed at real object storage runs no store it does not use.
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

    assert _profiles(env_file, "root") == ["root", "telemetry", "content"]


def test_several_operator_profiles_are_all_kept(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "COMPOSE_PROFILES= telemetry , extra \n")

    assert _profiles(env_file, "root") == ["root", "telemetry", "extra", "content"]


def test_a_root_node_brings_up_the_content_store_it_writes_to(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=root\n")

    assert _profiles(env_file, "root") == ["root", "content"]


def test_a_deployment_naming_its_own_store_runs_no_local_one(tmp_path: Path) -> None:
    env_file = _env_file(
        tmp_path,
        "NODE_ROLE=root\nCONTENT_STORE_ENDPOINT_URL=https://s3.amazonaws.com\n",
    )

    assert _profiles(env_file, "root") == ["root"]


def test_a_worker_node_runs_no_content_store_of_its_own(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=worker\n")

    assert _profiles(env_file, None) == []


def test_a_worker_node_still_gets_the_operators_profiles(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "NODE_ROLE=worker\nCOMPOSE_PROFILES=telemetry\n")

    assert _profiles(env_file, None) == ["telemetry"]


def test_a_duplicate_is_not_passed_twice(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, "COMPOSE_PROFILES=root,telemetry\n")

    assert _profiles(env_file, "root") == ["root", "telemetry", "content"]


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
