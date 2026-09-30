"""An SSH task's mount paths stay under the session mount root."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_live_worker_config, make_ssh_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import DockerSessionBackend, normalize_mount_path
from worker.executors.ssh_session.config import (
    MAX_MOUNT_PATH_CHARS,
    MAX_MOUNT_PATH_COMPONENTS,
)

_ESCAPES = ["/mnt/flowmesh/../../etc", "/mnt/flowmesh/inputs/../../../etc/cron.d"]


def _task(**spec: Any) -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": "tsk-mount",
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "assigned_worker": "worker-1",
            "dispatched_at": "2026-03-22T00:00:00Z",
            "upstream_task_ids": {"up": "tsk-up"},
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "Task",
                "metadata": {"name": "wf:s"},
                "spec": {
                    "taskType": "ssh",
                    "authorizedKeys": ["ssh-ed25519 AAAA test"],
                    **spec,
                },
            },
        }
    )


@pytest.mark.parametrize("path", _ESCAPES)
@pytest.mark.parametrize("field", ["inputs", "sshOutput"])
def test_a_mount_path_climbing_out_of_the_mount_root_fails_the_task(
    tmp_path: Path, path: str, field: str
) -> None:
    executor = make_ssh_executor(make_live_worker_config(tmp_path), lifecycle=None)
    backend = executor.backend
    assert isinstance(backend, DockerSessionBackend)
    backend._docker = MagicMock()
    spec: dict[str, Any] = (
        {"inputs": [{"stage": "up", "mountPath": path}]}
        if field == "inputs"
        else {"sshOutput": {"mountPath": path}}
    )

    with (
        patch.object(backend, "prepare"),
        pytest.raises(ExecutionError, match="must not contain '..'"),
    ):
        executor.run(_task(**spec), tmp_path / "out")

    backend._docker.containers.create.assert_not_called()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/mnt/flowmesh//inputs/./a", "/mnt/flowmesh/inputs/a"),
        ("  /mnt/flowmesh/output/  ", "/mnt/flowmesh/output"),
        ("//mnt/flowmesh/x", "/mnt/flowmesh/x"),
    ],
)
def test_a_mount_path_is_normalized(path: str, expected: str) -> None:
    assert normalize_mount_path(path, "mountPath") == expected


@pytest.mark.parametrize("path", ["/etc", "/mnt/flowmeshx", "mnt/flowmesh/x", "/mnt"])
def test_a_mount_path_outside_the_root_is_rejected(path: str) -> None:
    with pytest.raises(ExecutionError):
        normalize_mount_path(path, "mountPath")


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/flowmesh/" + "/".join(["d"] * (MAX_MOUNT_PATH_COMPONENTS - 1)),
        "/mnt/flowmesh/" + "x" * (MAX_MOUNT_PATH_CHARS - len("/mnt/flowmesh/") + 1),
    ],
)
def test_a_mount_path_too_deep_or_long_is_rejected(path: str) -> None:
    with pytest.raises(ExecutionError, match="at most"):
        normalize_mount_path(path, "mountPath")


def test_a_mount_path_at_the_bounds_is_accepted() -> None:
    deepest = "/mnt/flowmesh/" + "/".join(["d"] * (MAX_MOUNT_PATH_COMPONENTS - 2))
    longest = "/mnt/flowmesh/" + "x" * (MAX_MOUNT_PATH_CHARS - len("/mnt/flowmesh/"))
    for path in (deepest, longest):
        assert normalize_mount_path(path, "mountPath") == path
