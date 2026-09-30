"""A field a task spec retired is refused at submission, while a task stored with it
still loads, dispatches and reads without it."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
from lumid_hooks import PrincipalContext

from server.config import OrchestrationConfig
from server.registries.workflow import WorkflowRegistry
from server.routers.v1 import tasks as tasks_router
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.dispatcher.test_result_availability import _worker
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import FakeRegistry

# A task record as the previous SSH spec wrote it, carrying `spec.mounts`.
_STORED = json.loads(
    (Path(__file__).parent / "fixtures" / "ssh_task_with_mounts.json").read_text()
)

_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: ssh-mounts
spec:
  stages:
    - name: shell
      spec:
        taskType: ssh
        image: busybox:1.36.1
        command: ["true"]
"""


def _runtime(registry: Any) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("retired-fields"),
        credential_vault=InMemoryCredentialVault(),
    )


def _stored_blob(task_id: str, workflow_id: str) -> str:
    stored = json.loads(json.dumps(_STORED))
    stored["record"]["task_id"] = task_id
    stored["record"]["workflow_id"] = workflow_id
    return json.dumps(stored)


def test_a_submission_naming_mounts_is_rejected() -> None:
    runtime = _runtime(FakeRegistry())
    submitted = _WORKFLOW + "        mounts: [{name: scratch, mode: rw}]\n"

    with pytest.raises(ValueError, match="mounts"):
        asyncio.run(runtime.register("owner", "org", submitted, format="native"))


def test_the_registry_loads_a_stored_task_carrying_mounts() -> None:
    blob = json.dumps(_STORED)
    registry = WorkflowRegistry.__new__(WorkflowRegistry)
    registry._rds = mock.Mock()
    registry._rds.sync.mget.return_value = [blob]

    (loaded,) = registry.load_task_states(_STORED["record"]["task_id"])

    assert loaded is not None
    assert "mounts" not in loaded.record.task.spec.model_dump()


def test_a_rehydrated_task_carrying_mounts_dispatches_and_reads_without_them() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )
    task_id = results[0].task_id
    registry.task_blobs[task_id] = _stored_blob(task_id, workflow_id)

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1

    worker_registry = mock.Mock()
    worker_registry.idle_satisfying_pool.return_value = [_worker()]
    worker_registry.satisfying_workers.return_value = [_worker()]
    worker_registry.publish_task.return_value = 1
    worker_registry.get_worker.return_value = _worker()
    dispatcher = CapturingDispatcher(
        runtime=restored,
        worker_registry=worker_registry,
        logger=logging.getLogger("dispatch"),
    )
    dispatcher.dispatch_once(task_id)

    (published,) = worker_registry.publish_task.call_args_list
    assert "mounts=" not in str(published)

    info = asyncio.run(
        tasks_router.get_task(
            task_id,
            PrincipalContext(
                principal_id="owner",
                org_id="org",
                external_id="owner",
                principal_type="user",
                scopes=[],
            ),
            restored,
            logging.getLogger("tasks-router"),
        )
    )
    assert info.task_id == task_id
    assert '"command"' in info.model_dump_json()
    assert '"mounts"' not in info.model_dump_json()
