"""A cordoned worker drains the agent episodes sealed on it and takes no new work."""

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import fakeredis

from server.registries.worker import Worker, WorkerRegistry
from server.schemas.worker import WorkerCordon
from shared.schemas.worker import WorkerCapabilities, WorkerStatus
from shared.tasks.task_type import TaskType
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.redis_helpers import fake_redis_client
from tests.server.task.test_v2_orchestration import _register
from tests.server.task.test_worker_originated_boundary import (
    _HOLDER,
    _SEARCH_WF,
    _dispatch_agent,
    _report,
    _runtime,
)
from tests.worker.factories import make_worker_hardware

_ECHO_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: fresh}
spec:
  graph:
    nodes:
      - name: gen
        spec:
          taskType: echo
          data: {type: list, items: ["hello"]}
"""


def _registry(*aliases: str) -> WorkerRegistry:
    """Idle workers ``wkr-1``, ``wkr-2``, ... under ``aliases``; ``wkr-1`` holds the
    sealed agent."""
    registry = WorkerRegistry(fake_redis_client(fakeredis.FakeServer()))
    capabilities = WorkerCapabilities(
        supported_task_types=frozenset(
            {TaskType.AGENT, TaskType.ECHO, TaskType.DEV_MODEL}
        )
    )
    for alias in aliases or ("holder", "other"):
        meta = {
            "alias": alias,
            "namespace": "ns",
            "cluster": "c",
            "status": WorkerStatus.IDLE.value,
            "capabilities_json": capabilities.model_dump_json(),
            "hardware_json": make_worker_hardware().model_dump_json(),
        }
        worker_id = registry.register_worker("nde-1", "node", meta)
        registry.update_worker_hb(worker_id, "2026-01-01T00:00:00Z", 60)
    assert registry.get_worker(_HOLDER.worker_id) is not None
    return registry


def _offered(registry: WorkerRegistry, runtime: Any, task_id: str) -> list[list[str]]:
    """The pools the dispatcher selects ``task_id``'s worker from."""
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("cordon-drain-test"),
        no_worker_grace_sec=60,
    )
    seen: list[list[str]] = []

    def _capture(pool: list[Worker], *args: Any, **kwargs: Any) -> tuple[None, dict]:
        seen.append([worker.id for worker in pool])
        return None, {}

    with mock.patch("server.dispatcher.base.select_worker", _capture):
        dispatcher.dispatch_once(task_id)
    assert dispatcher.failed == []
    return seen


def test_a_cordoned_holder_resumes_its_sealed_agent_and_takes_no_new_work(
    tmp_path: Path,
) -> None:
    registry = _registry()
    runtime = _runtime()
    _, ids = asyncio.run(_register(runtime, _SEARCH_WF))
    writer = ids["writer"]
    _dispatch_agent(runtime, writer, seal_in=tmp_path)
    _report(runtime, writer, "m0", "sunny")
    assert runtime.private_state_owner(writer) == _HOLDER
    registry.set_cordon(WorkerCordon(node_alias="node", alias="holder"), True)
    _, fresh = asyncio.run(_register(runtime, _ECHO_WF))

    assert _offered(registry, runtime, writer) == [[_HOLDER.worker_id]]
    assert _offered(registry, runtime, fresh["gen"]) == [["wkr-2"]]


def test_new_work_does_not_wait_on_a_cordoned_worker() -> None:
    registry = _registry()
    registry.set_cordon(WorkerCordon(node_alias="node", alias="holder"), True)
    registry.set_cordon(WorkerCordon(node_alias="node", alias="other"), True)
    runtime = _runtime()
    _, fresh = asyncio.run(_register(runtime, _ECHO_WF))
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("cordon-drain-test"),
        no_worker_grace_sec=0,
    )

    assert dispatcher.dispatch_once(fresh["gen"]) is False

    assert [reason for _, _, reason in dispatcher.failed] == [
        {"worker_id": None, "payload": {"reason": "no_eligible_worker"}}
    ]
