"""Owner-affine dispatch of an episode bound to activation-private state."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

from server.config import OrchestrationConfig
from server.registries.worker import Worker
from server.task.runtime import TaskRuntime
from shared.private_state import OwnerFence, PrivateStateUnavailableReason
from tests.server.dispatcher.helpers import CapturingDispatcher, WorkflowRegistryStub
from tests.server.task.test_v2_orchestration import _NoopSecretVault

_OWNER = OwnerFence(worker_id="wkr-owner", incarnation=7)

_ECHO_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: private-state-affinity
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""


def _dispatcher(
    *, idle_ids: list[str], registered: Worker | None, stale: bool = False
) -> tuple[CapturingDispatcher, str]:
    runtime = TaskRuntime(
        cast(Any, WorkflowRegistryStub()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("private-state-affinity-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )
    _, results = asyncio.run(
        runtime.register("owner", "org", _ECHO_WORKFLOW, format="native")
    )
    task_id = results[0].task_id
    runtime.private_state_owner = mock.Mock(return_value=_OWNER)  # type: ignore[method-assign]
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker(wid) for wid in idle_ids]
    registry.get_worker.return_value = registered
    registry.is_worker_stale.return_value = stale
    dispatcher = CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        results_dir=Path(tempfile.gettempdir()),
        logger=logging.getLogger("private-state-affinity-test"),
        no_worker_grace_sec=0,
    )
    return dispatcher, task_id


def _worker(worker_id: str, incarnation: int = _OWNER.incarnation) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="cluster",
        node_id="nde-1",
        node_alias="node",
        incarnation=incarnation,
    )


def _live_owner() -> Worker:
    return _worker(_OWNER.worker_id)


def test_a_busy_owner_defers_the_episode_without_holding_a_worker() -> None:
    dispatcher, task_id = _dispatcher(idle_ids=["wkr-other"], registered=_live_owner())

    assert dispatcher.dispatch_once(task_id) is False

    assert dispatcher.failed == []
    assert dispatcher.requeued[0][1]["reason"] == "private_state_owner_busy"


def test_only_the_owner_reaches_worker_selection() -> None:
    dispatcher, task_id = _dispatcher(
        idle_ids=["wkr-other", _OWNER.worker_id], registered=_live_owner()
    )
    seen: list[list[str]] = []

    def _capture(pool: list[Worker], *args: Any, **kwargs: Any) -> tuple[None, dict]:
        seen.append([worker.id for worker in pool])
        return None, {}

    with mock.patch("server.dispatcher.base.select_worker", _capture):
        dispatcher.dispatch_once(task_id)

    assert seen == [[_OWNER.worker_id]]
    assert dispatcher.failed == []


def test_a_lost_owner_fails_closed_rather_than_resuming_elsewhere() -> None:
    dispatcher, task_id = _dispatcher(idle_ids=["wkr-other"], registered=None)

    assert dispatcher.dispatch_once(task_id) is False

    task, message, kwargs = dispatcher.failed[0]
    assert task == task_id
    assert PrivateStateUnavailableReason.OWNER_LOST.value in message
    assert kwargs["payload"]["private_state_owner"] == _OWNER.worker_id
    assert dispatcher.requeued == []


def test_a_restarted_owner_incarnation_does_not_satisfy_the_binding() -> None:
    replaced = _worker(_OWNER.worker_id, _OWNER.incarnation + 1)
    dispatcher, task_id = _dispatcher(idle_ids=[_OWNER.worker_id], registered=replaced)

    assert dispatcher.dispatch_once(task_id) is False

    assert PrivateStateUnavailableReason.OWNER_LOST.value in dispatcher.failed[0][1]


def test_a_stale_owner_heartbeat_fails_closed() -> None:
    dispatcher, task_id = _dispatcher(
        idle_ids=[_OWNER.worker_id], registered=_live_owner(), stale=True
    )

    assert dispatcher.dispatch_once(task_id) is False

    assert dispatcher.failed[0][1].startswith("PrivateStateUnavailable")
