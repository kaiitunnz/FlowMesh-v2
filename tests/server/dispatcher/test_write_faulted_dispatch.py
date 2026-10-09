"""A workflow whose write raised a fault of its own publishes nothing until a write of
it is made, without holding up the dispatch of any other workflow."""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.dispatcher.base import Dispatcher
from server.registries.worker import Worker
from server.task.models import PublishGate
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
from tests.server.task.test_task_merge import _siblings
from tests.server.task.test_v2_orchestration import FakeRegistry
from tests.support.waiting import pop_ready

_WORKFLOW = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: beside
spec:
  graph:
    nodes:
      - {name: a, spec: {taskType: echo}}
      - {name: b, dependsOn: [a], spec: {taskType: echo}}
      - {name: c, spec: {taskType: echo}}
"""

_WORKER = Worker(
    id="wkr-1",
    namespace="ns",
    cluster="c",
    node_id="nde-1",
    node_alias="node",
    incarnation=1,
)


class _FaultyStore(FakeRegistry):
    """A store whose writes of one workflow raise a fault of their own while
    ``faults`` lasts."""

    def __init__(self) -> None:
        super().__init__()
        self.faulty = ""
        self.faults = 0

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        if workflow_id == self.faulty and self.faults:
            self.faults -= 1
            raise TypeError("unserializable record")
        super().commit_transition(workflow_id, **kwargs)


def _workers() -> mock.Mock:
    workers = mock.Mock()
    workers.idle_satisfying_pool.return_value = [_WORKER]
    workers.reserve_worker.return_value = True
    workers.publish_task.return_value = 1
    return workers


def _register(runtime: TaskRuntime) -> tuple[str, dict[str, str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _WORKFLOW, format="native")
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


def _published(workers: mock.Mock) -> list[str]:
    return [call.args[1].task_id for call in workers.publish_task.call_args_list]


def test_a_write_faulted_workflow_waits_aside_while_others_dispatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FaultyStore()
    runtime = TaskRuntime(
        cast(Any, store),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("write-faulted"),
        credential_vault=InMemoryCredentialVault(),
    )
    faulted, ids = _register(runtime)
    assert {pop_ready(runtime), pop_ready(runtime)} == {ids["a"], ids["c"]}
    record_dispatch(runtime, ids["a"])
    store.faulty, store.faults = faulted, 3
    with pytest.raises(TypeError):
        runtime.mark_failed(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    _, other = _register(runtime)

    workers = _workers()
    dispatcher = Dispatcher(runtime, workers, logging.getLogger("write-faulted"))
    caplog.clear()
    order = [ids["c"], pop_ready(runtime), pop_ready(runtime)]
    # A True answer is one the dispatch loop takes without backing off.
    assert [dispatcher.dispatch_once(cast(str, t)) for t in order] == [True] * 3
    assert sorted(_published(workers)) == sorted([other["a"], other["c"]])
    assert ids["c"] not in runtime._ready.ready_index
    assert pop_ready(runtime) is None

    # A later write that raises again leaves it waiting, and logs nothing more.
    with pytest.raises(TypeError):
        runtime.mark_failed(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert pop_ready(runtime) is None
    assert not any(r.exc_info for r in caplog.records)
    assert not [r for r in caplog.records if "next write" in r.getMessage()]

    # A write of the workflow that is made releases what waited.
    runtime.mark_failed(ids["a"], "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert pop_ready(runtime) == ids["c"]
    assert dispatcher.dispatch_once(ids["c"]) is True
    assert _published(workers)[-1] == ids["c"]


def _siblings_runtime() -> tuple[_FaultyStore, TaskRuntime]:
    store = _FaultyStore()
    runtime = TaskRuntime(
        cast(Any, store),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("write-faulted"),
        credential_vault=InMemoryCredentialVault(),
    )
    return store, runtime


def _register_siblings(
    runtime: TaskRuntime, names: tuple[str, ...]
) -> tuple[str, dict[str, str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", _siblings(names=names), format="native")
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


def _fault(store: _FaultyStore, runtime: TaskRuntime, workflow_id: str) -> None:
    store.faulty, store.faults = workflow_id, 10_000
    with runtime._cv:
        runtime._committer.mark_dirty_locked(workflow_id)
        with pytest.raises(TypeError):
            runtime._committer.close_locked(workflow_id)


def test_a_healthy_task_merges_no_task_of_a_faulted_workflow() -> None:
    store, runtime = _siblings_runtime()
    _, healthy = _register_siblings(runtime, ("b1", "b2"))
    faulted, held = _register_siblings(runtime, ("a1", "a2", "a3"))
    _fault(store, runtime, faulted)

    assert pop_ready(runtime) == healthy["b1"]
    merged = runtime.plan_merge(healthy["b1"], 4, "wkr-1")
    assert merged == [healthy["b2"]]
    record_dispatch(runtime, healthy["b1"])
    assert all(task_id in runtime._ready.ready_index for task_id in held.values())


def test_a_task_whose_merged_child_faulted_runs_alone() -> None:
    store, runtime = _siblings_runtime()
    _, healthy = _register_siblings(runtime, ("b1",))
    faulted, held = _register_siblings(runtime, ("a1", "a2"))

    assert pop_ready(runtime) == healthy["b1"]
    merged = runtime.plan_merge(healthy["b1"], 4, "wkr-1")
    assert sorted(merged) == sorted(held.values())
    _fault(store, runtime, faulted)

    gate = runtime.begin_publish(healthy["b1"], cast(Any, _WORKER), None)
    assert gate is PublishGate.NOT_DURABLE
    assert runtime._write_faulted == {}
    record = runtime.get_record(healthy["b1"])
    assert record is not None and not record.merged_children
