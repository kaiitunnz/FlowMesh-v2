"""A merged task settles only with an outcome of its own."""

import logging
import threading
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.registries.workflow import PersistedTask, WorkflowSched
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.worker import WorkerCapabilities
from shared.tasks.executor_key import ExecutorKey
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _NoopSecretVault,
    _register,
    _WorkerRegistryStub,
)


def _siblings(model: str = "{source: {identifier: m}}", *names: str) -> str:
    nodes = "".join(f"""
      - name: {name}
        spec:
          taskType: inference
          model: {model}
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [{name}]}}""" for name in names or ("a", "b", "c"))
    return f"""
apiVersion: flowmesh/v1
kind: Workflow
metadata: {{name: siblings}}
spec:
  graph:
    nodes:{nodes}
"""


class _Registry(FakeRegistry):
    """Tracks the durable dispatched-set membership as well."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatched: set[str] = set()

    def commit_transition(
        self,
        workflow_id: str,
        *,
        records: Sequence[PersistedTask] = (),
        dispatched: Sequence[str] = (),
        pending: Sequence[str] = (),
        done: Sequence[str] = (),
        failed: Sequence[str] = (),
        cancelled: Sequence[str] = (),
        sched: WorkflowSched | None = None,
    ) -> None:
        super().commit_transition(
            workflow_id,
            records=records,
            done=done,
            failed=failed,
            cancelled=cancelled,
            sched=sched,
        )
        self.dispatched.update(dispatched)
        self.dispatched.difference_update({*pending, *done, *failed, *cancelled})


class _InterruptRecorder(_WorkerRegistryStub):
    def __init__(self) -> None:
        self.interrupted: list[str] = []

    def publish_interrupt(self, worker: Any, interrupt: Any) -> int:
        self.interrupted.append(interrupt.task_id)
        return 1


def _runtime(
    registry: FakeRegistry, worker_registry: Any = None, reader: Any = None
) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, worker_registry or _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader or make_result_reader(),
        logging.getLogger("task-merge"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _worker(*batching: ExecutorKey) -> Any:
    return SimpleNamespace(
        id="wkr-1",
        node_id="nde-1",
        capabilities=WorkerCapabilities(merge_batching_executors=frozenset(batching)),
    )


_VLLM_WORKER = _worker(ExecutorKey.VLLM)


def _next(runtime: TaskRuntime) -> str:
    task_id = runtime.next_ready(threading.Event(), timeout=0)
    assert task_id is not None
    return task_id


def _stored(runtime: TaskRuntime, task_id: str, value: str) -> dict[str, Any]:
    scope = runtime._tasks[task_id].org_id
    return result_payload(runtime._results, task_id, {"value": value}, scope)


def _value(runtime: TaskRuntime, task_id: str) -> Any:
    envelope = runtime.read_result(task_id)
    return None if envelope is None else envelope.result.model_dump().get("value")


def _merged_success(
    runtime: TaskRuntime, parent: str, *children: str
) -> dict[str, Any]:
    payload = _stored(runtime, parent, parent)
    payload["child_result_references"] = {
        child: _stored(runtime, child, child)["result_reference"] for child in children
    }
    return payload


async def _dispatch_merged(
    runtime: TaskRuntime, payload: str | None = None, dispatch: bool = True
) -> dict[str, str]:
    _, ids = await _register(runtime, payload or _siblings())
    parent = _next(runtime)
    assert parent == ids["a"]
    assert runtime.plan_merge(parent, 8, _VLLM_WORKER) == [ids["b"], ids["c"]]
    if dispatch:
        runtime.mark_dispatched(parent, _VLLM_WORKER)
    return ids


def _assert_returned(runtime: TaskRuntime, registry: _Registry, task_id: str) -> None:
    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.merged_parent_id is None
    assert record.merge_key is None
    assert runtime.result_binding(task_id) is None
    assert task_id in runtime._ready_index
    assert task_id not in registry.dispatched


@pytest.mark.anyio
async def test_a_merged_child_without_its_own_result_runs_again_on_its_own() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    a, b, c = ids["a"], ids["b"], ids["c"]
    payload = _merged_success(runtime, a, b)

    assert runtime.merged_children_settled_by(a, payload) == [b]
    runtime.mark_succeeded(a, "wkr-1", payload, _TS)

    assert (_value(runtime, a), _value(runtime, b)) == (a, b)
    assert runtime._tasks[b].status == TaskStatus.DONE
    _assert_returned(runtime, registry, c)
    assert runtime.plan_merge(c, 8, _VLLM_WORKER) == []


@pytest.mark.anyio
async def test_a_failed_parent_returns_its_merged_children_to_the_queue() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)

    impacted, _ = runtime.mark_failed(ids["a"], "wkr-1", {}, _TS, error="bad input")

    assert impacted == []
    assert runtime._tasks[ids["a"]].status == TaskStatus.FAILED
    for child in (ids["b"], ids["c"]):
        _assert_returned(runtime, registry, child)
        assert runtime._tasks[child].error is None


@pytest.mark.anyio
async def test_a_released_merge_leaves_the_durable_dispatched_set() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    assert {ids["b"], ids["c"]} <= registry.dispatched

    runtime.release_merge(ids["a"])

    assert not {ids["b"], ids["c"]} & registry.dispatched
    assert runtime._tasks[ids["b"]].merge_key is not None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("batching", "model"),
    [
        ((), "{source: {identifier: m}}"),
        ((ExecutorKey.VLLM_LORA,), "{source: {identifier: m}}"),
        ((ExecutorKey.VLLM,), "{source: {identifier: m}, transformers: {dtype: auto}}"),
        (
            (ExecutorKey.VLLM, ExecutorKey.DEFAULT),
            "{source: {identifier: m}, adapters: [{type: lora, path: /lora}]}",
        ),
    ],
)
async def test_a_task_merges_only_onto_a_worker_that_batches_its_executor(
    batching: tuple[ExecutorKey, ...], model: str
) -> None:
    runtime = _runtime(_Registry())
    _, ids = await _register(runtime, _siblings(model))

    assert runtime.plan_merge(_next(runtime), 8, _worker(*batching)) == []
    assert runtime._tasks[ids["b"]].status == TaskStatus.PENDING


@pytest.mark.anyio
async def test_lora_siblings_merge_onto_a_worker_that_batches_lora() -> None:
    runtime = _runtime(_Registry())
    model = "{source: {identifier: m}, adapters: [{type: lora, path: /lora}]}"
    _, ids = await _register(runtime, _siblings(model))
    parent = _next(runtime)

    merged = runtime.plan_merge(parent, 8, _worker(ExecutorKey.VLLM_LORA))

    assert merged == [ids["b"], ids["c"]]


@pytest.mark.anyio
async def test_a_template_that_leaves_its_executor_undecided_never_merges() -> None:
    runtime = _runtime(_Registry())
    _, ids = await _register(
        runtime,
        _siblings("{source: {identifier: m}}").replace(
            "taskType: inference", "taskType: inference\n          enforce_cpu: ${cpu}"
        ),
    )

    assert all(runtime._tasks[task].merge_key is None for task in ids.values())


@pytest.mark.anyio
async def test_a_restart_keeps_an_in_flight_merge() -> None:
    registry = _Registry()
    reader = make_result_reader()
    runtime = _runtime(registry, reader=reader)
    ids = await _dispatch_merged(runtime)
    a, b, c = ids["a"], ids["b"], ids["c"]
    payload = _merged_success(runtime, a, b, c)

    restored = _runtime(registry, reader=make_result_reader(reader.store))
    await restored.rehydrate()
    restored.mark_succeeded(a, "wkr-1", payload, _TS)

    for task_id in (a, b, c):
        assert restored._tasks[task_id].status == TaskStatus.DONE
        assert _value(restored, task_id) == task_id


@pytest.mark.anyio
@pytest.mark.parametrize("parent_status", [TaskStatus.PENDING, TaskStatus.FAILED])
async def test_a_restart_returns_the_children_of_a_parent_no_longer_running(
    parent_status: TaskStatus,
) -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    parent = runtime._tasks[ids["a"]]
    if parent_status is not TaskStatus.PENDING:
        parent.status = parent_status
        runtime._persist_locked(ids["a"])

    restored = _runtime(registry)
    await restored.rehydrate()

    assert restored._tasks[ids["a"]].merged_children is None
    for child in (ids["b"], ids["c"]):
        record = restored._tasks[child]
        assert record.status == TaskStatus.PENDING
        assert record.merged_parent_id is None
        assert child in restored._ready_index
        assert child not in registry.dispatched


@pytest.mark.anyio
async def test_cancelling_a_merged_batch_cancels_each_of_its_tasks() -> None:
    interrupts = _InterruptRecorder()
    runtime = _runtime(_Registry(), interrupts)
    ids = await _dispatch_merged(runtime)
    workflow_id = runtime._tasks[ids["a"]].workflow_id

    runtime.cancel_workflow(workflow_id)

    assert interrupts.interrupted == [ids["a"]]
    assert runtime._tasks[ids["a"]].status == TaskStatus.CANCELLING
    runtime.mark_succeeded(
        ids["a"], "wkr-1", _merged_success(runtime, *ids.values()), _TS
    )
    for task_id in ids.values():
        assert runtime._tasks[task_id].status == TaskStatus.CANCELLED


@pytest.mark.anyio
async def test_a_cancelled_merge_parent_returns_another_workflows_children() -> None:
    registry = _Registry()
    runtime = _runtime(registry, _InterruptRecorder())
    first, _ = await _register(runtime, _siblings())
    other, other_ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    merged = runtime.plan_merge(parent, 8, _VLLM_WORKER)
    runtime.mark_dispatched(parent, _VLLM_WORKER)
    assert set(other_ids.values()) <= set(merged)

    runtime.cancel_workflow(first)
    success = _merged_success(runtime, parent, *merged)
    assert runtime.merged_children_settled_by(parent, success) == []
    runtime.mark_succeeded(parent, "wkr-1", success, _TS)

    for child in other_ids.values():
        _assert_returned(runtime, registry, child)


@pytest.mark.anyio
async def test_a_merged_child_of_a_cancelled_workflow_stays_cancelled() -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    first, first_ids = await _register(runtime, _siblings())
    other, other_ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    merged = runtime.plan_merge(parent, 8, _VLLM_WORKER)
    runtime.mark_dispatched(parent, _VLLM_WORKER)

    runtime.cancel_workflow(other)
    runtime.mark_succeeded(
        parent, "wkr-1", _merged_success(runtime, parent, *merged), _TS
    )

    for task_id in other_ids.values():
        assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    for task_id in first_ids.values():
        assert runtime._tasks[task_id].status == TaskStatus.DONE
