"""A merged task settles only with an outcome of its own."""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.config import OrchestrationConfig
from server.dispatcher.base import Dispatcher
from server.registries.workflow import PersistedTask, WorkflowSched
from server.services.monitoring import EventMonitor
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.event import TaskEvent, WorkerEvent
from shared.tasks.specs.common import ConditionSpec
from tests.server.result_store import make_result_reader, result_payload, store_result
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _NoopSecretVault,
    _WorkerRegistryStub,
)

_MODEL = "{source: {identifier: m}}"
_LORA = "{source: {identifier: m}, adapters: [{type: lora, path: /lora}]}"


def _siblings(
    model: str = _MODEL,
    names: Sequence[str] = ("a", "b", "c"),
    epochs: Sequence[Sequence[str]] = (),
) -> str:
    """A v1 workflow of inference tasks whose specs differ only in their prompt."""
    nodes = "".join(f"""
      - name: {name}
        spec:
          taskType: inference
          model: {model}
          resources: {{hardware: {{gpu: {{count: 1}}}}}}
          data: {{type: list, items: [{name}]}}""" for name in names)
    order = "".join(f"\n        - [{', '.join(epoch)}]" for epoch in epochs)
    hint = (
        f"\n  annotations:\n    schedule_hint:\n      node_execution_order:{order}"
        if epochs
        else ""
    )
    return f"""
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: siblings{hint}
spec:
  graph:
    nodes:{nodes}
"""


class _Registry(FakeRegistry):
    """Tracks each workflow's durable dispatched set, and can fail its next commit."""

    def __init__(self) -> None:
        super().__init__()
        self.dispatched: dict[str, set[str]] = defaultdict(set)
        self.fail_next = False
        self.down = False

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
        if self.down or self.fail_next:
            self.fail_next = False
            raise ConnectionError("control redis unavailable")
        super().commit_transition(
            workflow_id,
            records=records,
            done=done,
            failed=failed,
            cancelled=cancelled,
            sched=sched,
        )
        self.dispatched[workflow_id].update(dispatched)
        self.dispatched[workflow_id].difference_update(
            {*pending, *done, *failed, *cancelled}
        )

    def is_dispatched(self, task_id: str) -> bool:
        return any(task_id in ids for ids in self.dispatched.values())

    def durable_status(self, task_id: str) -> str:
        return PersistedTask.model_validate_json(self.task_blobs[task_id]).record.status


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


def _worker(worker_id: str = "wkr-1") -> Any:
    return SimpleNamespace(id=worker_id, node_id="nde-1")


_WORKER = _worker()


async def _register(
    runtime: TaskRuntime, payload: str, org: str = "org"
) -> tuple[str, dict[str, str]]:
    workflow_id, results = await runtime.register(
        "owner", org, payload, format="native"
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


def _next(runtime: TaskRuntime) -> str | None:
    """The next ready task, or None when nothing is ready."""
    with runtime._cv:
        return runtime._pop_ready_locked()


def _value(runtime: TaskRuntime, task_id: str) -> Any:
    envelope = runtime.read_result(task_id)
    return None if envelope is None else envelope.result.model_dump().get("value")


def _merged_success(
    runtime: TaskRuntime, parent: str, *children: str
) -> dict[str, Any]:
    """A merged dispatch's success, with every result stored under the parent's scope
    as the worker stores it."""
    scope = runtime._tasks[parent].org_id
    payload = result_payload(runtime._results, parent, {"value": parent}, scope)
    payload["child_result_references"] = {
        child: store_result(
            runtime._results, child, {"value": child}, scope
        ).model_dump(mode="json")
        for child in children
    }
    return payload


async def _dispatch_merged(
    runtime: TaskRuntime, dispatch: bool = True
) -> dict[str, str]:
    _, ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    assert parent == ids["a"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [ids["b"], ids["c"]]
    if dispatch:
        runtime.mark_dispatched(parent, _WORKER)
    return ids


def _render(
    runtime: TaskRuntime,
    parent: str,
    resolve: Callable[[str, Any, Any], Any],
    condition_actual: Callable[[Any, Any], Any] | None = None,
) -> list[str]:
    """Render a parent's merged children as the dispatcher does before publishing."""
    dispatcher = cast(
        Dispatcher,
        SimpleNamespace(
            _runtime=runtime,
            _logger=logging.getLogger("task-merge"),
            _resolve_stage_references=resolve,
            _condition_actual=condition_actual,
        ),
    )
    record = runtime._tasks[parent]
    parent_spec = resolve(parent, record.task, record).spec
    rendered = Dispatcher._render_merged_children(
        dispatcher, parent, record, parent_spec
    )
    return [child.task_id for child in rendered or []]


def _assert_returned(runtime: TaskRuntime, registry: _Registry, task_id: str) -> None:
    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.merged_parent_id is None
    assert record.merge_key is None
    assert runtime.result_binding(task_id) is None
    assert task_id in runtime._ready_index
    assert not registry.is_dispatched(task_id)


@pytest.mark.anyio
async def test_a_merged_child_without_its_own_result_runs_again_on_its_own() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    a, b, c = ids["a"], ids["b"], ids["c"]

    settled, _ = runtime.mark_succeeded(a, "wkr-1", _merged_success(runtime, a, b), _TS)

    assert settled == [b]
    assert (_value(runtime, a), _value(runtime, b)) == (a, b)
    assert runtime._tasks[b].status == TaskStatus.DONE
    _assert_returned(runtime, registry, c)
    assert runtime.plan_merge(c, 8, _WORKER.id) == []


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


def _monitor(runtime: TaskRuntime, dispatcher: Any = None) -> EventMonitor:
    watchdog = MagicMock()
    watchdog.enabled = False
    return EventMonitor(
        redis_client=MagicMock(),
        logger=logging.getLogger("task-merge"),
        runtime=runtime,
        dispatcher=dispatcher or MagicMock(),
        worker_registry=MagicMock(),
        node_registry=MagicMock(),
        metrics_recorder=MagicMock(),
        watchdog=watchdog,
    )


def _failed(
    task_id: str, error: str, retryable: bool | None, worker_id: str = "wkr-1"
) -> TaskEvent:
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=worker_id,
        error=error,
        retryable=retryable,
        payload={},
        ts=_TS,
    )


def _assert_run_alone(runtime: TaskRuntime, registry: _Registry, task_id: str) -> None:
    _assert_returned(runtime, registry, task_id)
    assert "wkr-1" not in runtime._tasks[task_id].failed_workers


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        ("batch rejected", True),
        ("batch rejected", False),
        ("worker_heartbeat_expired", None),
        (None, None),
    ],
    ids=["retryable", "non-retryable", "heartbeat-expired", "unregistered"],
)
async def test_a_failed_or_lost_merged_dispatch_runs_each_task_alone(
    error: str | None, retryable: bool | None
) -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    monitor = _monitor(runtime)
    if error:
        monitor._handle_task_event(_failed(ids["a"], error, retryable))
    else:
        monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))

    for task_id in ids.values():
        _assert_run_alone(runtime, registry, task_id)
    assert runtime._tasks[ids["a"]].merged_children is None


@pytest.mark.anyio
async def test_a_task_that_fails_alone_after_its_merge_failed_is_charged() -> None:
    runtime = _runtime(_Registry())
    ids = await _dispatch_merged(runtime)
    a = ids["a"]
    runtime._tasks[a].max_attempts = 1
    monitor = _monitor(runtime)
    monitor._handle_task_event(_failed(a, "batch rejected", retryable=False))
    assert _next(runtime) is not None
    runtime.mark_dispatched(a, _worker("wkr-2"))

    monitor._handle_task_event(
        _failed(a, "own input", retryable=False, worker_id="wkr-2")
    )

    record = runtime._tasks[a]
    assert record.status == TaskStatus.FAILED
    assert "wkr-2" in record.failed_workers


@pytest.mark.anyio
@pytest.mark.parametrize("retryable", [True, False])
async def test_a_replayed_merged_failure_is_absorbed(retryable: bool) -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    monitor = _monitor(runtime)
    event = _failed(ids["a"], "batch rejected", retryable=retryable)

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        monitor._handle_task_event(event)
    monitor._handle_task_event(event)

    for task_id in ids.values():
        _assert_run_alone(runtime, registry, task_id)
        assert registry.durable_status(task_id) == TaskStatus.PENDING


@pytest.mark.anyio
async def test_a_merged_dispatch_failing_after_a_restart_runs_each_task_alone() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    restored = _runtime(registry)
    await restored.rehydrate()

    _monitor(restored)._handle_task_event(
        _failed(ids["a"], "batch rejected", retryable=False)
    )

    for task_id in ids.values():
        _assert_run_alone(restored, registry, task_id)


@pytest.mark.anyio
async def test_the_same_loss_reported_twice_is_absorbed() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    monitor = _monitor(runtime)

    monitor._handle_worker_event(WorkerEvent(type="UNREGISTER", worker_id="wkr-1"))
    monitor._handle_task_event(_failed(ids["a"], "worker lost", retryable=True))

    for task_id in ids.values():
        _assert_run_alone(runtime, registry, task_id)


@pytest.mark.anyio
async def test_a_failed_batch_whose_children_were_all_cancelled_is_not_charged() -> (
    None
):
    registry = _Registry()
    runtime = _runtime(registry, _InterruptRecorder())
    _, a = await _register(runtime, _siblings(names=["a1"]))
    other, b = await _register(runtime, _siblings(names=["b1"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"]]
    runtime.mark_dispatched(parent, _WORKER)
    runtime.cancel_workflow(other)

    _monitor(runtime)._handle_task_event(
        _failed(parent, "batch rejected", retryable=False)
    )

    _assert_run_alone(runtime, registry, parent)
    assert runtime._tasks[b["b1"]].status == TaskStatus.CANCELLED


@pytest.mark.anyio
async def test_a_returned_parent_keeps_nothing_of_its_dispatch() -> None:
    runtime = _runtime(_Registry())
    ids = await _dispatch_merged(runtime)
    assert runtime._tasks[ids["a"]].topic == "tasks"

    _monitor(runtime)._handle_task_event(
        _failed(ids["a"], "batch rejected", retryable=True)
    )

    record = runtime._tasks[ids["a"]]
    assert (record.topic, record.assigned_worker, record.dispatched_ts) == (
        None,
        None,
        None,
    )


@pytest.mark.anyio
async def test_a_released_merge_leaves_the_durable_dispatched_set() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    assert registry.is_dispatched(ids["b"]) and registry.is_dispatched(ids["c"])

    runtime.release_merge(ids["a"])

    assert not registry.is_dispatched(ids["b"])
    assert not registry.is_dispatched(ids["c"])
    assert runtime._tasks[ids["b"]].merge_key is not None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "model", [_MODEL, _LORA, "{source: {identifier: m}, transformers: {dtype: auto}}"]
)
async def test_inference_siblings_merge_on_any_inference_executor(model: str) -> None:
    runtime = _runtime(_Registry())
    _, ids = await _register(runtime, _siblings(model))
    parent = _next(runtime)
    assert parent == ids["a"]

    assert runtime.plan_merge(parent, 8, _WORKER.id) == [ids["b"], ids["c"]]


@pytest.mark.anyio
async def test_tasks_of_different_orgs_never_merge() -> None:
    runtime = _runtime(_Registry())
    _, x = await _register(runtime, _siblings(names=["a"]), org="org-x")
    _, y = await _register(runtime, _siblings(names=["b"]), org="org-y")
    parent = _next(runtime)
    assert parent == x["a"]

    assert runtime.plan_merge(parent, 8, _WORKER.id) == []
    assert runtime._tasks[y["b"]].status == TaskStatus.PENDING


@pytest.mark.anyio
async def test_a_child_result_outside_its_scope_runs_the_child_again() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime)
    a, b, c = ids["a"], ids["b"], ids["c"]
    payload = _merged_success(runtime, a, b)
    payload["child_result_references"][c] = store_result(
        runtime._results, c, {"value": c}, "another-org"
    ).model_dump(mode="json")

    settled, _ = runtime.mark_succeeded(a, "wkr-1", payload, _TS)

    assert settled == [b]
    _assert_returned(runtime, registry, c)


@pytest.mark.anyio
async def test_a_merged_child_advances_its_own_workflows_epochs() -> None:
    runtime = _runtime(_Registry())
    _, a = await _register(runtime, _siblings(names=["a1"]))
    _, b = await _register(
        runtime, _siblings(names=["b1", "b2"], epochs=[["b1"], ["b2"]])
    )
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"]]
    runtime.mark_dispatched(parent, _WORKER)

    runtime.mark_succeeded(
        parent, "wkr-1", _merged_success(runtime, parent, b["b1"]), _TS
    )

    assert runtime._tasks[b["b1"]].status == TaskStatus.DONE
    assert _next(runtime) == b["b2"]


@pytest.mark.anyio
async def test_a_merge_across_workflows_is_dispatched_in_each_workflow() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    first, a = await _register(runtime, _siblings(names=["a1"]))
    other, b = await _register(runtime, _siblings(names=["b1"]))
    parent = _next(runtime)
    assert parent is not None

    runtime.plan_merge(parent, 8, _WORKER.id)

    assert registry.dispatched[other] == {b["b1"]}
    assert b["b1"] not in registry.dispatched[first]


@pytest.mark.anyio
async def test_a_replayed_success_heals_a_merged_childs_other_workflow() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, a = await _register(runtime, _siblings(names=["a1"]))
    _, b = await _register(runtime, _siblings(names=["b1"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    runtime.plan_merge(parent, 8, _WORKER.id)
    runtime.mark_dispatched(parent, _WORKER)
    payload = _merged_success(runtime, parent, b["b1"])

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        runtime.mark_succeeded(parent, "wkr-1", payload, _TS)
    assert registry.durable_status(b["b1"]) == TaskStatus.DISPATCHED
    runtime.mark_succeeded(parent, "wkr-1", payload, _TS)

    assert registry.durable_status(b["b1"]) == TaskStatus.DONE


@pytest.mark.anyio
async def test_a_failed_commit_leaves_a_parent_failure_whole_for_its_replay() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, t = await _register(
        runtime,
        _siblings(names=["a1", "a2", "a3", "a4"], epochs=[["a1", "a2", "a3"], ["a4"]]),
    )
    parent = _next(runtime)
    assert parent == t["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [t["a2"], t["a3"]]
    runtime.mark_dispatched(parent, _WORKER)

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        runtime.mark_failed(parent, "wkr-1", {}, _TS, error="bad input")
    runtime.mark_failed(parent, "wkr-1", {}, _TS, error="bad input")

    assert runtime._tasks[t["a4"]].status == TaskStatus.FAILED
    assert registry.durable_status(t["a4"]) == TaskStatus.FAILED
    assert {_next(runtime), _next(runtime)} == {t["a2"], t["a3"]}


@pytest.mark.anyio
async def test_a_failed_commit_leaves_a_parent_success_whole_for_its_replay() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, a = await _register(
        runtime, _siblings(names=["a1", "a2"], epochs=[["a1"], ["a2"]])
    )
    _, b = await _register(runtime, _siblings(names=["b1"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"]]
    runtime.mark_dispatched(parent, _WORKER)
    payload = _merged_success(runtime, parent)

    registry.fail_next = True
    with pytest.raises(ConnectionError):
        runtime.mark_succeeded(parent, "wkr-1", payload, _TS)
    runtime.mark_succeeded(parent, "wkr-1", payload, _TS)

    assert {_next(runtime), _next(runtime)} == {a["a2"], b["b1"]}


@pytest.mark.anyio
async def test_a_merge_planned_while_the_store_is_down_returns_its_siblings() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, t = await _register(runtime, _siblings())
    parent = _next(runtime)
    assert parent is not None

    registry.down = True
    with pytest.raises(ConnectionError):
        runtime.plan_merge(parent, 8, _WORKER.id)
    with pytest.raises(ConnectionError):
        runtime.release_merge(parent)
    runtime.requeue(parent, front=True)
    registry.down = False

    for child in (t["b"], t["c"]):
        assert runtime._tasks[child].status == TaskStatus.PENDING
        assert runtime._tasks[child].merged_parent_id is None
    assert {_next(runtime), _next(runtime), _next(runtime)} == set(t.values())


@pytest.mark.anyio
async def test_a_child_the_dispatch_cannot_render_leaves_the_merge() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    a, b, c = ids["a"], ids["b"], ids["c"]

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        if task_id == c:
            raise ValueError("bad child input")
        return task

    assert _render(runtime, a, _resolve) == [b]
    assert runtime._tasks[a].merged_children == [b]
    _assert_returned(runtime, registry, c)


@pytest.mark.anyio
async def test_a_child_whose_condition_is_not_met_leaves_the_merge() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    a, b, c = ids["a"], ids["b"], ids["c"]
    condition = ConditionSpec(node="gate", field="value", equals="go")

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        spec = task.spec.model_copy(update={"condition": condition})
        return task.model_copy(update={"spec": spec})

    def _actual(record: Any, _condition: Any) -> str:
        return "stop" if record.task_id == c else "go"

    assert _render(runtime, a, _resolve, _actual) == [b]
    _assert_returned(runtime, registry, c)


@pytest.mark.anyio
async def test_a_child_whose_spec_cannot_dispatch_leaves_the_merge() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    a, b, c = ids["a"], ids["b"], ids["c"]

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        if task_id != c:
            return task
        model = task.spec.model.model_copy(update={"vllm": {"dtype": "auto"}})
        spec = task.spec.model_copy(update={"model": model, "enforce_cpu": True})
        return task.model_copy(update={"spec": spec})

    assert _render(runtime, a, _resolve) == [b]
    _assert_returned(runtime, registry, c)


@pytest.mark.anyio
async def test_a_child_that_renders_a_different_spec_leaves_the_merge() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    a, b, c = ids["a"], ids["b"], ids["c"]

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        spec = task.spec.model_copy(
            update={
                "upstreamResults": {"up": {"value": task_id}},
                "inference": {"system_prompt": task_id},
            }
        )
        if task_id == c:
            source = spec.model.source.model_copy(update={"identifier": "m-other"})
            model = spec.model.model_copy(update={"source": source})
            spec = spec.model_copy(update={"model": model})
        return task.model_copy(update={"spec": spec})

    assert _render(runtime, a, _resolve) == [b]
    _assert_returned(runtime, registry, c)


@pytest.mark.anyio
async def test_a_child_cancelled_during_the_render_is_left_out_of_it() -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    _, a = await _register(runtime, _siblings(names=["a1"]))
    _, b = await _register(runtime, _siblings(names=["b1"]))
    other, c = await _register(runtime, _siblings(names=["c1"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"], c["c1"]]

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        if task_id == c["c1"]:
            runtime.cancel_workflow(other)
        return task

    assert _render(runtime, parent, _resolve) == [b["b1"]]
    assert runtime._tasks[c["c1"]].status == TaskStatus.CANCELLED


@pytest.mark.anyio
async def test_children_returned_during_the_render_are_left_out_of_it() -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    first, a = await _register(runtime, _siblings(names=["a1"]))
    _, b = await _register(runtime, _siblings(names=["b1", "b2"]))
    parent = _next(runtime)
    assert parent == a["a1"]
    assert runtime.plan_merge(parent, 8, _WORKER.id) == [b["b1"], b["b2"]]

    def _resolve(task_id: str, task: Any, record: Any) -> Any:
        if task_id == b["b1"]:
            runtime.cancel_workflow(first)
        return task

    assert _render(runtime, parent, _resolve) == []
    for child in (b["b1"], b["b2"]):
        assert runtime._tasks[child].status == TaskStatus.PENDING
        assert child in runtime._ready_index


@pytest.mark.anyio
async def test_a_restored_task_merges_under_its_current_key() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, x = await _register(runtime, _siblings(names=["a"]), org="org-x")
    _, y = await _register(runtime, _siblings(names=["b"]), org="org-y")
    legacy_key = "vllm:legacy-unscoped"
    for task_id in (x["a"], y["b"]):
        runtime._tasks[task_id].merge_key = legacy_key
        runtime._persist_locked(task_id)

    restored = _runtime(registry)
    await restored.rehydrate()

    assert restored._tasks[x["a"]].merge_key != restored._tasks[y["b"]].merge_key
    parent = _next(restored)
    assert parent is not None
    assert restored.plan_merge(parent, 8, _WORKER.id) == []


@pytest.mark.anyio
async def test_a_child_not_ready_yet_leaves_the_merge_still_mergeable() -> None:
    runtime = _runtime(_Registry())
    ids = await _dispatch_merged(runtime, dispatch=False)

    runtime.release_merged_child(ids["a"], ids["c"], unmerge=False)

    record = runtime._tasks[ids["c"]]
    assert record.status == TaskStatus.PENDING
    assert record.merge_key is not None
    assert runtime._tasks[ids["a"]].merged_children == [ids["b"]]


@pytest.mark.anyio
async def test_releasing_a_child_no_longer_merged_leaves_it_alone() -> None:
    runtime = _runtime(_Registry())
    ids = await _dispatch_merged(runtime, dispatch=False)
    runtime.release_merged_child(ids["a"], ids["c"], unmerge=True)
    assert _next(runtime) == ids["c"]
    runtime.mark_dispatched(ids["c"], _WORKER)

    runtime.release_merged_child(ids["a"], ids["c"], unmerge=True)

    assert runtime._tasks[ids["c"]].status == TaskStatus.DISPATCHED


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
@pytest.mark.parametrize(
    ("parent_status", "mergeable"),
    [(TaskStatus.PENDING, True), (TaskStatus.FAILED, False)],
)
async def test_a_restart_returns_the_children_of_a_parent_no_longer_running(
    parent_status: TaskStatus, mergeable: bool
) -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    ids = await _dispatch_merged(runtime, dispatch=False)
    if parent_status != TaskStatus.PENDING:
        runtime._tasks[ids["a"]].status = parent_status
        runtime._persist_locked(ids["a"])

    restored = _runtime(registry)
    await restored.rehydrate()

    for child in (ids["b"], ids["c"]):
        record = restored._tasks[child]
        assert record.status == TaskStatus.PENDING
        assert record.merged_parent_id is None
        assert (record.merge_key is not None) is mergeable
        assert child in restored._ready_index
        assert not registry.is_dispatched(child)


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
    _, other_ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    assert parent is not None
    merged = runtime.plan_merge(parent, 8, _WORKER.id)
    runtime.mark_dispatched(parent, _WORKER)
    assert set(other_ids.values()) <= set(merged)

    runtime.cancel_workflow(first)
    settled, _ = runtime.mark_succeeded(
        parent, "wkr-1", _merged_success(runtime, parent, *merged), _TS
    )

    assert settled == []
    for child in other_ids.values():
        _assert_returned(runtime, registry, child)


@pytest.mark.anyio
async def test_a_merged_child_of_a_cancelled_workflow_stays_cancelled() -> None:
    runtime = _runtime(_Registry(), _InterruptRecorder())
    _, first_ids = await _register(runtime, _siblings())
    other, other_ids = await _register(runtime, _siblings())
    parent = _next(runtime)
    assert parent is not None
    merged = runtime.plan_merge(parent, 8, _WORKER.id)
    runtime.mark_dispatched(parent, _WORKER)

    runtime.cancel_workflow(other)
    assert not set(other_ids.values()) & set(
        runtime._tasks[parent].merged_children or []
    )
    runtime.mark_succeeded(
        parent, "wkr-1", _merged_success(runtime, parent, *merged), _TS
    )

    for task_id in other_ids.values():
        assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    for task_id in first_ids.values():
        assert runtime._tasks[task_id].status == TaskStatus.DONE
