"""Durable persistence and restart rehydration of TaskRuntime."""

import logging
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from server.clients.redis import workflow_credential_key
from server.config import OrchestrationConfig
from server.registries.workflow import (
    PersistedTask,
    WorkflowSched,
    load_task_state,
    task_sources,
)
from server.task.models import PublishGate, TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
from tests.server.runtime_helpers import manual_durability_retry
from tests.server.stored_state import StoredLedgers, StoredTaskStates
from tests.support.waiting import pop_ready


class FakeWorkflowRegistry(StoredTaskStates, StoredLedgers):
    """In-memory registry that round-trips state through the real model JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.sched: dict[str, str] = {}
        self.workflow_task_ids: dict[str, list[str]] = {}
        self.v2_blobs: dict[str, str] = {}
        self.dynamic_task_ids: dict[str, set[str]] = {}
        self.blueprints: dict[str, list[PersistedTask]] = {}

    async def register_workflow_async(
        self,
        workflow_id: str,
        tasks: Sequence[PersistedTask],
        sched: WorkflowSched,
        v2: Any = None,
        ledger: Any = None,
        submitted_at: str | None = None,
        blueprints: Any = (),
    ) -> None:
        self.workflow_task_ids[workflow_id] = [t.record.task_id for t in tasks]
        self.put_sources([*tasks, *blueprints])
        self.put_tasks(tasks)
        self.sched[workflow_id] = sched.model_dump_json()
        if v2 is not None:
            self.v2_blobs[workflow_id] = v2.model_dump_json()
        if ledger is not None:
            self.put_ledger(workflow_id, ledger)
        self.blueprints[workflow_id] = list(blueprints)

    async def load_blueprints_async(self, workflow_id: str) -> list[PersistedTask]:
        return self.blueprints.get(workflow_id, [])

    async def get_v2_workflow_async(self, workflow_id: str) -> Any:
        from server.task.v2 import PersistedV2Workflow

        blob = self.v2_blobs.get(workflow_id)
        return PersistedV2Workflow.model_validate_json(blob) if blob else None

    def save_ledger(self, workflow_id: str, ledger: Any, control: Any = None) -> None:
        self.put_ledger(workflow_id, ledger)

    async def get_remaining_tasks_async(self, workflow_id: str) -> set[str]:
        ids = [
            *self.workflow_task_ids.get(workflow_id, ()),
            *sorted(self.dynamic_task_ids.get(workflow_id, set())),
        ]
        return {
            task_id
            for task_id in ids
            if (stored := self.stored_task(task_id)) is not None
            and stored.record.status
            not in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
        }

    def get_workflow_ids(self) -> set[str]:
        return set(self.workflow_task_ids)

    async def get_workflow_ids_async(self) -> set[str]:
        return self.get_workflow_ids()

    def get_workflow_record(self, workflow_id: str) -> Any:
        ids = self.workflow_task_ids.get(workflow_id)
        return SimpleNamespace(task_ids=list(ids)) if ids is not None else None

    async def get_workflow_record_async(self, workflow_id: str) -> Any:
        return self.get_workflow_record(workflow_id)

    def save_task_states(self, items: Sequence[PersistedTask]) -> None:
        self.put_sources(items)
        self.put_tasks(items)

    async def save_task_states_async(self, items: list[PersistedTask]) -> None:
        self.save_task_states(items)

    def save_workflow_sched(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.sched[workflow_id] = WorkflowSched(
            in_epoch_order=in_epoch_order, epoch_frontier=frontier
        ).model_dump_json()

    async def save_workflow_sched_async(
        self, workflow_id: str, in_epoch_order: bool, frontier: int
    ) -> None:
        self.save_workflow_sched(workflow_id, in_epoch_order, frontier)

    def load_workflow_sched(self, workflow_id: str) -> WorkflowSched | None:
        blob = self.sched.get(workflow_id)
        return WorkflowSched.model_validate_json(blob) if blob else None

    async def load_workflow_sched_async(self, workflow_id: str) -> WorkflowSched | None:
        return self.load_workflow_sched(workflow_id)

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
        control: Any = None,
    ) -> None:
        self.put_tasks(records)
        if sched is not None:
            self.sched[workflow_id] = sched.model_dump_json()

    def commit_dynamic_tasks(
        self,
        workflow_id: str,
        records: Sequence[PersistedTask],
        ledger: Any,
        retire: Sequence[str] = (),
        **membership: Any,
    ) -> None:
        self.put_tasks(records)
        for item in records:
            self.dynamic_task_ids.setdefault(workflow_id, set()).add(
                item.record.task_id
            )
        self.put_ledger(workflow_id, ledger)

    async def get_dynamic_task_ids_async(self, workflow_id: str) -> set[str]:
        return set(self.dynamic_task_ids.get(workflow_id, set()))


class _WorkerRegistryStub:
    def get_worker(self, worker_id: str) -> Any:
        return SimpleNamespace(id=worker_id, node_id="nde-1")

    def publish_interrupt(self, *args: Any) -> int:
        return 0

    def release_worker(self, *args: Any) -> bool:
        return False

    def reservations(self) -> list[Any]:
        return []


def _runtime(
    registry: FakeWorkflowRegistry, vault: InMemoryCredentialVault | None = None
) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("rehydrate-test"),
        credential_vault=vault or InMemoryCredentialVault(),
        durability_retry=manual_durability_retry,
    )


async def _register(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, results = await runtime.register(
        "owner", "org", payload, format="native"
    )
    return workflow_id, {str(r.graph_node_name): r.task_id for r in results}


GRAPH = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: graph
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
"""

WIDE = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: wide
spec:
  graph:
    nodes:
      - {name: a, spec: {taskType: echo}}
      - {name: b, spec: {taskType: echo}}
      - {name: c, spec: {taskType: echo}}
      - {name: d, spec: {taskType: echo}}
"""

EPOCH_GRAPH = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: graph
  annotations:
    schedule_hint:
      node_schedule_in_epoch_order: true
      node_execution_order:
        - [a, b]
        - [c]
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
      - name: b
        spec:
          taskType: echo
      - name: c
        dependsOn: [a]
        spec:
          taskType: echo
"""


@pytest.mark.anyio
async def test_persisted_task_round_trips_failed_workers_and_deps() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    record = runtime.get_record(b)
    assert record is not None
    record.failed_workers.append("wkr-dead")

    pt = PersistedTask(record=record, depends_on={a}, epoch_index=2)
    restored = load_task_state(
        pt.model_dump_json(), task_sources([pt])[pt.record.workflow_id]
    )

    # failed_workers is exclude=True on TaskRecord but must survive persistence;
    # depends_on round-trips through a JSON list back to a set.
    assert restored.record.failed_workers == ["wkr-dead"]
    assert restored.depends_on == {a}
    assert restored.epoch_index == 2
    assert restored.record.task_id == b


@pytest.mark.anyio
async def test_rehydrate_restores_completed_and_ready_state() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1

    record_a = restored.get_record(a)
    record_b = restored.get_record(b)
    assert record_a is not None
    assert record_a.status == TaskStatus.DONE
    assert record_b is not None
    assert record_b.status == TaskStatus.PENDING

    # b's only dependency completed, so it is the sole ready task.
    assert restored.ready_queue_length() == 1
    assert pop_ready(restored) == b
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrate_keeps_in_flight_task_dispatched() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-9", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()

    record = restored.get_record(a)
    assert record is not None
    assert record.status == TaskStatus.DISPATCHED
    assert record.assigned_worker == "wkr-9"
    # An in-flight task is not re-queued; its completion arrives via the stream.
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrate_restores_epoch_frontier() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, EPOCH_GRAPH)

    runtime.mark_succeeded(ids["a"], None, {}, "2026-06-01T00:00:00Z")
    runtime.mark_succeeded(ids["b"], None, {}, "2026-06-01T00:00:01Z")
    assert runtime._epochs.workflow_epoch_frontier[workflow_id] == 1

    restored = _runtime(registry)
    await restored.rehydrate()

    assert restored._epochs.workflow_epoch_frontier[workflow_id] == 1
    assert pop_ready(restored) == ids["c"]


@pytest.mark.anyio
async def test_mark_succeeded_is_idempotent_under_replay() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    # A replayed completion must not re-apply.
    replay = runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert replay is not None and (replay.merged_children, replay.usages) == ([], [])

    # b is enqueued exactly once despite the replay.
    assert runtime.ready_queue_length() == 1
    assert pop_ready(runtime) == b
    assert runtime.ready_queue_length() == 0


@pytest.mark.anyio
async def test_rehydrated_in_flight_task_is_protected_then_released() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()

    # Within the grace window the worker's rehydrated task is protected; with a
    # zero window it is not.
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is True
    assert restored.has_rehydrated_in_flight("wkr-7", 0.0) is False
    assert restored.has_rehydrated_in_flight("wkr-other", 600.0) is False


@pytest.mark.anyio
async def test_rehydrated_protection_clears_on_completion() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is True

    restored.mark_succeeded(a, "wkr-7", {}, "2026-06-01T00:00:00Z")
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is False


@pytest.mark.anyio
async def test_recover_clears_rehydrated_protection() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-7", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))

    restored = _runtime(registry)
    await restored.rehydrate()
    assert restored.recover_tasks_for_worker("wkr-7", spend_attempt=True).lost == [a]
    assert restored.has_rehydrated_in_flight("wkr-7", 600.0) is False


@pytest.mark.anyio
async def test_terminal_task_does_not_regress_on_replayed_dispatch_or_start() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))
    runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    # A replayed dispatch / start / progress update must not move a's status
    # back to DISPATCHED.
    record_dispatch(runtime, a, cast(Any, worker), expect=PublishGate.NOT_PENDING)
    runtime.mark_started(a, "wkr-1", {}, "2026-06-01T00:00:01Z")
    runtime.mark_updated(a, "wkr-1", {"note": "stale"})

    record = runtime.get_record(a)
    assert record is not None
    assert record.status == TaskStatus.DONE
    assert record.latest_update is None


@pytest.mark.anyio
async def test_mark_succeeded_applies_in_memory_atomically_when_persist_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]
    record_dispatch(runtime, a)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("redis down")

    monkeypatch.setattr(registry, "commit_transition", boom)
    with pytest.raises(RuntimeError):
        runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    # Persistence is last, so the in-memory transition is fully applied (not
    # half-done): 'a' is DONE and its dependent 'b' is enqueued.
    record_a = runtime.get_record(a)
    assert record_a is not None and record_a.status == TaskStatus.DONE
    assert b in runtime._ready.ready_index

    # The at-least-once replay re-runs and is a no-op via the idempotency guard.
    monkeypatch.setattr(registry, "commit_transition", lambda *args, **kwargs: None)
    replay = runtime.mark_succeeded(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert replay is not None and (replay.merged_children, replay.usages) == ([], [])


@pytest.mark.anyio
async def test_mark_failed_applies_cascade_atomically_when_persist_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("redis down")

    monkeypatch.setattr(registry, "commit_transition", boom)
    with pytest.raises(RuntimeError):
        runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    # The whole cascade is applied in memory before persistence runs: both the
    # task and its dependent are FAILED, never a partial mix.
    record_a = runtime.get_record(a)
    record_b = runtime.get_record(b)
    assert record_a is not None and record_a.status == TaskStatus.FAILED
    assert record_b is not None and record_b.status == TaskStatus.FAILED

    # Replay is a no-op via the idempotency guard (task already terminal).
    monkeypatch.setattr(registry, "commit_transition", lambda *args, **kwargs: None)
    impacted, _ = runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert impacted == []


@pytest.mark.anyio
async def test_a_replayed_terminal_event_lands_the_cascade_its_store_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    real_commit = registry.commit_transition
    calls = {"n": 0}

    def flaky_commit(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RedisConnectionError("redis down")
        real_commit(*args, **kwargs)

    monkeypatch.setattr(registry, "commit_transition", flaky_commit)

    def persisted_status(task_id: str) -> str:
        return registry.stored_record(task_id).status

    # The cascade applies in memory and its write is held.
    runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert persisted_status(a) == TaskStatus.PENDING
    assert persisted_status(b) == TaskStatus.PENDING
    assert not runtime._committer.durable(workflow_id)

    # A replay of the same TASK_FAILED writes what the workflow holds.
    impacted, _ = runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert impacted == []
    assert persisted_status(a) == TaskStatus.FAILED
    assert persisted_status(b) == TaskStatus.FAILED
    assert runtime._committer.durable(workflow_id)


_BESIDE = GRAPH + """      - name: c
        spec:
          taskType: echo
"""


class _RaisingOnce:
    """A registry write that raises a fault of its own the first ``times`` times."""

    def __init__(self, registry: FakeWorkflowRegistry, times: int) -> None:
        self._commit = registry.commit_transition
        self.times = times
        self.written: list[str] = []

    def __call__(self, workflow: str, **kwargs: Any) -> None:
        if self.times:
            self.times -= 1
            raise RuntimeError("the write could not be encoded")
        self.written.extend(item.record.task_id for item in kwargs.get("records", ()))
        self._commit(workflow, **kwargs)


def _persisted_status(registry: FakeWorkflowRegistry, task_id: str) -> str:
    return registry.stored_record(task_id).status


@pytest.mark.anyio
@pytest.mark.parametrize("next_write", ["replay", "another_task"])
async def test_a_cascade_whose_write_raised_lands_with_the_next_write(
    monkeypatch: pytest.MonkeyPatch, next_write: str
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _BESIDE)
    a, b, c = ids["a"], ids["b"], ids["c"]
    writes = _RaisingOnce(registry, times=1)
    monkeypatch.setattr(registry, "commit_transition", writes)

    # The cascade's own write raised: the transition's error, and a rewrite owed.
    with pytest.raises(RuntimeError):
        runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert _persisted_status(registry, b) == TaskStatus.PENDING
    assert not runtime._committer.durable(workflow_id)
    assert not runtime._durability.pending(workflow_id)

    match next_write:
        case "replay":
            impacted, _ = runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
            assert impacted == []
        case _:
            # Publishing another task makes the workflow's owed writes first.
            record_dispatch(runtime, c)
    assert _persisted_status(registry, a) == TaskStatus.FAILED
    assert _persisted_status(registry, b) == TaskStatus.FAILED
    assert runtime._committer.durable(workflow_id)


@pytest.mark.anyio
@pytest.mark.parametrize("report", ["succeeded", "failed", "cancelled"])
async def test_a_replayed_terminal_event_rewrites_only_its_own_task(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, WIDE)
    finished = [ids[name] for name in ("a", "b", "c")]
    for task_id in finished:
        record_dispatch(runtime, task_id)
        runtime.mark_succeeded(task_id, "wkr-1", {}, "2026-06-01T00:00:00Z")
    last = ids["d"]
    record_dispatch(runtime, last)
    settle: Callable[..., Any]
    match report:
        case "succeeded":
            settle = runtime.mark_succeeded
        case "failed":
            settle = runtime.mark_failed
        case _:
            runtime.cancel_workflow(workflow_id)
            settle = runtime.mark_cancelled

    written: list[str] = []
    real_commit = registry.commit_transition

    def recording(workflow: str, **kwargs: Any) -> None:
        written.extend(item.record.task_id for item in kwargs.get("records", ()))
        real_commit(workflow, **kwargs)

    monkeypatch.setattr(registry, "commit_transition", recording)
    settle(last, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert written == [last]


@pytest.mark.anyio
async def test_mark_cancelled_applies_in_memory_atomically_when_persist_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, GRAPH)
    a = ids["a"]
    record_dispatch(runtime, a)
    runtime.cancel_workflow(workflow_id)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("redis down")

    monkeypatch.setattr(registry, "commit_transition", boom)
    with pytest.raises(RuntimeError):
        runtime.mark_cancelled(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    # Persistence is last, so the cancellation is fully applied in memory.
    record_a = runtime.get_record(a)
    assert record_a is not None and record_a.status == TaskStatus.CANCELLED

    # Replay is a no-op via the idempotency guard (task already cancelled).
    monkeypatch.setattr(registry, "commit_transition", lambda *args, **kwargs: None)
    runtime.mark_cancelled(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    record_a = runtime.get_record(a)
    assert record_a is not None and record_a.status == TaskStatus.CANCELLED


@pytest.mark.anyio
async def test_mark_cancelled_repersists_on_replay_after_failed_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, GRAPH)
    a = ids["a"]
    record_dispatch(runtime, a)
    runtime.cancel_workflow(workflow_id)

    real_commit = registry.commit_transition
    calls = {"n": 0}

    def flaky_commit(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("redis down")
        real_commit(*args, **kwargs)

    monkeypatch.setattr(registry, "commit_transition", flaky_commit)

    def persisted_status(task_id: str) -> str:
        return registry.stored_record(task_id).status

    # Attempt 1: cancellation applies in memory, but the durable write fails.
    with pytest.raises(RuntimeError):
        runtime.mark_cancelled(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert persisted_status(a) == TaskStatus.CANCELLING

    # Replay of the same cancellation: the guard heals by re-persisting.
    runtime.mark_cancelled(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    assert persisted_status(a) == TaskStatus.CANCELLED


@pytest.mark.anyio
async def test_cancel_workflow_commits_atomically_on_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("redis down")

    def persisted_status(task_id: str) -> str:
        return registry.stored_record(task_id).status

    monkeypatch.setattr(registry, "commit_transition", boom)
    with pytest.raises(RuntimeError):
        runtime.cancel_workflow(workflow_id)

    # Persist is the single last step, so the cancellation is fully applied in
    # memory even though the commit failed.
    record_a = runtime.get_record(a)
    assert record_a is not None and record_a.status == TaskStatus.CANCELLED

    # The cancel has no event-replay backstop, but the atomic commit never ran,
    # so durable state is untouched — nothing is half-cancelled. A fresh restart
    # restores the pre-cancel workflow, which the operator can cancel again.
    assert persisted_status(a) == TaskStatus.PENDING
    assert persisted_status(b) == TaskStatus.PENDING

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1
    restored_a = restored.get_record(a)
    restored_b = restored.get_record(b)
    assert restored_a is not None and restored_a.status == TaskStatus.PENDING
    assert restored_b is not None and restored_b.status == TaskStatus.PENDING


@pytest.mark.anyio
async def test_rehydrate_restores_cancelled_workflow() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]

    runtime.cancel_workflow(workflow_id)

    restored = _runtime(registry)
    await restored.rehydrate()

    # Cancelled tasks rehydrate terminal and are never re-enqueued.
    record_a = restored.get_record(a)
    record_b = restored.get_record(b)
    assert record_a is not None and record_a.status == TaskStatus.CANCELLED
    assert record_b is not None and record_b.status == TaskStatus.CANCELLED
    assert restored.ready_queue_length() == 0


@pytest.mark.anyio
async def test_a_cascaded_dependent_reads_failed_before_and_after_a_restart() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, GRAPH)
    a, b = ids["a"], ids["b"]
    runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    info = runtime.describe_task(b)
    assert info is not None and info.failed

    restored = _runtime(registry)
    await restored.rehydrate()
    info = restored.describe_task(b)
    assert info is not None and info.failed


_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo}
      - name: c
        dependsOn: [b]
        spec: {taskType: echo}
"""


@pytest.mark.anyio
async def test_a_failure_cascades_through_every_level_of_dependents() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, _CHAIN)
    a, b, c = ids["a"], ids["b"], ids["c"]

    impacted, _ = runtime.mark_failed(a, "wkr-1", {}, "2026-06-01T00:00:00Z")

    reason = f"Dependency {a} failed"
    assert sorted(impacted) == sorted([(b, reason), (c, reason)])
    for task_id in (b, c):
        record = runtime.get_record(task_id)
        assert record is not None and record.status == TaskStatus.FAILED
        assert record.error == reason
    assert runtime.workflow_settlement(workflow_id).settled


_AGENT_WITH_KEY = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: bind}
spec:
  taskType: echo
  graph:
    nodes:
      - name: solver
        spec:
          taskType: agent
          v2: {authority: {invoke: [model], delegate: []}, tools: [{name: model}]}
          harness: {backend: scripted, version: v1, params: {script: []}}
          model_binding:
            mode: openai
            url: "https://api.example/v1"
            model: m
            api_key: "sk-user-key"
"""


@pytest.mark.anyio
async def test_a_restart_keeps_live_vaults_and_drops_settled_and_unregistered_ones():
    registry = FakeWorkflowRegistry()
    vault = InMemoryCredentialVault()
    runtime = _runtime(registry, vault)
    live, _ = await runtime.register("owner", "org", _AGENT_WITH_KEY, format="native")
    settled, _ = await runtime.register("owner", "org", GRAPH, format="native")
    runtime.cancel_workflow(settled)
    # A crash after the terminal commit but before the purge, and one between vaulting
    # and registering, each leave a vault behind.
    await vault.store_values(settled, {"msk-left": "sk"})
    await vault.store_values("wfl-never-registered", {"msk-orphan": "sk"})
    live_key = workflow_credential_key(live)
    vault.redis.expiring.add(live_key)

    await _runtime(registry, vault).rehydrate()

    assert live_key in vault.redis.hashes
    assert live_key not in vault.redis.expiring
    assert workflow_credential_key(settled) not in vault.redis.hashes
    assert workflow_credential_key("wfl-never-registered") not in vault.redis.hashes


SERVE = """
apiVersion: flowmesh/v1
kind: Workflow
metadata:
  name: serve
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: dev_model
"""


@pytest.mark.anyio
async def test_rehydrate_keeps_a_requeued_serve_tasks_first_start() -> None:
    registry = FakeWorkflowRegistry()
    runtime = _runtime(registry)
    _, ids = await _register(runtime, SERVE)
    a = ids["a"]

    worker = SimpleNamespace(id="wkr-1", node_id="nde-1")
    record_dispatch(runtime, a, cast(Any, worker))
    runtime.mark_started(a, "wkr-1", {}, "2026-06-01T00:00:00Z")
    record = runtime.get_record(a)
    assert record is not None and record.first_started_ts is not None
    first_started = record.first_started_ts
    runtime.return_dispatch(a, "wkr-1", increment_retry=False, front=True)

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1

    restored_record = restored.get_record(a)
    assert restored_record is not None
    assert restored_record.status == TaskStatus.PENDING
    assert restored_record.started_ts is None
    assert restored_record.first_started_ts == first_started
