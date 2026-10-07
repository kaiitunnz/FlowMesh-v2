"""A task whose worker could not reach its inputs runs again at no cost.

The report spends no attempt and has to come from the dispatch holding the task. Control
reads the inputs the task consumes itself before the task runs again or fails.
"""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.config import OrchestrationConfig
from server.orchestration.state import AttemptStatus, WorkItemStatus
from server.services.task_events import TaskEventPublisher
from server.task.models import TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.results import ResultUnavailable, ResultUnreadable
from server.task.runtime import TaskRuntime
from shared.content import ContentReference, reference_for
from shared.schemas.event import TaskEvent, TaskFailureKind
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_task_merge import (
    _monitor,
    _next,
    _register,
    _Registry,
    _runtime,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
)
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import (
    _worker,
    _WorkerRegistryStub,
)

_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: unavailable}
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


def _failure(
    task_id: str, worker_id: str, dispatch_id: str, **fields: Any
) -> TaskEvent:
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id=worker_id,
        dispatch_id=dispatch_id,
        error="task cannot reach input",
        ts=_TS,
        **{"retryable": True, **fields},
    )


async def _consumer(runtime: TaskRuntime) -> tuple[str, ContentReference]:
    """A dispatched task and the upstream result it consumes."""
    _, ids = await _register(runtime, _CHAIN)
    producer = _next(runtime)
    record_dispatch(runtime, producer, "wkr-0", "dsp-0")
    payload = result_payload(runtime._results, producer, {"value": "x"}, "org")
    runtime.mark_succeeded(producer, "wkr-0", payload, _TS, dispatch_id="dsp-0")
    task_id = _next(runtime)
    assert task_id == ids["b"]
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    binding = runtime.result_binding(producer)
    assert binding is not None and binding.reference is not None
    return task_id, binding.reference


def _unavailable(
    task_id: str, references: list[ContentReference], dispatch_id: str = "dsp-1"
) -> TaskEvent:
    return _failure(
        task_id,
        "wkr-1",
        dispatch_id,
        failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
        unavailable_inputs=references,
    )


@pytest.mark.anyio
async def test_an_ordinary_retryable_failure_still_spends_an_attempt() -> None:
    runtime = _runtime(_Registry())
    task_id, _ = await _consumer(runtime)

    _monitor(runtime).handle_task_event(_failure(task_id, "wkr-1", "dsp-1"))

    record = runtime._tasks[task_id]
    assert record.attempts == 1
    assert record.failed_workers == ["wkr-1"]


@pytest.mark.anyio
async def test_a_report_from_another_dispatch_changes_nothing() -> None:
    runtime = _runtime(_Registry())
    task_id, reference = await _consumer(runtime)

    _monitor(runtime).handle_task_event(_unavailable(task_id, [reference], "dsp-old"))

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.DISPATCHED
    assert record.dispatch_id == "dsp-1"


@pytest.mark.anyio
async def test_a_cancelling_task_settles_cancelled() -> None:
    runtime = _runtime(_Registry())
    task_id, reference = await _consumer(runtime)
    runtime.cancel_workflow(runtime._tasks[task_id].workflow_id)
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING

    _monitor(runtime).handle_task_event(_unavailable(task_id, [reference]))

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED


_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: v2-unavailable}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


async def _v2_consumer(
    runtime: TaskRuntime,
) -> tuple[str, str, ContentReference]:
    workflow_id, ids = await _register_v2(runtime, _V2)
    producer, task_id = ids["a"], ids["b"]
    assert _pop_ready(runtime) == [producer]
    record_dispatch(runtime, producer, cast(Any, _worker("wkr-0")), "dsp-0")
    payload = _planned(runtime, producer, ["x"])
    runtime.mark_succeeded(producer, "wkr-0", payload, _TS, dispatch_id="dsp-0")
    assert _pop_ready(runtime) == [task_id]
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-1")), "dsp-1")
    runtime.mark_started(task_id, "wkr-1", {}, _TS, dispatch_id="dsp-1")
    binding = runtime.result_binding(producer)
    assert binding is not None and binding.reference is not None
    return workflow_id, task_id, binding.reference


@pytest.mark.anyio
async def test_a_v2_return_closes_its_attempt_without_charging_it() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, task_id, reference = await _v2_consumer(runtime)

    runtime.fail_dispatch(
        task_id,
        "wkr-1",
        {},
        _TS,
        "dsp-1",
        error="task cannot reach input",
        retryable=True,
        failure_kind=TaskFailureKind.INPUT_UNAVAILABLE,
        unavailable_inputs=[reference],
    )
    runtime._redrive.run_due()

    engine = runtime._engines[workflow_id]
    work_item = engine.work_item(task_id)
    assert work_item is not None
    assert work_item.status is WorkItemStatus.READY
    assert [engine._ledger.attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED
    ]
    assert runtime._tasks[task_id].attempts == 0

    assert _pop_ready(runtime) == [task_id]
    record_dispatch(runtime, task_id, cast(Any, _worker("wkr-2")), "dsp-2")
    runtime.mark_started(task_id, "wkr-2", {}, _TS, dispatch_id="dsp-2")
    runtime.mark_succeeded(task_id, "wkr-2", {}, _TS, dispatch_id="dsp-2")
    assert [engine._ledger.attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED,
        AttemptStatus.SUCCEEDED,
    ]


@pytest.mark.anyio
async def test_returning_an_undispatched_task_saves_no_ledger() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    _, ids = await _register_v2(runtime, _V2)
    saves: list[str] = []
    save = registry.save_ledger_snapshot

    def _counted(workflow_id: str, snapshot: Any) -> Any:
        saves.append(workflow_id)
        return save(workflow_id, snapshot)

    registry.save_ledger_snapshot = _counted  # type: ignore[method-assign]
    for _ in range(3):
        assert _pop_ready(runtime) == [ids["a"]]
        runtime.return_dispatch(ids["a"], None, increment_retry=False, front=False)

    assert saves == []


class _Probe:
    """Control's own read of a stored object, answering as told."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.reads: list[ContentReference] = []

    def __call__(self, reference: ContentReference) -> None:
        self.reads.append(reference)
        if self.error is not None:
            raise self.error


class _Attributing:
    """A runtime whose reads of held inputs answer through a probe, reporting its
    failures through an event monitor as the server wires it."""

    def __init__(self, registry: Any = None) -> None:
        schedulers: list[StoreRedriveScheduler] = []

        def _scheduler(fire: Any, logger: logging.Logger) -> StoreRedriveScheduler:
            scheduler = StoreRedriveScheduler(
                fire, logger, base_delay_sec=0, run_thread=False
            )
            schedulers.append(scheduler)
            return scheduler

        reader = make_result_reader()
        self.probe = _Probe()
        reader.verify = self.probe  # type: ignore[method-assign]
        self.runtime = TaskRuntime(
            cast(Any, registry or _Registry()),
            cast(Any, _WorkerRegistryStub()),
            OrchestrationConfig(),
            reader,
            logging.getLogger("input-unavailable"),
            credential_vault=InMemoryCredentialVault(),
            redrive=_scheduler,
        )
        self.scheduler = schedulers[0]
        self.monitor = _monitor(self.runtime)
        self.metrics = cast(MagicMock, self.monitor._metrics)
        self.runtime.set_failure_reporter(self.monitor.handle_task_event)

    def report(self, event: TaskEvent) -> None:
        self.monitor.handle_task_event(event)

    def ready(self) -> str | None:
        with self.runtime._cv:
            return self.runtime._ready.pop_ready_locked()


@pytest.mark.anyio
async def test_an_unreachable_input_returns_the_task_without_spending_an_attempt() -> (
    None
):
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)

    fixture.report(_unavailable(task_id, [reference]))

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert fixture.ready() is None
    fixture.scheduler.run_due()
    assert fixture.probe.reads == [reference]


@pytest.mark.anyio
async def test_input_control_reads_fine_runs_again_blaming_no_worker() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)

    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.PENDING
    assert record.attempts == 0
    assert record.failed_workers == []
    assert task_id not in runtime._inputs.input_checks
    assert fixture.ready() == task_id


@pytest.mark.anyio
async def test_input_missing_at_control_fails_the_task_as_a_reported_failure() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.probe.error = ResultUnreadable("no content")

    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.status == TaskStatus.FAILED
    assert (record.error or "").startswith("input_unreadable:")
    assert record.attempts == 0
    assert record.failed_workers == []
    fixture.scheduler.run_due()
    assert task_id not in runtime._inputs.input_checks
    dependent = next(
        other for other, deps in runtime._original_deps.items() if task_id in deps
    )
    assert runtime._tasks[dependent].status == TaskStatus.FAILED
    finalized = [
        c.args[0] for c in fixture.metrics.finalize_task_failure.call_args_list
    ]
    assert finalized == [task_id, dependent]


@pytest.mark.anyio
async def test_a_v2_input_missing_at_control_keeps_the_returned_attempt() -> None:
    fixture = _Attributing(FakeRegistry())
    runtime = fixture.runtime
    workflow_id, task_id, reference = await _v2_consumer(runtime)
    fixture.probe.error = ResultUnreadable("no content")

    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    assert runtime._tasks[task_id].status == TaskStatus.FAILED
    engine = runtime._engines[workflow_id]
    work_item = engine.work_item(task_id)
    assert work_item is not None
    assert work_item.status is WorkItemStatus.SETTLED
    assert [engine._ledger.attempts[a].status for a in work_item.attempt_ids] == [
        AttemptStatus.RETURNED
    ]


@pytest.mark.anyio
async def test_a_worker_cannot_report_its_own_input_unreadable() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.probe.error = ResultUnavailable("store down")
    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    fixture.report(
        _failure(
            task_id,
            "wkr-1",
            "dsp-1",
            failure_kind=TaskFailureKind.INPUT_UNREADABLE,
            retryable=False,
        )
    )

    assert runtime._tasks[task_id].status == TaskStatus.PENDING
    assert task_id in runtime._inputs.input_checks


@pytest.mark.anyio
async def test_a_store_control_cannot_reach_either_holds_the_task() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.probe.error = ResultUnavailable("store down")

    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    assert runtime._tasks[task_id].status == TaskStatus.PENDING
    assert fixture.ready() is None
    assert fixture.scheduler.pending(runtime._tasks[task_id].workflow_id)

    fixture.probe.error = None
    fixture.scheduler.run_due()

    record = runtime._tasks[task_id]
    assert record.failed_workers == []
    assert record.attempts == 0
    assert fixture.ready() == task_id


@pytest.mark.anyio
async def test_an_unexpected_read_error_holds_the_task_for_another_drive() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.probe.error = PermissionError(13, "Permission denied")

    fixture.report(_unavailable(task_id, [reference]))
    fixture.scheduler.run_due()

    workflow_id = runtime._tasks[task_id].workflow_id
    assert fixture.ready() is None
    assert fixture.scheduler.pending(workflow_id)

    fixture.probe.error = None
    fixture.scheduler.run_due()
    assert fixture.ready() == task_id


@pytest.mark.anyio
async def test_a_check_that_fails_to_apply_runs_again_without_stalling_the_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    workflow_id = runtime._tasks[task_id].workflow_id
    fixture.report(_unavailable(task_id, [reference]))
    drives: list[str] = []
    monkeypatch.setattr(runtime, "_redrive_workflow", drives.append)
    commit = runtime._committer.commit_locked
    failures = [RuntimeError("redis down")]

    def _flaky_commit(*task_ids: str) -> None:
        if failures:
            raise failures.pop()
        commit(*task_ids)

    monkeypatch.setattr(runtime._committer, "commit_locked", _flaky_commit)
    fixture.scheduler.run_due()

    assert drives == [workflow_id]
    assert task_id in runtime._inputs.input_checks
    assert fixture.scheduler.pending(workflow_id)

    fixture.scheduler.run_due()
    assert task_id not in runtime._inputs.input_checks


@pytest.mark.anyio
async def test_a_report_naming_inputs_the_task_does_not_consume_is_charged() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, _ = await _consumer(runtime)
    foreign = reference_for("other-org", b"secret", media_type="application/json")
    unrelated = reference_for("org", b"unrelated", media_type="application/json")

    fixture.report(_unavailable(task_id, [foreign, unrelated]))
    fixture.scheduler.run_due()

    record = runtime._tasks[task_id]
    assert fixture.probe.reads == []
    assert task_id not in runtime._inputs.input_checks
    assert record.attempts == 1
    assert record.failed_workers == ["wkr-1"]


@pytest.mark.anyio
async def test_a_task_cancelled_while_held_is_left_settled() -> None:
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.report(_unavailable(task_id, [reference]))

    runtime.cancel_workflow(runtime._tasks[task_id].workflow_id)
    assert task_id not in runtime._inputs.input_checks
    fixture.scheduler.run_due()

    assert runtime._tasks[task_id].status == TaskStatus.CANCELLED
    assert fixture.ready() is None


class _Stream:
    """The durable task-event stream as the monitor consumes it: in order, handing a
    report over again until one handling of it completes."""

    def __init__(self, monitor: Any) -> None:
        self._monitor = monitor
        self.entries: list[TaskEvent] = []

    def publish(self, event: TaskEvent) -> None:
        self.entries.append(event)

    def pump(self) -> None:
        while self.entries:
            try:
                self._monitor.handle_task_event(self.entries[0])
            except ConnectionError:
                return
            self.entries.pop(0)


def _streamed(registry: _Registry) -> tuple[_Attributing, _Stream]:
    fixture = _Attributing(registry)
    stream = _Stream(fixture.monitor)
    fixture.runtime.set_failure_reporter(stream.publish)
    return fixture, stream


def _finalized(fixture: _Attributing) -> list[str]:
    return [c.args[0] for c in fixture.metrics.finalize_task_failure.call_args_list]


@pytest.mark.anyio
async def test_a_verdict_whose_write_failed_is_handled_again_and_finalizes_once() -> (
    None
):
    registry = _Registry()
    fixture, stream = _streamed(registry)
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    fixture.report(_unavailable(task_id, [reference]))
    fixture.probe.error = ResultUnreadable("no content")
    fixture.scheduler.run_due()

    registry.fail_next = True
    stream.pump()
    assert registry.durable_status(task_id) != TaskStatus.FAILED
    assert task_id in runtime._inputs.input_checks
    fixture.scheduler.run_due()  # control reports its verdict again
    stream.pump()

    assert registry.durable_status(task_id) == TaskStatus.FAILED
    assert _finalized(fixture).count(task_id) == 1
    fixture.scheduler.run_due()
    assert task_id not in runtime._inputs.input_checks


@pytest.mark.anyio
async def test_a_verdict_applied_directly_after_a_failed_write_is_reported_again() -> (
    None
):
    registry = _Registry()
    fixture = _Attributing(registry)
    runtime = fixture.runtime
    redis = MagicMock()
    redis.xadd_telemetry.side_effect = ConnectionError("telemetry redis down")
    publisher = TaskEventPublisher(redis, logging.getLogger("input-unavailable"))
    publisher.set_fallback(fixture.monitor.handle_task_event)
    runtime.set_failure_reporter(publisher.publish)
    task_id, reference = await _consumer(runtime)
    fixture.report(_unavailable(task_id, [reference]))
    fixture.probe.error = ResultUnreadable("no content")

    registry.fail_next = True
    fixture.scheduler.run_due()
    assert registry.durable_status(task_id) != TaskStatus.FAILED
    fixture.scheduler.run_due()

    assert registry.durable_status(task_id) == TaskStatus.FAILED
    assert _finalized(fixture).count(task_id) == 1


@pytest.mark.anyio
async def test_a_verdict_is_never_answered_with_the_workers_stashed_report() -> None:
    registry = _Registry()
    fixture, stream = _streamed(registry)
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    report = _unavailable(task_id, [reference])
    registry.fail_next = True
    with pytest.raises(ConnectionError):
        fixture.report(report)  # the stream hands this over again later
    fixture.probe.error = ResultUnreadable("no content")

    fixture.scheduler.run_due()
    stream.pump()  # the verdict, handled before the worker's report comes back

    assert runtime._tasks[task_id].status == TaskStatus.FAILED
    assert _finalized(fixture).count(task_id) == 1
    fixture.report(report)
    assert registry.durable_status(task_id) == TaskStatus.FAILED
    assert _finalized(fixture).count(task_id) == 1


@pytest.mark.anyio
async def test_a_verdict_applied_directly_supersedes_the_workers_stashed_report() -> (
    None
):
    registry = _Registry()
    fixture = _Attributing(registry)
    runtime = fixture.runtime
    redis = MagicMock()
    redis.xadd_telemetry.side_effect = ConnectionError("telemetry redis down")
    publisher = TaskEventPublisher(redis, logging.getLogger("input-unavailable"))
    publisher.set_fallback(fixture.monitor.handle_task_event)
    runtime.set_failure_reporter(publisher.publish)
    task_id, reference = await _consumer(runtime)
    report = _unavailable(task_id, [reference])
    registry.fail_next = True
    with pytest.raises(ConnectionError):
        fixture.report(report)
    fixture.probe.error = ResultUnreadable("no content")
    fixture.scheduler.run_due()
    handled = len(fixture.metrics.record_task_event.call_args_list)

    fixture.report(report)  # the stream hands the worker's report over again

    later = fixture.metrics.record_task_event.call_args_list[handled:]
    assert [c.args[0].type for c in later if c.args[0].task_id == task_id] == []
    assert registry.durable_status(task_id) == TaskStatus.FAILED
    assert _finalized(fixture).count(task_id) == 1


@pytest.mark.anyio
async def test_reporting_a_verdict_again_does_not_count_toward_the_store_backoff() -> (
    None
):
    fixture = _Attributing()
    runtime = fixture.runtime
    task_id, reference = await _consumer(runtime)
    workflow_id = runtime._tasks[task_id].workflow_id
    fixture.report(_unavailable(task_id, [reference]))
    fixture.probe.error = ResultUnreadable("no content")

    fixture.scheduler.run_due()

    assert fixture.scheduler.pending(workflow_id)
    assert fixture.scheduler._streak.get(workflow_id, 0) == 0


@pytest.mark.anyio
async def test_verdicts_reported_again_while_the_monitor_lags_back_off() -> None:
    registry = _Registry()
    fixture, stream = _streamed(registry)
    runtime = fixture.runtime
    now = [0.0]
    fixture.scheduler._clock = lambda: now[0]
    fixture.scheduler._base = 1.0
    fixture.scheduler._max = 30.0
    task_id, reference = await _consumer(runtime)
    workflow_id = runtime._tasks[task_id].workflow_id
    fixture.report(_unavailable(task_id, [reference]))
    fixture.probe.error = ResultUnreadable("no content")

    while now[0] <= 120:  # the monitor handles nothing for two minutes
        fixture.scheduler.run_due()
        now[0] += 0.5

    assert 1 < len(stream.entries) < 12
    assert fixture.scheduler._streak.get(workflow_id, 0) == 0
    stream.pump()
    now[0] += 60
    fixture.scheduler.run_due()
    assert _finalized(fixture).count(task_id) == 1
    assert task_id not in runtime._inputs.input_checks
    assert workflow_id not in fixture.scheduler._recheck_streak
