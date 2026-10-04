"""A v2 task whose executor failed after its external effect may have happened settles
by its effect's replay contract, as on its worker's loss: it runs again only when the
effect is safe to replay. A v1 task retries as any retryable failure does."""

from typing import Any, cast

import pytest

from server.orchestration.state import AttemptStatus
from server.task.models import DispatchEnd, TaskStatus
from server.task.runtime import TaskRuntime
from server.task.v2.representations.operators import EffectReplayContract
from shared.schemas.event import TaskEvent
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_unrequested_cancel import _next
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _runtime,
    _worker,
)

_ERROR = "API request failed: timed out"


def _api_then_echo(api_version: str = "flowmesh/v2", v2: str = "") -> str:
    return f"""
apiVersion: {api_version}
kind: Workflow
metadata: {{name: api-then-echo}}
spec:
  graph:
    nodes:
      - name: call
        spec:
          taskType: api
          api: {{url: 'http://api.example/v1', method: POST}}
          {v2}
      - name: after
        dependsOn: [call]
        spec: {{taskType: echo, data: {{type: list, items: [after]}}}}
"""


def _failed(task_id: str, *, ambiguous: bool, dispatch_id: str = "dsp-1") -> TaskEvent:
    return TaskEvent(
        type="TASK_FAILED",
        task_id=task_id,
        worker_id="wkr-1",
        dispatch_id=dispatch_id,
        error=_ERROR,
        retryable=True,
        ambiguous=ambiguous,
        ts="2026-06-01T00:00:00Z",
    )


async def _dispatched(runtime: TaskRuntime, payload: str) -> tuple[str, dict[str, str]]:
    workflow_id, ids = await _register(runtime, payload)
    assert _next(runtime) == ids["call"]
    record_dispatch(runtime, ids["call"], cast(Any, _worker()), "dsp-1")
    return workflow_id, ids


def _fail(runtime: TaskRuntime, task_id: str, *, ambiguous: bool) -> DispatchEnd:
    return runtime.fail_dispatch(
        task_id,
        "wkr-1",
        {},
        "2026-06-01T00:00:00Z",
        "dsp-1",
        error=_ERROR,
        retryable=True,
        ambiguous=ambiguous,
    ).end


@pytest.mark.anyio
async def test_an_ambiguous_api_failure_fails_once_with_its_dependents() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    workflow_id, ids = await _dispatched(runtime, _api_then_echo())

    monitor.handle_task_event(_failed(ids["call"], ambiguous=True))

    for rt in (runtime, _runtime(registry)):
        if rt is not runtime:
            await rt.rehydrate()
        call = rt.get_record(ids["call"])
        after = rt.get_record(ids["after"])
        assert call is not None and call.status == TaskStatus.FAILED
        assert call.error == f"{_ERROR} (ambiguity-terminal effect)"
        assert call.attempts == 0
        assert after is not None and after.status == TaskStatus.FAILED
        assert after.error == f"Dependency {ids['call']} failed"
        assert rt.ready_queue_length() == 0
        assert rt.workflow_settlement(workflow_id).settled
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    (attempt,) = engine.to_snapshot().attempts[:1]
    assert attempt.status is AttemptStatus.LOST and attempt.error == _ERROR
    assert ("invocation_ambiguity_terminal", "") in {
        (kind, "") for kind, _subject in engine.contract_trace()
    }


@pytest.mark.anyio
async def test_a_replayed_ambiguous_failure_settles_once() -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    monitor = _monitor(runtime)
    _, ids = await _dispatched(runtime, _api_then_echo())

    event = _failed(ids["call"], ambiguous=True)
    monitor.handle_task_event(event)
    monitor.handle_task_event(event)
    restored = _runtime(registry)
    await restored.rehydrate()
    restored_monitor = _monitor(restored)
    restored_monitor.handle_task_event(event)

    derived = [
        call.args[0]
        for each in (monitor, restored_monitor)
        for call in cast(Any, each._metrics).record_task_event.call_args_list
        if call.args[0].type == "TASK_FAILED" and call.args[0].task_id == ids["after"]
    ]
    assert len(derived) == 1
    for rt in (runtime, restored):
        after = rt.get_record(ids["after"])
        assert after is not None and after.status == TaskStatus.FAILED


@pytest.mark.anyio
async def test_an_api_leaf_its_author_marks_pure_runs_again() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _dispatched(runtime, _api_then_echo(v2="v2: {effect: pure}"))

    assert _fail(runtime, ids["call"], ambiguous=True) is DispatchEnd.RETURNED

    record = runtime.get_record(ids["call"])
    assert record is not None and record.status == TaskStatus.PENDING
    assert record.attempts == 1
    assert _next(runtime) == ids["call"]


@pytest.mark.anyio
async def test_a_pure_leaf_ambiguous_on_its_last_attempt_fails_as_reported() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _dispatched(runtime, _api_then_echo(v2="v2: {effect: pure}"))
    record = runtime.get_record(ids["call"])
    assert record is not None
    record.max_attempts = 1

    assert _fail(runtime, ids["call"], ambiguous=True) is DispatchEnd.FAILED

    assert record.status == TaskStatus.FAILED and record.error == _ERROR


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("contract", "end", "event"),
    [
        (EffectReplayContract.REPLAYABLE_DEDUP, DispatchEnd.RETURNED, None),
        (
            EffectReplayContract.COMPENSABLE,
            DispatchEnd.FAILED,
            "invocation_compensation_required",
        ),
    ],
)
async def test_an_external_effect_settles_by_its_replay_contract(
    contract: EffectReplayContract, end: DispatchEnd, event: str | None
) -> None:
    runtime = _runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _api_then_echo())
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    work_item = engine.work_item(ids["call"])
    assert work_item is not None
    work_item.replay_contract = contract
    assert _next(runtime) == ids["call"]
    record_dispatch(runtime, ids["call"], cast(Any, _worker()), "dsp-1")

    assert _fail(runtime, ids["call"], ambiguous=True) is end

    if event is not None:
        assert event in {kind for kind, _subject in engine.contract_trace()}


@pytest.mark.anyio
async def test_a_definite_failure_of_a_v2_api_leaf_runs_again() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _dispatched(runtime, _api_then_echo())

    assert _fail(runtime, ids["call"], ambiguous=False) is DispatchEnd.RETURNED

    record = runtime.get_record(ids["call"])
    assert record is not None and record.attempts == 1


@pytest.mark.anyio
async def test_a_v1_api_task_retries_an_ambiguous_failure() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _dispatched(runtime, _api_then_echo(api_version="flowmesh/v1"))

    assert _fail(runtime, ids["call"], ambiguous=True) is DispatchEnd.RETURNED

    record = runtime.get_record(ids["call"])
    assert record is not None and record.attempts == 1
    assert record.status == TaskStatus.PENDING
