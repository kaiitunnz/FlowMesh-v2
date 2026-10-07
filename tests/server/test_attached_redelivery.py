"""A worker whose task stream attaches is sent again what a detached stream may have
lost: its pending mediated operations, re-minted, and the interrupts of the tasks it is
cancelling, keyed to their dispatches."""

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from google.protobuf.empty_pb2 import Empty

from server.task.models import TaskStatus
from shared.schemas.event import parse_event
from shared.tools.contract import (
    MediatedOperationOutcome,
    ToolOutcome,
    ToolOutcomeStatus,
)
from tests.server.dispatch_helpers import record_dispatch
from tests.server.servicer_helpers import ServicerHarness, WorkerContext
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_v2_orchestration import LINEAR, FakeRegistry
from tests.server.task.test_v2_orchestration import _register as _register_v2
from tests.server.task.test_v2_orchestration import _runtime as _runtime_v2
from tests.server.task.test_worker_originated_boundary import (
    _SEARCH_WF,
    _dispatch_agent,
    _permit_frames,
    _register,
    _runtime,
)
from tests.support.waiting import pop_ready


def test_an_attached_worker_gets_its_pending_operation_re_minted() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer)
        [first] = _permit_frames(runtime)

        runtime.redeliver_to_worker("wkr-1")

        permits = _permit_frames(runtime)
        assert len(permits) == 2
        again = permits[1]
        assert again["permit_id"] != first["permit_id"]
        assert (again["agent_task_id"], again["call_correlation"]) == (
            first["agent_task_id"],
            first["call_correlation"],
        )
        assert list(runtime._mediated_ops.pending_ops) == [again["permit_id"]]

    asyncio.run(run())


def test_the_first_permit_s_outcome_settles_its_re_mint_too() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer)
        [first] = _permit_frames(runtime)
        runtime.redeliver_to_worker("wkr-1")

        runtime.settle_mediated_operation(
            MediatedOperationOutcome(
                permit_id=first["permit_id"],
                agent_task_id=writer,
                call_correlation=first["call_correlation"],
                invocation_id=first["invocation_id"],
                idempotency_key=first["idempotency_key"],
                outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="sunny"),
            )
        )

        assert runtime._mediated_ops.pending_ops == {}

    asyncio.run(run())


def test_another_worker_attaching_re_relays_nothing() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        _dispatch_agent(runtime, ids["writer"])

        runtime.redeliver_to_worker("wkr-2")

        assert len(_permit_frames(runtime)) == 1

    asyncio.run(run())


def test_an_attached_worker_is_interrupted_again_for_its_cancelling_task() -> None:
    registry = MagicMock()
    runtime = _runtime_v2(FakeRegistry(), registry)
    workflow_id, ids = asyncio.run(_register_v2(runtime, LINEAR))
    task_id = ids["a"]
    assert pop_ready(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    runtime.cancel_workflow(workflow_id)
    assert runtime._tasks[task_id].status == TaskStatus.CANCELLING
    registry.publish_interrupt.reset_mock()

    runtime.redeliver_to_worker("wkr-1")

    [(_worker, interrupt)] = [c.args for c in registry.publish_interrupt.call_args_list]
    assert (interrupt.task_id, interrupt.dispatch_id) == (task_id, "dsp-1")


@pytest.mark.asyncio
async def test_a_task_stream_attaching_tells_the_root() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    listener = cast(MagicMock, harness.servicer._task_listener)
    listener.attach_stream.return_value = MagicMock(next=_closed_stream)
    async for _ in harness.servicer.StreamTasks(Empty(), cast(Any, WorkerContext())):
        pass

    runtime = MagicMock()
    monitor = _monitor(cast(Any, runtime))
    for relayed in harness.relay.events:
        monitor._handle_worker_event(cast(Any, parse_event(relayed)))

    runtime.redeliver_to_worker.assert_called_once_with(worker_id)


async def _closed_stream() -> None:
    return None
