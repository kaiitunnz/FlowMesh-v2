"""A worker-originated operation whose outcome never arrives, as when its egress could
not be finalized while the root restarted, is re-driven from its origin worker's
heartbeat once its permit's deadline passes, a bounded number of times."""

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

from server.orchestration.state import WorkItemStatus
from server.registries.worker import ReportOutcome
from server.task import runtime as runtime_module
from server.task.runtime import TaskRuntime
from shared.schemas.event import WorkerEvent
from shared.schemas.worker import WorkerStatus
from shared.tools.contract import (
    MediatedOperationOutcome,
    ToolOutcome,
    ToolOutcomeStatus,
)
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_worker_originated_boundary import (
    _SEARCH_WF,
    _dispatch_agent,
    _permit_frames,
    _register,
    _runtime,
)


def _heartbeat(runtime: TaskRuntime) -> None:
    monitor = _monitor(runtime)
    registry = MagicMock()
    registry.update_worker_hb.return_value = MagicMock(outcome=ReportOutcome.APPLIED)
    monitor._worker_registry = registry
    monitor._handle_worker_event(
        WorkerEvent(
            type="HEARTBEAT",
            worker_id="wkr-1",
            status=WorkerStatus.IDLE,
            payload={"ttl_sec": 120},
        )
    )


def _overdue(runtime: TaskRuntime) -> None:
    for op in runtime._pending_ops.values():
        op.redrive_at = 0.0


def _outcome(writer: str, permit: dict[str, Any]) -> MediatedOperationOutcome:
    return MediatedOperationOutcome(
        permit_id=permit["permit_id"],
        agent_task_id=writer,
        call_correlation=permit["call_correlation"],
        invocation_id=permit["invocation_id"],
        idempotency_key=permit["idempotency_key"],
        outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="sunny"),
    )


def test_an_operation_whose_outcome_never_arrives_settles_through_a_re_drive() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        engine = _dispatch_agent(runtime, writer)
        # The first permit's egress could not finalize its outcome, so none arrives.
        [first] = _permit_frames(runtime)

        _heartbeat(runtime)
        assert len(_permit_frames(runtime)) == 1

        _overdue(runtime)
        _heartbeat(runtime)

        [_, again] = _permit_frames(runtime)
        assert again["permit_id"] != first["permit_id"]
        assert again["idempotency_key"] == first["idempotency_key"]
        runtime.settle_mediated_operation(_outcome(writer, again))
        assert runtime._pending_ops == {}
        _dispatch_agent(runtime, writer)
        writer_wi = engine.work_item(writer)
        assert writer_wi is not None and writer_wi.status is WorkItemStatus.SETTLED

    asyncio.run(run())


def test_an_operation_re_driven_to_its_limit_fails_its_boundary() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        engine = _dispatch_agent(runtime, writer)

        for _ in range(runtime_module._OP_REDRIVE_LIMIT):
            _overdue(runtime)
            _heartbeat(runtime)
        assert len(_permit_frames(runtime)) == 1 + runtime_module._OP_REDRIVE_LIMIT
        _overdue(runtime)
        _heartbeat(runtime)

        assert len(_permit_frames(runtime)) == 1 + runtime_module._OP_REDRIVE_LIMIT
        assert runtime._pending_ops == {}
        assert not engine.boundary_settleable(
            writer, _permit_frames(runtime)[0]["call_correlation"]
        )

    asyncio.run(run())


def test_each_re_drive_waits_longer() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        _dispatch_agent(runtime, ids["writer"])
        [first] = _permit_frames(runtime)
        [op] = runtime._pending_ops.values()
        assert op.redrive_at == first["deadline_epoch"]

        _overdue(runtime)
        _heartbeat(runtime)

        [_, again] = _permit_frames(runtime)
        [op] = runtime._pending_ops.values()
        assert (op.redrives, op.redrive_at) == (
            1,
            again["deadline_epoch"] + runtime_module._OP_REDRIVE_BACKOFF_SEC,
        )

    asyncio.run(run())


def test_no_operation_is_re_minted_to_another_node_s_worker_under_its_id() -> None:
    async def run() -> None:
        runtime = _runtime()
        _, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        engine = _dispatch_agent(runtime, writer)
        [first] = _permit_frames(runtime)
        # A store wipe let a worker of another node register under the same id.
        cast(Any, runtime._worker_registry).node_alias = "elsewhere"

        _overdue(runtime)
        _heartbeat(runtime)
        runtime.redeliver_to_worker("wkr-1")

        assert _permit_frames(runtime) == [first]
        assert runtime._pending_ops == {}
        assert not engine.boundary_settleable(writer, first["call_correlation"])

    asyncio.run(run())
