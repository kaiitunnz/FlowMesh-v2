"""A workflow's closure, made or found made, is reported to the runtime; a close that
cannot be written is retried."""

import asyncio
import logging
from typing import Any, cast

from server.clients.redis import workflow_log_closed_key
from server.task.finalizer import CloseOutcome, WorkflowFinalizer
from server.task.runtime import TaskRuntime
from server.task.workflow_retry import WorkflowRetryScheduler
from tests.server.services.test_workflow_span_close import (
    _RecordingWorkflowSpanEmitter,
)
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _drain,
    _register,
    _runtime,
)
from tests.server.task.test_workflow_finalizer import (
    _CHAIN,
    _TS,
    _RedisMirroringRemainingSet,
)


class _Redis(_RedisMirroringRemainingSet):
    """A Redis stand-in whose log-stream writes can be made to fail."""

    def __init__(self, registry: FakeRegistry) -> None:
        super().__init__(registry)
        self.refuse_writes = False

    def xadd_telemetry(self, *args: Any, **kwargs: Any) -> None:
        if self.refuse_writes:
            raise ConnectionError("telemetry unavailable")


def _closing(
    settle: bool = True,
) -> tuple[TaskRuntime, str, WorkflowFinalizer, _Redis, WorkflowRetryScheduler]:
    registry = FakeRegistry()
    registry.submitted_at = _TS
    runtime = _runtime(registry)
    workflow_id, _ = asyncio.run(_register(runtime, _CHAIN))
    if settle:
        _drain(runtime)
    redis = _Redis(registry)
    redis.keys[f"workflow:{workflow_id}"] = "1"
    retries: list[WorkflowRetryScheduler] = []

    def retry(fire: Any, logger: logging.Logger) -> WorkflowRetryScheduler:
        retries.append(
            WorkflowRetryScheduler(fire, logger, base_delay_sec=0.0, run_thread=False)
        )
        return retries[0]

    finalizer = WorkflowFinalizer(
        redis_client=cast(Any, redis),
        runtime=runtime,
        logger=logging.getLogger("test.finalizer-outcome"),
        workflow_span_emitter=cast(Any, _RecordingWorkflowSpanEmitter()),
        retry=retry,
    )
    return runtime, workflow_id, finalizer, redis, retries[0]


def _closed(runtime: TaskRuntime, workflow_id: str) -> bool:
    with runtime._lock:
        return workflow_id in runtime.settled_history._closed


def test_a_closed_workflow_reports_its_closure() -> None:
    runtime, workflow_id, finalizer, _, _ = _closing()

    assert finalizer.close(workflow_id) is CloseOutcome.CLOSED
    assert _closed(runtime, workflow_id)
    assert finalizer.close(workflow_id) is CloseOutcome.ALREADY_CLOSED


def test_a_closure_found_made_is_reported() -> None:
    runtime, workflow_id, finalizer, redis, _ = _closing()
    redis.keys[workflow_log_closed_key(workflow_id)] = "1"

    assert finalizer.close(workflow_id) is CloseOutcome.ALREADY_CLOSED
    assert _closed(runtime, workflow_id)


def test_a_workflow_still_running_reports_nothing() -> None:
    runtime, workflow_id, finalizer, _, _ = _closing(settle=False)

    assert finalizer.close(workflow_id) is CloseOutcome.NOT_COMPLETE
    assert not _closed(runtime, workflow_id)


def test_a_close_that_cannot_be_written_is_retried_until_it_is() -> None:
    runtime, workflow_id, finalizer, redis, retry = _closing()
    redis.refuse_writes = True

    assert finalizer.close(workflow_id) is CloseOutcome.FAILED
    assert not _closed(runtime, workflow_id)
    assert retry.pending(workflow_id)

    redis.refuse_writes = False
    assert retry.run_due() == [workflow_id]
    finalizer.drain()
    assert _closed(runtime, workflow_id)
    assert not retry.pending(workflow_id)
