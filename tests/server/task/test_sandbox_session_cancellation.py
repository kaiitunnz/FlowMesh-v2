"""A cancelled sandbox session releases the host capacity its claim admitted.

A session holds its host claim on its own work item rather than a mediated boundary, so
the boundary-invocation sweep never covers it: cancellation has to release it from the
same fenced terminal, or the family fills with credit no session can use.
"""

import asyncio
from typing import Any

from server.task.models import TaskStatus
from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    HarnessBackendKey,
    HarnessCapsule,
    HarnessResult,
    HarnessResultKind,
)
from tests.server.task.test_v2_orchestration import FakeRegistry, _register, _runtime

_TS = "2026-09-11T00:00:00Z"

_SANDBOX_WF = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: sandbox-cancel}
spec:
  graph:
    nodes:
      - name: session
        spec:
          taskType: sandbox
          sandbox: {profile: posix-default}
          commands:
            - argv: ["sh", "-c", "echo one"]
            - argv: ["sh", "-c", "echo two"]
"""


def _released(runtime: Any) -> list[str]:
    """Record the invocations whose resident credit the runtime released."""
    released: list[str] = []
    runtime.set_resident_terminal_hook(
        lambda invocation_id, failed: released.append(invocation_id)
    )
    return released


def _session(runtime: Any, tasks: dict[str, str]) -> tuple[str, str]:
    """The session task and the invocation identity its host claim links to."""
    task_id = tasks["session"]
    engine = runtime.orchestration_engine(runtime._tasks[task_id].workflow_id)
    invocation_id = engine.ensure_invocation(task_id)
    return task_id, invocation_id


def test_cancelling_a_workflow_releases_its_session_host_credit() -> None:
    runtime = _runtime(FakeRegistry())
    workflow_id, tasks = asyncio.run(_register(runtime, _SANDBOX_WF))
    released = _released(runtime)
    task_id, invocation_id = _session(runtime, tasks)

    runtime.cancel_workflow(workflow_id, reason="cancelled")

    assert runtime._tasks[task_id].status is TaskStatus.CANCELLED
    # The session's admission is settled from the cancellation terminal, not stranded.
    assert invocation_id in released


def test_cancelling_a_dispatched_session_releases_its_host_credit() -> None:
    runtime = _runtime(FakeRegistry())
    _workflow_id, tasks = asyncio.run(_register(runtime, _SANDBOX_WF))
    released = _released(runtime)
    task_id, invocation_id = _session(runtime, tasks)

    runtime.mark_cancelled(task_id, "wkr-1", {"finished_at": _TS}, _TS)

    assert runtime._tasks[task_id].status is TaskStatus.CANCELLED
    assert invocation_id in released


def test_a_step_reported_after_cancellation_settles_the_record() -> None:
    """A worker finishing a command after the cancel must not leave the task pending."""
    runtime = _runtime(FakeRegistry())
    workflow_id, tasks = asyncio.run(_register(runtime, _SANDBOX_WF))
    task_id, _invocation_id = _session(runtime, tasks)
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    engine.on_dispatched(task_id, "wkr-1")

    runtime.cancel_workflow(workflow_id, reason="cancelled")
    runtime._tasks[task_id].status = TaskStatus.DISPATCHED  # the step was still running

    # The command completes and yields, as a mid-session step does. The real caller
    # holds the runtime's condition while it applies a step.
    with runtime._cv:
        runtime._apply_episode_step_locked(
            task_id,
            HarnessResult(
                kind=HarnessResultKind.YIELD,
                request=BoundaryRequest(kind=BoundaryEventKind.YIELD),
                capsule=HarnessCapsule(
                    backend=HarnessBackendKey(backend="sandbox_session", version="1"),
                    blob="[]",
                ),
            ),
        )

    # The ledger closed the work item, so the step re-readies nothing; the record must
    # still reach a terminal rather than waiting on a lane that will never run.
    assert runtime._tasks[task_id].status is TaskStatus.CANCELLED
