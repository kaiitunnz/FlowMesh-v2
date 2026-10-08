"""A held model turn is authorized only from durable state.

A turn proposed while its workflow's writes are held gets no frame; its authorization
waits for the writes to commit and then runs again in full, relaying one permit, or
nothing once its permit deadline passed, a later proposal superseded it, or its task
was cancelled.
"""

import asyncio
from typing import Any

from server.task.runtime import TaskRuntime
from shared.tools.contract import AgentModelTurnProposal, MediatedOperationPermit
from tests.server.task.test_runtime_commit_then_act import _runtime
from tests.server.task.test_v2_orchestration import FakeRegistry, _register
from tests.server.task.test_worker_originated_boundary import (
    _MODEL_WF,
    HELD_DISPATCH,
    _deny_frames,
    _hold_dispatch,
    _permit_frames,
)


class _Store(FakeRegistry):
    down = False

    def commit_transition(self, workflow_id: str, **kwargs: Any) -> None:
        if self.down:
            raise ConnectionError("records refused")
        super().commit_transition(workflow_id, **kwargs)


def _held_turn() -> tuple[_Store, TaskRuntime, str, str]:
    """An agent holding a model turn whose workflow owes a write."""
    store = _Store()
    runtime = _runtime(store)
    workflow_id, ids = asyncio.run(_register(runtime, _MODEL_WF))
    writer = ids["writer"]
    _hold_dispatch(runtime, writer)
    store.down = True
    with runtime._lock:
        runtime._committer.mark_dirty_locked(workflow_id)
    return store, runtime, workflow_id, writer


def _propose(runtime: TaskRuntime, writer: str, digest: str = "deadbeef") -> None:
    runtime.authorize_model_turn(
        AgentModelTurnProposal(
            agent_task_id=writer,
            call_correlation="t0",
            request_digest=digest,
            dispatch_id=HELD_DISPATCH,
        ),
        "wkr-1",
    )


def test_a_turn_proposed_while_writes_are_held_gets_no_frame() -> None:
    _, runtime, _, writer = _held_turn()

    _propose(runtime, writer)

    assert _permit_frames(runtime) == [] and _deny_frames(runtime) == []


def test_a_turn_is_authorized_once_its_writes_commit_before_its_deadline() -> None:
    store, runtime, workflow_id, writer = _held_turn()
    _propose(runtime, writer)

    store.down = False
    runtime._retry_durability(workflow_id)
    runtime._retry_durability(workflow_id)

    (permit,) = _permit_frames(runtime)
    assert MediatedOperationPermit.model_validate(permit).request_digest == "deadbeef"
    assert _deny_frames(runtime) == []


def test_a_turn_whose_deadline_passed_while_held_is_never_authorized() -> None:
    store, runtime, workflow_id, writer = _held_turn()
    runtime._mediated_ops._model_egress_timeout_sec = 0.0
    _propose(runtime, writer)

    store.down = False
    runtime._retry_durability(workflow_id)

    assert _permit_frames(runtime) == [] and _deny_frames(runtime) == []


def test_a_turn_cancelled_before_its_writes_commit_is_never_authorized() -> None:
    store, runtime, workflow_id, writer = _held_turn()
    _propose(runtime, writer)
    runtime.cancel_workflow(workflow_id)

    store.down = False
    runtime._retry_durability(workflow_id)

    assert _permit_frames(runtime) == []


def test_a_later_proposal_of_the_turn_supersedes_the_parked_one() -> None:
    store, runtime, workflow_id, writer = _held_turn()
    _propose(runtime, writer, "first")
    _propose(runtime, writer, "second")

    store.down = False
    runtime._retry_durability(workflow_id)

    (permit,) = _permit_frames(runtime)
    assert MediatedOperationPermit.model_validate(permit).request_digest == "second"


def test_a_parked_authorization_whose_relay_fails_is_retried() -> None:
    store, runtime, workflow_id, writer = _held_turn()
    _propose(runtime, writer)
    workers = runtime._worker_registry
    publish = workers.publish_mediated_op
    failures = [ConnectionError("relay down")]

    def flaky(worker: Any, payload: Any) -> int:
        if failures:
            raise failures.pop()
        return publish(worker, payload)

    setattr(workers, "publish_mediated_op", flaky)
    store.down = False
    runtime._retry_durability(workflow_id)
    assert _permit_frames(runtime) == []

    runtime._retry_durability(workflow_id)
    assert len(_permit_frames(runtime)) == 1
