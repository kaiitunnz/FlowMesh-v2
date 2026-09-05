"""The off-lane worker executor for a worker-originated mediated tool operation."""

from pathlib import Path
from typing import Any

import pytest

from shared.tools.contract import (
    MediatedOperationPermit,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)
from shared.utils.ids import new_mediated_permit_id
from tests.shared.outcome_helpers import InMemoryContentStore
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import pending_tool_request, tool_operation_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.tool_operation_executor import ToolOperationExecutor

_WORKER = "wkr-1"
_GEN = 3
_AGENT = "tsk-agent"
_CALL = "m0"
_REQUEST = ToolRequest(interface=SEARCH_INTERFACE, query="weather", max_results=3)


class _Client:
    worker_id = _WORKER
    incarnation = _GEN


class _Lifecycle:
    client = _Client()


class _Sidecar:
    def __init__(self, outcome: ToolOutcome) -> None:
        self._outcome = outcome
        self.calls = 0

    def execute(self, envelope: Any, request: Any) -> ToolOutcome:
        self.calls += 1
        return self._outcome


def _permit(**overrides: Any) -> MediatedOperationPermit:
    fields: dict[str, Any] = {
        "permit_id": new_mediated_permit_id(),
        "agent_task_id": _AGENT,
        "call_correlation": _CALL,
        "interface": SEARCH_INTERFACE,
        "subject": SEARCH_INTERFACE,
        "invocation_id": "inv-1",
        "idempotency_key": "idm-1",
        "request_digest": tool_request_digest(SEARCH_INTERFACE, "weather", 3),
        "target_id": _WORKER,
        "target_generation": _GEN,
        "deadline_epoch": 2_000_000_000.0,
        "max_results": 5,
        "timeout_sec": 10.0,
        "result_char_cap": 4000,
    }
    fields.update(overrides)
    return MediatedOperationPermit(**fields)


def _executor(outcome: ToolOutcome | None = None) -> ToolOperationExecutor:
    ex = ToolOperationExecutor(make_worker_config(), None, _Lifecycle())
    if outcome is not None:
        ex._sidecar = _Sidecar(outcome)  # type: ignore[assignment]
    return ex


def _task(permit: MediatedOperationPermit | None) -> Any:
    from shared.tasks import TaskType
    from shared.tasks.specs import ToolOperationSpecStrict

    return make_worker_task_message(
        ToolOperationSpecStrict(taskType=TaskType.TOOL_OPERATION),
        task_type=TaskType.TOOL_OPERATION,
        tool_operation=permit,
    )


def _stash() -> None:
    pending_tool_request.put(_AGENT, _CALL, _REQUEST)


def test_fails_closed_without_a_permit(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="without a permit"):
        _executor().run(_task(None), tmp_path)


def test_fails_when_no_worker_private_request(tmp_path: Path) -> None:
    pending_tool_request.take(_AGENT, _CALL)  # ensure absent
    with pytest.raises(ExecutionError, match="no worker-private request"):
        _executor().run(_task(_permit()), tmp_path)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"request_digest": "deadbeef"}, "digest"),
        ({"target_id": "wkr-other"}, "audience"),
        ({"target_generation": 99}, "generation"),
        ({"deadline_epoch": 1.0}, "expired"),
        ({"interface": "other/v1"}, "interface"),
    ],
)
def test_fence_rejects_a_tampered_permit(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    _stash()
    with pytest.raises(ExecutionError, match=reason):
        _executor().run(_task(_permit(**overrides)), tmp_path)
    # A rejected operation never egresses and leaves no residual request.
    assert pending_tool_request.peek(_AGENT, _CALL) is None


def test_non_success_outcome_is_inline(tmp_path: Path) -> None:
    _stash()
    out = ToolOutcome(status=ToolOutcomeStatus.QUOTA, value="budget exhausted")
    result = _executor(out).run(_task(_permit()), tmp_path)
    assert result.outcome == out and result.outcome_ref is None


def test_successful_result_is_returned_by_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stash()
    store = InMemoryContentStore()
    monkeypatch.setattr(
        tool_operation_executor, "build_content_store", lambda _url: store
    )
    out = ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="results...")
    result = _executor(out).run(_task(_permit()), tmp_path)
    assert result.outcome is None
    assert result.outcome_ref is not None
    assert result.outcome_ref.idempotency_key == "idm-1"


def test_redrive_after_materialize_recovers_the_prior_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A crash between materialize and the recorded settle re-dispatches the op; its
    # worker-private request is already consumed, but a prior materialization under the
    # same idempotency key must be recovered rather than failing the boundary clean.
    _stash()
    store = InMemoryContentStore()
    monkeypatch.setattr(
        tool_operation_executor, "build_content_store", lambda _url: store
    )
    out = ToolOutcome(status=ToolOutcomeStatus.SUCCESS, value="results...")
    first = _executor(out)
    ref = first.run(_task(_permit()), tmp_path).outcome_ref
    assert ref is not None
    assert pending_tool_request.peek(_AGENT, _CALL) is None  # request consumed

    # A fresh op (no worker-private request, a fresh sidecar that must not be called)
    # recovers the prior outcome by reference.
    redrive = _executor()
    redrive._sidecar = _Sidecar(  # type: ignore[assignment]
        ToolOutcome(status=ToolOutcomeStatus.UNAVAILABLE, value="must not egress")
    )
    result = redrive.run(_task(_permit()), tmp_path)
    assert result.outcome is None
    assert result.outcome_ref is not None
    assert result.outcome_ref.content_digest == ref.content_digest
    assert redrive._sidecar.calls == 0  # type: ignore[attr-defined]
