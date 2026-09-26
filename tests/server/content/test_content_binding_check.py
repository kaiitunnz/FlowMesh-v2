"""What a task's own bindings entitle its worker to read."""

import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.content import reference_for
from shared.harness.adapter import DeliveredOutcome
from shared.outcome import OutcomeManifest

_PREPARED = reference_for("local", b"prepared", media_type="application/json")
_DELIVERED = reference_for("local", b"delivered", media_type="application/json")
_UNRELATED = reference_for("local", b"unrelated", media_type="application/json")


class _Engine:
    """An engine with only the two bindings the check reads."""

    def __init__(self, resolution: Any = None, delivered: Any = ()) -> None:
        self._resolution = resolution
        self._delivered = delivered

    def input_resolution(self, task_id: str) -> Any:
        return self._resolution

    def episode_context(self, task_id: str) -> tuple[None, Any]:
        return None, self._delivered


def _runtime(
    *,
    assigned: str,
    resolution: Any = None,
    delivered: Any = (),
    status: str = TaskStatus.DISPATCHED,
) -> TaskRuntime:
    """A runtime with only what the binding check reads."""
    runtime = object.__new__(TaskRuntime)
    runtime._lock = threading.RLock()
    runtime._publishing = {}
    runtime._tasks = cast(
        Any,
        {
            "tsk-1": SimpleNamespace(
                task_id="tsk-1",
                assigned_worker=assigned,
                dispatch_id=None,
                status=status,
                workflow_id="wfl-1",
            )
        },
    )
    runtime._engines = cast(Any, {"wfl-1": _Engine(resolution, delivered)})
    return runtime


def test_the_worker_running_a_prepared_task_may_read_its_request() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)


def test_another_worker_may_not_read_it() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-9", _PREPARED)


def test_a_reference_the_task_is_not_bound_to_is_refused() -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED)
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_an_outcome_the_engine_delivered_may_be_read() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        delivered=(
            DeliveredOutcome(
                call_correlation="c1",
                outcome_ref=OutcomeManifest(content=_DELIVERED),
            ),
        ),
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _DELIVERED)
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _UNRELATED)


def test_an_unknown_task_authorizes_nothing() -> None:
    runtime = _runtime(assigned="wkr-2")
    assert not runtime.content_binding_authorizes("tsk-absent", "wkr-2", _PREPARED)


@pytest.mark.parametrize(
    "status", [TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
def test_the_worker_of_a_settled_task_may_not_read_it(status: str) -> None:
    runtime = _runtime(
        assigned="wkr-2", resolution=SimpleNamespace(reference=_PREPARED), status=status
    )
    assert not runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)


def test_the_worker_winding_down_a_cancelling_task_may_read_it() -> None:
    runtime = _runtime(
        assigned="wkr-2",
        resolution=SimpleNamespace(reference=_PREPARED),
        status=TaskStatus.CANCELLING,
    )
    assert runtime.content_binding_authorizes("tsk-1", "wkr-2", _PREPARED)
