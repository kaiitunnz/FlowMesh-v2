"""A dispatch that ends releases its worker's reservation after the runtime lock, and a
release that fails is retried at the next."""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from server.config import OrchestrationConfig
from server.registries.worker import WorkerRegistry
from server.task.runtime import TaskRuntime, TransitionNotDurable
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader
from tests.server.task.test_task_merge import _Registry
from tests.server.task.test_v2_orchestration import (
    _TS,
    LINEAR,
    FakeRegistry,
    _planned,
    _register,
    _worker,
)
from tests.support.waiting import pop_ready


def _runtime(worker_registry: Any) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, worker_registry),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("reservation-release"),
        credential_vault=InMemoryCredentialVault(),
    )


def _dispatched(worker_registry: Any) -> tuple[TaskRuntime, str]:
    runtime = _runtime(worker_registry)
    _, ids = asyncio.run(_register(runtime, LINEAR))
    task_id = ids["a"]
    assert pop_ready(runtime) == task_id
    record_dispatch(runtime, task_id, cast(Any, _worker()), "dsp-1")
    return runtime, task_id


def _succeed(runtime: TaskRuntime, task_id: str) -> None:
    runtime.mark_succeeded(
        task_id, "wkr-1", _planned(runtime, task_id, ["x"]), _TS, "dsp-1"
    )


def test_a_release_runs_after_the_runtime_lock() -> None:
    registry = MagicMock()
    runtime, task_id = _dispatched(registry)
    locked: list[bool] = []
    registry.release_worker.side_effect = lambda *_: locked.append(
        cast(Any, runtime._lock)._is_owned()
    )

    _succeed(runtime, task_id)

    registry.release_worker.assert_called_once_with("wkr-1", "dsp-1")
    assert locked == [False]


def test_a_failed_release_is_retried_at_the_next() -> None:
    registry = MagicMock()
    runtime, task_id = _dispatched(registry)
    registry.release_worker.side_effect = [ConnectionError("down"), True]

    _succeed(runtime, task_id)
    runtime.return_dispatch("tsk-unknown", None, increment_retry=False, front=False)

    assert [call.args for call in registry.release_worker.call_args_list] == [
        ("wkr-1", "dsp-1"),
        ("wkr-1", "dsp-1"),
    ]


@pytest.mark.parametrize("entry", ["plan_merge", "release_merge", "release_child"])
def test_a_failed_release_is_retried_at_a_merge(entry: str) -> None:
    registry = MagicMock()
    runtime, task_id = _dispatched(registry)
    registry.release_worker.side_effect = [ConnectionError("down"), True]
    _succeed(runtime, task_id)

    if entry == "plan_merge":
        runtime.plan_merge(task_id, 2, "wkr-1")
    elif entry == "release_merge":
        runtime.release_merge(task_id)
    else:
        runtime.release_merged_child(task_id, "tsk-unknown", None)

    assert registry.release_worker.call_count == 2


def test_a_failed_announcement_still_releases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rds: Any = MagicMock()
    rds.sync.eval.return_value = b"IDLE"
    rds.sync.publish_telemetry.side_effect = ConnectionError("telemetry down")

    with caplog.at_level(logging.WARNING):
        released = WorkerRegistry(cast(Any, rds)).release_worker("wkr-1", "dsp-1")

    assert released is True
    assert "Failed to announce worker wkr-1" in caplog.text


@pytest.mark.parametrize("recorded_by", ["dispatcher", "first_event"])
def test_a_redispatch_releases_the_earlier_reservation_it_ends(
    recorded_by: str,
) -> None:
    registry = MagicMock()
    workflows = _Registry()
    runtime = TaskRuntime(
        cast(Any, workflows),
        cast(Any, registry),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("reservation-release"),
        credential_vault=InMemoryCredentialVault(),
    )
    _, ids = asyncio.run(_register(runtime, LINEAR))
    task_id = ids["a"]
    assert pop_ready(runtime) == task_id
    record_dispatch(runtime, task_id, "wkr-1", "dsp-1")
    # The failure's commit is lost, so its release waits for the next dispatch.
    workflows.down = True
    with pytest.raises(TransitionNotDurable), runtime.acknowledging():
        runtime.fail_dispatch(task_id, "wkr-1", {}, _TS, "dsp-1", retryable=True)
    workflows.down = False
    registry.release_worker.reset_mock()
    assert runtime.ready_queue_length() == 1
    assert pop_ready(runtime) == task_id

    runtime.begin_publish(
        task_id, cast(Any, _worker("wkr-2")), "dsp-2", input_preparation=False
    )
    if recorded_by == "dispatcher":
        runtime.mark_dispatched(task_id)
    else:
        runtime.mark_started(task_id, "wkr-2", {}, _TS, "dsp-2")

    registry.release_worker.assert_called_once_with("wkr-1", "dsp-1")
