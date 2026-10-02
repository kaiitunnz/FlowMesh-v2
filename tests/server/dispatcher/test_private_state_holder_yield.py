"""A long-lived allocation keeps off a private-state holder, and yields one on need."""

import asyncio
import json
import logging
import time
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.registries.worker import Reservation, Worker
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from shared.private_state import OwnerFence
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import CapturingDispatcher, WorkflowRegistryStub
from tests.server.result_store import make_result_reader
from tests.support.waiting import pop_ready

_HOLDER = OwnerFence(worker_id="wkr-holder", incarnation=1)

_ECHO_WORKFLOW = """
apiVersion: mloc/v1
kind: Workflow
metadata:
  name: holder-yield
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
"""

_SERVE = json.dumps(
    {
        "apiVersion": "flowmesh/v1",
        "kind": "Serve",
        "metadata": {"name": "serve"},
        "spec": {
            "taskType": "dev_model",
            "resources": {"hardware": {"cpu": 1, "memory": "1Gi"}},
            "model": {"source": {"type": "huggingface", "identifier": "m"}},
        },
    }
)


def _runtime() -> TaskRuntime:
    return TaskRuntime(
        cast(Any, WorkflowRegistryStub()),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("holder-yield-test"),
        credential_vault=InMemoryCredentialVault(),
    )


def _register(runtime: TaskRuntime, payload: str, **kwargs: Any) -> str:
    _, entries = asyncio.run(
        runtime.register("owner", "org", payload, format="native", **kwargs)
    )
    return entries[0].task_id


def _worker(worker_id: str) -> Worker:
    return Worker(
        id=worker_id,
        namespace="ns",
        cluster="cluster",
        node_id="nde-1",
        node_alias="node",
        incarnation=1,
    )


def _dispatcher(runtime: TaskRuntime, idle_ids: list[str]) -> CapturingDispatcher:
    registry = mock.Mock()
    registry.idle_satisfying_pool.return_value = [_worker(wid) for wid in idle_ids]
    registry.satisfying_workers.return_value = [_worker(wid) for wid in idle_ids]
    registry.get_worker.return_value = _worker(_HOLDER.worker_id)
    registry.is_worker_stale.return_value = False
    return CapturingDispatcher(
        runtime=runtime,
        worker_registry=registry,
        logger=logging.getLogger("holder-yield-test"),
        no_worker_grace_sec=0,
    )


def _registry(dispatcher: CapturingDispatcher) -> mock.Mock:
    return cast(mock.Mock, dispatcher._worker_registry)


def _selected_pool(
    runtime: TaskRuntime, task_id: str, idle_ids: list[str]
) -> list[str]:
    """The pool the dispatcher selects from, with a selector that would pick the
    holder whenever it is offered one."""
    runtime.private_state_holders = mock.Mock(  # type: ignore[method-assign]
        return_value=[_HOLDER]
    )
    seen: list[list[str]] = []

    def _prefer_holder(pool: list[Worker], *args: Any, **kwargs: Any) -> Any:
        seen.append([worker.id for worker in pool])
        return None, {}

    with mock.patch("server.dispatcher.base.select_worker", _prefer_holder):
        _dispatcher(runtime, idle_ids).dispatch_once(task_id)
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("resident", [True, False])
def test_a_serve_task_keeps_off_a_holder_while_another_worker_is_idle(
    resident: bool,
) -> None:
    runtime = _runtime()
    serve = _register(runtime, _SERVE, resident=resident)

    pool = _selected_pool(runtime, serve, [_HOLDER.worker_id, "wkr-other"])

    assert pool == ["wkr-other"]


def test_a_serve_task_takes_a_holder_when_no_other_worker_is_idle() -> None:
    runtime = _runtime()
    serve = _register(runtime, _SERVE, resident=True)

    assert _selected_pool(runtime, serve, [_HOLDER.worker_id]) == [_HOLDER.worker_id]


def test_an_ordinary_task_may_take_a_holder() -> None:
    runtime = _runtime()
    task = _register(runtime, _ECHO_WORKFLOW)

    pool = _selected_pool(runtime, task, [_HOLDER.worker_id, "wkr-other"])

    assert sorted(pool) == sorted([_HOLDER.worker_id, "wkr-other"])


class _Clock:
    """Stands in for the dispatcher's ``time`` module, its monotonic clock settable."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return time.time()


def _waiting_episode(
    resident: bool, stale_dispatch: bool = False
) -> tuple[TaskRuntime, CapturingDispatcher, str, str, list[str]]:
    """An owner-affine episode whose holder runs a serve task; no worker is idle.

    With ``stale_dispatch``, an earlier task still reads as dispatched to the holder,
    as a lost dispatch does until its worker's loss is settled.
    """
    runtime = _runtime()
    if stale_dispatch:
        stale = _register(runtime, _ECHO_WORKFLOW)
        assert pop_ready(runtime) == stale
        record_dispatch(runtime, stale, _HOLDER.worker_id, "dsp-stale")
    serve = _register(runtime, _SERVE, resident=resident)
    assert pop_ready(runtime) == serve
    record_dispatch(runtime, serve, _HOLDER.worker_id, "dsp-serve")
    episode = _register(runtime, _ECHO_WORKFLOW)
    runtime.private_state_owner = mock.Mock(  # type: ignore[method-assign]
        side_effect=lambda task_id: _HOLDER if task_id == episode else None
    )
    asked: list[str] = []
    runtime.set_resident_yield_hook(asked.append)
    dispatcher = _dispatcher(runtime, [])
    _registry(dispatcher).reservation.return_value = Reservation(
        _HOLDER.worker_id, serve, "dsp-serve"
    )
    return runtime, dispatcher, episode, serve, asked


@pytest.mark.parametrize("stale_dispatch", [False, True])
def test_a_waiting_episode_asks_resident_capacity_to_free_its_holder(
    stale_dispatch: bool,
) -> None:
    _runtime_, dispatcher, episode, serve, asked = _waiting_episode(
        resident=True, stale_dispatch=stale_dispatch
    )
    clock = _Clock()

    with mock.patch("server.dispatcher.base.time", clock):
        for _ in range(50):
            assert dispatcher.dispatch_once(episode) is False
        assert asked == [serve]
        clock.now += 2.5
        dispatcher.dispatch_once(episode)

    assert asked == [serve, serve]
    assert {kw["reason"] for _, kw in dispatcher.requeued} == {
        "private_state_owner_busy"
    }


@pytest.mark.parametrize(
    "unread", [None, ConnectionError("control Redis dropped")], ids=["none", "error"]
)
def test_a_holder_with_no_readable_reservation_asks_its_resident_serve_task(
    unread: Exception | None,
) -> None:
    _runtime_, dispatcher, episode, serve, asked = _waiting_episode(resident=True)
    _registry(dispatcher).reservation.side_effect = unread
    _registry(dispatcher).reservation.return_value = None

    assert dispatcher.dispatch_once(episode) is False

    assert asked == [serve]


def test_a_reservation_for_an_earlier_dispatch_asks_nothing() -> None:
    _runtime_, dispatcher, episode, serve, asked = _waiting_episode(resident=True)
    _registry(dispatcher).reservation.return_value = Reservation(
        _HOLDER.worker_id, serve, "dsp-earlier"
    )

    dispatcher.dispatch_once(episode)

    assert asked == []


def test_a_standing_serve_task_is_never_asked_to_yield() -> None:
    _runtime_, dispatcher, episode, _serve, asked = _waiting_episode(resident=False)
    clock = _Clock()

    with mock.patch("server.dispatcher.base.time", clock):
        for _ in range(5):
            dispatcher.dispatch_once(episode)
            clock.now += 3

    assert asked == []


def test_a_long_wait_on_a_holder_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _runtime_, dispatcher, episode, serve, _asked = _waiting_episode(resident=False)
    clock = _Clock()

    with (
        caplog.at_level(logging.INFO, logger="holder-yield-test"),
        mock.patch("server.dispatcher.base.time", clock),
    ):
        dispatcher.dispatch_once(episode)
        clock.now += 31
        for _ in range(5):
            dispatcher.dispatch_once(episode)

    waits = [r for r in caplog.records if "for its private-state holder" in r.message]
    assert len(waits) == 1 and serve in waits[0].message


def test_a_wait_is_forgotten_once_its_task_leaves_the_queue() -> None:
    runtime, dispatcher, episode, _serve, _asked = _waiting_episode(resident=True)
    other = _register(runtime, _ECHO_WORKFLOW)
    runtime.private_state_owner = mock.Mock(  # type: ignore[method-assign]
        side_effect=lambda task_id: _HOLDER if task_id in (episode, other) else None
    )
    dispatcher.dispatch_once(episode)
    assert episode in dispatcher._owner_wait_since
    record = runtime.get_record(episode)
    assert record is not None
    runtime.cancel_workflow(record.workflow_id)
    assert record.status is not TaskStatus.PENDING

    dispatcher.dispatch_once(other)

    assert set(dispatcher._owner_wait_since) == {other}
    assert set(dispatcher._yield_requested_at) == {other}


def test_a_wait_is_forgotten_when_its_holder_is_lost() -> None:
    _runtime_, dispatcher, episode, _serve, _asked = _waiting_episode(resident=True)
    dispatcher.dispatch_once(episode)
    assert episode in dispatcher._owner_wait_since
    _registry(dispatcher).get_worker.return_value = None

    dispatcher.dispatch_once(episode)

    assert episode not in dispatcher._owner_wait_since
    assert episode not in dispatcher._yield_requested_at
