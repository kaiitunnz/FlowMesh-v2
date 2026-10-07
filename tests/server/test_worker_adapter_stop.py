"""A worker adapter stops what it started, whatever the worker's status reads.

The status reads STOPPED whenever the worker's event stream closes, as after a worker
crash or a reconnect, while its container or instance still runs.
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from docker.errors import NotFound

from server.hooks import PrincipalContext
from server.supervisor.adapters.base import ProviderSpec, WorkerTokenType
from server.supervisor.adapters.vastai import VastAIWorkerAdapter, VastAIWorkerConfig
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus
from server.supervisor.services.grpc_server import SupervisorServicer
from server.utils.helpers import ResourcePool
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.supervisor_helpers import StubWorkerManager
from tests.server.test_docker_removal_in_progress import _adapter


class _Docker:
    def __init__(self) -> None:
        self.container = MagicMock(id="c1", status="exited")
        self.container.remove.side_effect = self._remove
        self.gone = True
        self.client = MagicMock()
        self.client.containers.get.side_effect = self._get
        self.client.containers.run.side_effect = self._run
        self.client.containers.list.return_value = []
        self.client.volumes.list.return_value = []
        self.adapter = _adapter(self.client)

    def _get(self, _: str) -> MagicMock:
        if self.gone:
            raise NotFound("gone")
        return self.container

    def _remove(self, **_: Any) -> None:
        self.gone = True

    def _run(self, **_: Any) -> MagicMock:
        self.gone = False
        self.container.status = "running"
        return self.container

    async def start(self) -> None:
        # What a successful start leaves behind.
        self._run()
        self.adapter._container_id = self.container.id
        self.adapter.set_status(WorkerStatus.RUNNING)

    @property
    def stops(self) -> int:
        return self.container.stop.call_count


class _VastAI:
    def __init__(self, instance_id: int | None = None) -> None:
        self.client = MagicMock()
        self.client.search_offers.return_value = [{"id": 7, "gpu_name": None}]
        self.client.create_instance.return_value = {
            "success": True,
            "new_contract": 70,
        }
        self.client.destroy_instance.return_value = None
        self.client.stop_instance.return_value = None
        self.client.show_instance.return_value = {}
        self.adapter = VastAIWorkerAdapter(
            token=WorkerTokenType("vast_0.token"),
            alias="vast_0",
            config=VastAIWorkerConfig(instance_id=instance_id),
            vastai_client=self.client,
            instance_pool=ResourcePool(),
            owner=PrincipalContext(
                principal_id="u",
                org_id="o",
                external_id="u",
                principal_type="user",
                scopes=[],
            ),
        )
        self.adapter._STOP_TIMEOUT = 0

    async def start(self) -> None:
        assert await self.adapter.start()
        self.adapter.set_status(WorkerStatus.RUNNING)

    @property
    def stops(self) -> int:
        return (
            self.client.destroy_instance.call_count
            + self.client.stop_instance.call_count
        )


def _world(kind: str) -> Any:
    if kind == "docker":
        return _Docker()
    # A configured instance is the operator's until this adapter starts it.
    return _VastAI(instance_id=5 if kind == "vastai-configured" else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_destroy_stops_a_worker_whose_event_stream_closed(kind: str) -> None:
    world = _world(kind)
    await world.start()
    world.adapter.set_status(WorkerStatus.STOPPED)
    registry = WorkerRegistry()
    registry.add(world.adapter)
    wm = StubWorkerManager(registry)
    factory = MagicMock()
    wm._providers = {
        kind: ProviderSpec(
            kind, type(world.adapter.config), type(world.adapter), factory
        )
    }

    assert await wm.destroy_worker(world.adapter.alias)

    assert world.stops == 1
    factory.destroy_worker.assert_called_once_with(world.adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai", "vastai-configured"])
async def test_a_worker_never_started_gets_no_stop(kind: str) -> None:
    world = _world(kind)

    assert await world.adapter.stop()

    assert world.stops == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_stopped_worker_is_not_stopped_again(kind: str) -> None:
    world = _world(kind)
    await world.start()
    assert await world.adapter.stop()

    assert await world.adapter.stop()

    assert world.stops == 1


def _manager(world: Any, kind: str) -> StubWorkerManager:
    registry = WorkerRegistry()
    registry.add(world.adapter)
    wm = StubWorkerManager(registry)
    wm._is_started = True
    wm._providers = {
        kind: ProviderSpec(
            kind, type(world.adapter.config), type(world.adapter), MagicMock()
        )
    }
    return wm


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_an_operator_stops_a_worker_whose_event_stream_closed(kind: str) -> None:
    world = _world(kind)
    await world.start()
    world.adapter.set_status(WorkerStatus.STOPPED)

    assert await _manager(world, kind).stop_worker(world.adapter.alias)

    assert world.stops == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_crashed_worker_an_operator_stopped_starts_again(kind: str) -> None:
    world = _world(kind)
    await world.start()
    wm = _manager(world, kind)
    wm._registry.set_worker_id(world.adapter.token, "wkr-1")
    world.adapter.set_worker_id("wkr-1")
    # The worker crashed: its event stream closed without an unregister.
    world.adapter.clear_worker_id()
    world.adapter.set_status(WorkerStatus.STOPPED)

    assert await wm.stop_worker(world.adapter.alias)
    assert world.adapter.status is WorkerStatus.STOPPED

    assert await wm.start_worker(world.adapter.alias)
    assert world.adapter.holds_worker()


def _events_servicer(wm: StubWorkerManager) -> SupervisorServicer:
    servicer = SupervisorServicer.__new__(SupervisorServicer)
    servicer._registry = wm._registry
    servicer._relay_service = MagicMock()
    servicer._relay_bridges = {}
    servicer._logger = logging.getLogger("test")
    return servicer


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_worker_that_died_before_its_event_stream_opened_starts_again(
    kind: str,
) -> None:
    world = _world(kind)
    await world.start()
    wm = _manager(world, kind)
    # Registered, so its id and binding are set, but it died before PushEvents.
    wm._registry.set_worker_id(world.adapter.token, "wkr-1")
    world.adapter.set_worker_id("wkr-1")
    world.adapter.set_status(WorkerStatus.STARTING)

    assert await wm.stop_worker(world.adapter.alias)
    assert world.adapter.status is WorkerStatus.STOPPED

    assert await wm.start_worker(world.adapter.alias)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_stopped_worker_whose_event_stream_is_open_stops_at_its_close(
    kind: str,
) -> None:
    world = _world(kind)
    await world.start()
    wm = _manager(world, kind)
    wm._registry.set_worker_id(world.adapter.token, "wkr-1")
    world.adapter.set_worker_id("wkr-1")
    closed = asyncio.Event()

    async def events() -> AsyncIterator[supervisor_pb2.EventMessage]:
        yield supervisor_pb2.EventMessage()
        await closed.wait()

    context = MagicMock()
    context.invocation_metadata.return_value = [("x-worker-token", world.adapter.token)]
    stream = asyncio.ensure_future(_events_servicer(wm).PushEvents(events(), context))
    await asyncio.sleep(0)
    assert world.adapter.has_event_stream

    assert await wm.stop_worker(world.adapter.alias)
    assert world.adapter.status is WorkerStatus.STOPPING

    closed.set()
    await stream
    assert not world.adapter.has_event_stream
    assert world.adapter.status is WorkerStatus.STOPPED
    assert await wm.start_worker(world.adapter.alias)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_worker_whose_event_stream_closed_is_not_started_again(
    kind: str,
) -> None:
    world = _world(kind)
    await world.start()
    world.adapter.set_status(WorkerStatus.STOPPED)

    with pytest.raises(ValueError, match="starting, running or stopping"):
        await _manager(world, kind).start_worker(world.adapter.alias)

    if kind == "vastai":
        assert world.client.create_instance.call_count == 1


@pytest.mark.asyncio
async def test_a_start_waits_for_the_stop_still_running() -> None:
    world = _VastAI()
    contracts = iter([70, 80])
    world.client.create_instance.side_effect = lambda **_: {
        "success": True,
        "new_contract": next(contracts),
    }
    await world.start()
    destroying = threading.Event()
    release = threading.Event()

    def destroy_instance(**_: Any) -> None:
        destroying.set()
        release.wait(5)

    world.client.destroy_instance.side_effect = destroy_instance
    stop = asyncio.ensure_future(world.adapter.stop())
    await asyncio.to_thread(destroying.wait, 5)
    world.adapter.set_status(WorkerStatus.STOPPED)  # the stream closes mid-stop
    start = asyncio.ensure_future(world.adapter.start())
    await asyncio.sleep(0.05)
    assert world.client.create_instance.call_count == 1

    release.set()
    assert await stop
    assert await start

    assert world.client.create_instance.call_count == 2
    assert [c.kwargs["id"] for c in world.client.destroy_instance.call_args_list] == [
        70
    ]
    assert world.adapter.holds_worker()


@pytest.mark.asyncio
async def test_two_starts_behind_a_finishing_stop_create_one_instance() -> None:
    world = _VastAI()
    contracts = iter([70, 80, 90])
    world.client.create_instance.side_effect = lambda **_: {
        "success": True,
        "new_contract": next(contracts),
    }
    await world.start()
    destroyed = threading.Event()
    release = threading.Event()
    stop_instance = world.adapter._stop

    def stop_then_return_late() -> bool:
        ok = stop_instance()
        destroyed.set()
        release.wait(5)
        return ok

    world.adapter._stop = stop_then_return_late  # type: ignore[method-assign]
    stop = asyncio.ensure_future(world.adapter.stop())
    await asyncio.to_thread(destroyed.wait, 5)
    world.adapter.set_status(WorkerStatus.STOPPED)  # the stream closes mid-stop
    wm = _manager(world, "vastai")
    starts = [
        asyncio.ensure_future(wm.start_worker(world.adapter.alias)) for _ in range(2)
    ]
    await asyncio.sleep(0.05)

    release.set()
    assert await stop
    assert await asyncio.gather(*starts) == [True, True]

    assert world.client.create_instance.call_count == 2
    assert world.adapter._instance_id == 80


@pytest.mark.asyncio
@pytest.mark.parametrize("instance_id", [None, 5])
async def test_a_cancelled_start_that_then_failed_leaves_the_worker_stopped(
    instance_id: int | None,
) -> None:
    world = _VastAI(instance_id)
    refusing = threading.Event()
    release = threading.Event()

    def refuse(**_: Any) -> Any:
        refusing.set()
        release.wait(5)
        return "vast refused" if instance_id else {"success": False}

    world.client.start_instance.side_effect = refuse
    world.client.create_instance.side_effect = refuse
    wm = _manager(world, "vastai")
    start = asyncio.ensure_future(wm.start_worker(world.adapter.alias))
    await asyncio.to_thread(refusing.wait, 5)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    release.set()
    assert world.adapter._last is not None
    await asyncio.wait({world.adapter._last.future})
    await asyncio.sleep(0)

    assert world.adapter.status is WorkerStatus.STOPPED
    with pytest.raises(ValueError, match="not starting or running"):
        await wm.stop_worker(world.adapter.alias)
    assert world.stops == 0
    assert await wm.start_worker(world.adapter.alias) is False


@pytest.mark.asyncio
async def test_a_start_behind_a_stop_queued_on_a_start_runs_after_the_stop() -> None:
    world = _VastAI()
    contracts = iter([70, 80])
    creating = threading.Event()
    release = threading.Event()

    def create_instance(**_: Any) -> dict[str, Any]:
        creating.set()
        release.wait(5)
        return {"success": True, "new_contract": next(contracts)}

    world.client.create_instance.side_effect = create_instance
    wm = _manager(world, "vastai")
    first = asyncio.ensure_future(wm.start_worker(world.adapter.alias))
    await asyncio.to_thread(creating.wait, 5)
    stop = asyncio.ensure_future(wm.stop_worker(world.adapter.alias))
    await asyncio.sleep(0.05)
    world.adapter.set_status(WorkerStatus.STOPPED)  # a late stream close
    second = asyncio.ensure_future(wm.start_worker(world.adapter.alias))
    await asyncio.sleep(0.05)

    release.set()
    assert await asyncio.gather(first, stop, second) == [True, True, True]

    assert [c.kwargs["id"] for c in world.client.destroy_instance.call_args_list] == [
        70
    ]
    assert world.adapter._instance_id == 80
    assert world.adapter.holds_worker()


@pytest.mark.asyncio
async def test_a_destroy_logs_stopping_a_worker_whose_event_stream_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = _world("docker")
    await world.start()
    world.adapter.set_status(WorkerStatus.STOPPED)
    wm = _manager(world, "docker")

    with caplog.at_level(logging.INFO, logger=wm.logger.name):
        assert await wm.destroy_worker(world.adapter.alias)

    assert f"Stopping worker {world.adapter.alias}..." in caplog.messages
    assert f"Worker {world.adapter.alias} stopped." in caplog.messages


def _gate_first_start(world: Any) -> tuple[threading.Event, threading.Event]:
    starting = threading.Event()
    release = threading.Event()
    start = world.adapter._start
    contracts = iter([70, 80])
    world.client.create_instance.side_effect = lambda **_: {
        "success": True,
        "new_contract": next(contracts),
    }

    def gated_start() -> bool:
        if not starting.is_set():
            starting.set()
            release.wait(5)
        return start()

    world.adapter._start = gated_start
    world.adapter.set_status(WorkerStatus.STOPPED)
    return starting, release


def _created(world: Any) -> int:
    if isinstance(world, _Docker):
        return world.client.containers.run.call_count
    return world.client.create_instance.call_count


async def _start_queued_behind_a_stop_on_a_start(
    world: Any, wm: StubWorkerManager
) -> tuple[threading.Event, asyncio.Future[bool], asyncio.Future[bool], Any]:
    starting, release = _gate_first_start(world)
    first = asyncio.ensure_future(wm.start_worker(world.adapter.alias))
    await asyncio.to_thread(starting.wait, 5)
    stop = asyncio.ensure_future(wm.stop_worker(world.adapter.alias))
    await asyncio.sleep(0.05)
    world.adapter.set_status(WorkerStatus.STOPPED)  # a late stream close
    second = asyncio.ensure_future(wm.start_worker(world.adapter.alias))
    await asyncio.sleep(0.05)
    return release, first, stop, second


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_start_queued_behind_a_stop_creates_nothing_once_destroyed(
    kind: str,
) -> None:
    world = _world(kind)
    wm = _manager(world, kind)
    release, first, stop, second = await _start_queued_behind_a_stop_on_a_start(
        world, wm
    )
    destroy = asyncio.ensure_future(wm.destroy_worker(world.adapter.alias))
    await asyncio.sleep(0.05)

    release.set()
    assert await asyncio.gather(first, stop, second, destroy) == [
        True,
        True,
        False,
        True,
    ]

    assert _created(world) == 1
    assert not world.adapter.holds_worker()
    assert wm._registry.try_get_by_alias(world.adapter.alias) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_start_queued_behind_a_failed_stop_keeps_the_running_worker(
    kind: str,
) -> None:
    world = _world(kind)
    if kind == "docker":
        world.container.stop.side_effect = RuntimeError("daemon refused")
    else:
        world.client.destroy_instance.return_value = "vast refused"
    wm = _manager(world, kind)
    release, first, stop, second = await _start_queued_behind_a_stop_on_a_start(
        world, wm
    )

    release.set()
    assert await asyncio.gather(first, stop, second) == [True, False, True]

    assert _created(world) == 1
    assert world.adapter.holds_worker()
    if kind == "vastai":
        assert world.adapter._instance_id == 70


class _GatedStarts:
    """Holds each of the adapter's ``_start`` calls until it is let through."""

    def __init__(self, world: Any) -> None:
        self.entered = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]
        self._calls = 0
        start = world.adapter._start
        contracts = iter([70, 80])
        world.client.create_instance.side_effect = lambda **_: {
            "success": True,
            "new_contract": next(contracts),
        }

        def gated() -> bool:
            call, self._calls = self._calls, self._calls + 1
            self.entered[call].set()
            self.release[call].wait(5)
            return start()

        world.adapter._start = gated
        world.adapter.set_status(WorkerStatus.STOPPED)


def _fail_stops(world: Any) -> dict[str, bool]:
    failing = {"on": True}

    def stop_container(*_: Any, **__: Any) -> None:
        if failing["on"]:
            raise RuntimeError("refused")

    def destroy_instance(**_: Any) -> str | None:
        return "refused" if failing["on"] else None

    if isinstance(world, _Docker):
        world.container.stop.side_effect = stop_container
    else:
        world.client.destroy_instance.side_effect = destroy_instance
    return failing


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
@pytest.mark.parametrize("stop_fails", [False, True])
@pytest.mark.parametrize("late_stop", ["before_the_start_resumes", "during_the_start"])
async def test_a_stop_accepted_after_a_queued_start_stops_what_it_starts(
    kind: str, stop_fails: bool, late_stop: str
) -> None:
    world = _world(kind)
    starts = _GatedStarts(world)
    failing = _fail_stops(world) if stop_fails else {"on": False}
    adapter = world.adapter
    late: list[asyncio.Future[bool]] = []

    first = asyncio.ensure_future(adapter.start())
    await asyncio.to_thread(starts.entered[0].wait, 5)
    stop = asyncio.ensure_future(adapter.stop())
    await asyncio.sleep(0.05)
    assert adapter._last is not None
    if late_stop == "before_the_start_resumes":

        def accept_the_late_stop(_: Any) -> None:
            failing["on"] = False
            late.append(asyncio.ensure_future(adapter.stop()))

        # Runs before the queued start below wakes on the stop.
        adapter._last.future.add_done_callback(accept_the_late_stop)
    second = asyncio.ensure_future(adapter.start())
    await asyncio.sleep(0.05)
    starts.release[0].set()
    if late_stop == "during_the_start" and not stop_fails:
        await asyncio.to_thread(starts.entered[1].wait, 5)
        late.append(asyncio.ensure_future(adapter.stop()))
    elif late_stop == "during_the_start":
        await asyncio.wait({stop})
        failing["on"] = False
        late.append(asyncio.ensure_future(adapter.stop()))
    starts.release[1].set()

    assert await asyncio.gather(first, stop, second) == [True, not stop_fails, True]
    assert await late[0]
    assert not adapter.holds_worker()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_starts_accepted_together_start_one_worker(kind: str) -> None:
    world = _world(kind)
    starts = _GatedStarts(world)

    first = asyncio.ensure_future(world.adapter.start())
    await asyncio.to_thread(starts.entered[0].wait, 5)
    second = asyncio.ensure_future(world.adapter.start())
    await asyncio.sleep(0.05)
    starts.release[0].set()

    assert await asyncio.gather(first, second) == [True, True]
    assert _created(world) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_stops_accepted_together_stop_once(kind: str) -> None:
    world = _world(kind)
    await world.start()
    stopping = threading.Event()
    release = threading.Event()
    stop = world.adapter._stop

    def gated_stop() -> bool:
        stopping.set()
        release.wait(5)
        return stop()

    world.adapter._stop = gated_stop
    first = asyncio.ensure_future(world.adapter.stop())
    await asyncio.to_thread(stopping.wait, 5)
    second = asyncio.ensure_future(world.adapter.stop())
    await asyncio.sleep(0.05)
    release.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert world.stops == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_destroy_whose_command_is_cancelled_still_removes_the_worker(
    kind: str,
) -> None:
    world = _world(kind)
    await world.start()
    wm = _manager(world, kind)
    stopping = threading.Event()
    release = threading.Event()
    stop = world.adapter._stop

    def gated_stop() -> bool:
        stopping.set()
        release.wait(5)
        return stop()

    world.adapter._stop = gated_stop
    destroy = asyncio.ensure_future(wm.destroy_worker(world.adapter.alias))
    await asyncio.to_thread(stopping.wait, 5)
    destroy.cancel()
    with pytest.raises(asyncio.CancelledError):
        await destroy
    with pytest.raises(ValueError, match="is being destroyed"):
        await wm.start_worker(world.adapter.alias)

    release.set()
    deadline = time.monotonic() + 5
    while wm._registry.try_get_by_alias(world.adapter.alias) is not None:
        assert time.monotonic() < deadline, "the destroy never removed the worker"
        await asyncio.sleep(0.01)
    assert not world.adapter.holds_worker()


@pytest.mark.asyncio
async def test_a_vastai_worker_with_no_event_stream_stops_without_waiting() -> None:
    world = _VastAI()
    world.adapter._STOP_TIMEOUT = 5.0
    await world.start()
    wm = _manager(world, "vastai")
    wm._registry.set_worker_id(world.adapter.token, "wkr-1")
    world.adapter.set_worker_id("wkr-1")
    world.adapter.set_status(WorkerStatus.STARTING)

    started = time.monotonic()
    assert await wm.stop_worker(world.adapter.alias)

    assert time.monotonic() - started < 1.0
    assert world.adapter.status is WorkerStatus.STOPPED


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_bulk_destroy_keeps_a_worker_added_while_it_ran(kind: str) -> None:
    world = _world(kind)
    await world.start()
    wm = _manager(world, kind)
    stopping = threading.Event()
    release = threading.Event()
    stop = world.adapter._stop

    def gated_stop() -> bool:
        stopping.set()
        release.wait(5)
        return stop()

    world.adapter._stop = gated_stop
    destroy = asyncio.ensure_future(wm.destroy_workers(None))
    await asyncio.to_thread(stopping.wait, 5)
    late = _world(kind).adapter
    late.alias, late.token = "late", WorkerTokenType("late.token")
    wm._registry.add(late)
    release.set()
    await destroy

    assert wm._registry.try_get_by_alias(world.adapter.alias) is None
    assert wm._registry.try_get_by_alias("late") is late


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_an_operator_stop_behind_a_queued_start_stops_what_it_starts(
    kind: str,
) -> None:
    world = _world(kind)
    wm = _manager(world, kind)
    release, first, stop, second = await _start_queued_behind_a_stop_on_a_start(
        world, wm
    )
    late = asyncio.ensure_future(wm.stop_worker(world.adapter.alias))
    await asyncio.sleep(0.05)

    release.set()

    assert await asyncio.gather(first, stop, second, late) == [True, True, True, True]
    assert not world.adapter.holds_worker()


@pytest.mark.asyncio
async def test_a_start_after_a_failed_removal_runs_the_container_again() -> None:
    world = _Docker()
    await world.start()
    world.container.remove.side_effect = RuntimeError("daemon refused")
    wm = _manager(world, "docker")
    assert not await wm.stop_worker(world.adapter.alias)
    assert world.adapter.holds_worker()
    world.container.status = "exited"
    world.container.remove.side_effect = None
    world.adapter.set_status(WorkerStatus.STOPPED)  # the stream closed

    assert await wm.start_worker(world.adapter.alias)

    assert world.client.containers.run.call_count == 1


@pytest.mark.asyncio
async def test_a_start_queued_behind_a_failed_removal_runs_the_container_again() -> (
    None
):
    world = _Docker()

    def stop(**_: Any) -> None:
        world.container.status = "exited"

    world.container.stop.side_effect = stop
    # The stop fails to remove the container it stopped, and the queued start clears it.
    outcomes = iter([RuntimeError("daemon refused"), None])

    def remove(**_: Any) -> None:
        if (error := next(outcomes)) is not None:
            raise error
        world.gone = True

    world.container.remove.side_effect = remove
    wm = _manager(world, "docker")
    release, first, stopping, second = await _start_queued_behind_a_stop_on_a_start(
        world, wm
    )

    release.set()

    assert await asyncio.gather(first, stopping, second) == [True, False, True]
    assert _created(world) == 2
    assert world.container.status == "running"
