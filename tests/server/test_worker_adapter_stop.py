"""A worker adapter stops what it started, whatever the worker's status reads.

The status reads STOPPED whenever the worker's event stream closes, as after a worker
crash or a reconnect, while its container or instance still runs.
"""

import asyncio
import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from server.hooks import PrincipalContext
from server.supervisor.adapters.base import ProviderSpec, WorkerTokenType
from server.supervisor.adapters.vastai import VastAIWorkerAdapter, VastAIWorkerConfig
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus
from server.utils.helpers import ResourcePool
from tests.server.supervisor_helpers import StubWorkerManager
from tests.server.test_docker_removal_in_progress import _adapter


class _Docker:
    def __init__(self) -> None:
        self.container = MagicMock()
        self.client = MagicMock()
        self.client.containers.get.return_value = self.container
        self.client.containers.list.return_value = []
        self.client.volumes.list.return_value = []
        self.adapter = _adapter(self.client)

    async def start(self) -> None:
        # What a successful start leaves behind.
        self.adapter._is_started = True
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
            name="vast_0",
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

    assert await wm.destroy_worker(world.adapter.name)

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

    assert await _manager(world, kind).stop_worker(world.adapter.name)

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

    assert await wm.stop_worker(world.adapter.name)
    assert world.adapter.status is WorkerStatus.STOPPED

    assert await wm.start_worker(world.adapter.name)
    assert world.adapter.holds_worker()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "vastai"])
async def test_a_worker_whose_event_stream_closed_is_not_started_again(
    kind: str,
) -> None:
    world = _world(kind)
    await world.start()
    world.adapter.set_status(WorkerStatus.STOPPED)

    with pytest.raises(ValueError, match="starting, running or stopping"):
        await _manager(world, kind).start_worker(world.adapter.name)

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
        asyncio.ensure_future(wm.start_worker(world.adapter.name)) for _ in range(2)
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
    start = asyncio.ensure_future(wm.start_worker(world.adapter.name))
    await asyncio.to_thread(refusing.wait, 5)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    release.set()
    assert world.adapter._starting is not None
    await asyncio.wait({world.adapter._starting})
    await asyncio.sleep(0)

    assert world.adapter.status is WorkerStatus.STOPPED
    with pytest.raises(ValueError, match="not starting or running"):
        await wm.stop_worker(world.adapter.name)
    assert world.stops == 0
    assert await wm.start_worker(world.adapter.name) is False


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
    first = asyncio.ensure_future(wm.start_worker(world.adapter.name))
    await asyncio.to_thread(creating.wait, 5)
    stop = asyncio.ensure_future(wm.stop_worker(world.adapter.name))
    await asyncio.sleep(0.05)
    world.adapter.set_status(WorkerStatus.STOPPED)  # a late stream close
    second = asyncio.ensure_future(wm.start_worker(world.adapter.name))
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
        assert await wm.destroy_worker(world.adapter.name)

    assert f"Stopping worker {world.adapter.name}..." in caplog.messages
    assert f"Worker {world.adapter.name} stopped." in caplog.messages


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
    first = asyncio.ensure_future(wm.start_worker(world.adapter.name))
    await asyncio.to_thread(starting.wait, 5)
    stop = asyncio.ensure_future(wm.stop_worker(world.adapter.name))
    await asyncio.sleep(0.05)
    world.adapter.set_status(WorkerStatus.STOPPED)  # a late stream close
    second = asyncio.ensure_future(wm.start_worker(world.adapter.name))
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
    destroy = asyncio.ensure_future(wm.destroy_worker(world.adapter.name))
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
    assert wm._registry.try_get_by_name(world.adapter.name) is None


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
