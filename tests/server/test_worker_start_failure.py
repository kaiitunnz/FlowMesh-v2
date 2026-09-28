"""Tests for how the worker manager reports a worker that fails to start or is
torn down."""

import asyncio
import contextlib
import logging
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from docker.errors import NotFound

from server.supervisor.adapters.base import ProviderSpec, WorkerAdapter
from server.supervisor.adapters.docker import DockerWorkerAdapter, DockerWorkerConfig
from server.supervisor.manager import WorkerInitConfig
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus
from tests.server.supervisor_helpers import StubWorkerManager
from tests.server.test_docker_removal_in_progress import _adapter


def _worker(*, started: bool) -> MagicMock:
    worker = MagicMock(spec=WorkerAdapter)
    worker.name = "gpu_0"
    worker.token = "gpu_0.token"
    worker.status = WorkerStatus.STOPPED
    worker.start = AsyncMock(return_value=started)
    return worker


def _info_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


class TestStartWorkerFailure:
    @pytest.mark.asyncio
    async def test_failed_start_is_logged_as_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        worker = _worker(started=False)

        with caplog.at_level(logging.ERROR, logger="test.supervisor"):
            result = await wm._start_worker(worker)

        assert result is False
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == ["Worker gpu_0 failed to start"]

    @pytest.mark.asyncio
    async def test_failed_start_keeps_the_worker_registered(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=False)

        assert await wm._start_worker(worker) is False
        wm._stop_and_destroy_worker.assert_not_awaited()
        registry.try_pop.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_start_logs_no_error_and_keeps_the_worker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]
        worker = _worker(started=True)

        with caplog.at_level(logging.ERROR, logger="test.supervisor"):
            result = await wm._start_worker(worker)

        assert result is True
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        wm._stop_and_destroy_worker.assert_not_awaited()
        registry.try_pop.assert_not_called()


class TestCreateWorkerFailure:
    @pytest.mark.asyncio
    async def test_failed_create_unwinds_the_worker_it_made(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        worker = _worker(started=False)
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Failed to start worker 'gpu_0'"):
            await wm.create_worker(WorkerInitConfig(init_on_start=True))

        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)

    @pytest.mark.asyncio
    async def test_create_whose_start_raises_unwinds_the_worker_it_made(self) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        worker = _worker(started=False)
        worker.start = AsyncMock(side_effect=OSError("docker unavailable"))
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with pytest.raises(OSError, match="docker unavailable"):
            await wm.create_worker(WorkerInitConfig(init_on_start=True))

        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)

    @pytest.mark.asyncio
    async def test_a_create_its_command_times_out_unwinds_the_worker_it_made(
        self,
    ) -> None:
        registry = MagicMock()
        wm = StubWorkerManager(registry)
        worker = _worker(started=True)

        async def long_pull() -> bool:
            await asyncio.sleep(60)
            return True

        worker.start = AsyncMock(side_effect=long_pull)
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]
        wm._stop_and_destroy_worker = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                wm.create_worker(WorkerInitConfig(init_on_start=True)), timeout=0.05
            )

        wm._stop_and_destroy_worker.assert_awaited_once_with(worker)
        registry.try_pop.assert_called_once_with(worker.token)


class TestStopAndDestroyWorkerLog:
    @pytest.mark.asyncio
    async def test_a_worker_that_failed_to_start_is_logged_as_destroyed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        wm._destroy_worker = MagicMock()  # type: ignore[method-assign]
        worker = _worker(started=False)
        wm._create_worker = MagicMock(return_value=worker)  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO, logger="test.supervisor"):
            with pytest.raises(RuntimeError):
                await wm.create_worker(WorkerInitConfig(init_on_start=True))

        assert _info_messages(caplog) == [
            "Destroying worker gpu_0 that is not running.",
            "Worker gpu_0 destroyed.",
        ]

    @pytest.mark.asyncio
    async def test_a_running_worker_is_logged_as_stopped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wm = StubWorkerManager()
        wm._destroy_worker = MagicMock()  # type: ignore[method-assign]
        worker = _worker(started=True)
        worker.status = WorkerStatus.RUNNING
        worker.stop = AsyncMock(return_value=True)

        with caplog.at_level(logging.INFO, logger="test.supervisor"):
            assert await wm._stop_and_destroy_worker(worker) is True

        worker.stop.assert_awaited_once_with()
        assert _info_messages(caplog) == [
            "Stopping worker gpu_0...",
            "Worker gpu_0 stopped.",
        ]


def _docker_creating(release: threading.Event) -> tuple[MagicMock, MagicMock]:
    """A Docker client whose create outlives the cancel of the start awaiting it."""
    created = MagicMock(status="running")
    containers: list[MagicMock] = []
    docker = MagicMock()

    def get(_name: str) -> MagicMock:
        if not containers:
            raise NotFound("gone")
        return containers[0]

    def run(**_: object) -> MagicMock:
        release.wait(timeout=5.0)
        containers.append(created)
        return created

    docker.containers.get.side_effect = get
    docker.containers.run.side_effect = run
    docker.containers.list.return_value = []
    docker.volumes.list.return_value = []
    return docker, created


def _creating_manager(
    release: threading.Event,
) -> tuple[StubWorkerManager, WorkerRegistry, WorkerAdapter, MagicMock, MagicMock]:
    """A manager over a real registry whose create's start waits on ``release``."""
    docker, created = _docker_creating(release)
    adapter = _adapter(docker)
    registry = WorkerRegistry()
    wm = StubWorkerManager(registry)
    factory = MagicMock()
    wm._providers = {
        "docker": ProviderSpec(
            "docker", DockerWorkerConfig, DockerWorkerAdapter, factory
        )
    }

    def create(_config: WorkerInitConfig) -> WorkerAdapter:
        registry.add(adapter)
        return adapter

    wm._create_worker = MagicMock(side_effect=create)  # type: ignore[method-assign]
    return wm, registry, adapter, created, factory


class TestCancelledCreate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", ["timeout", "timeout_then_shutdown", "shutdown"])
    async def test_a_cancelled_create_stops_what_it_created_once(
        self, shape: str
    ) -> None:
        release = threading.Event()
        wm, registry, adapter, created, factory = _creating_manager(release)
        create = WorkerInitConfig(init_on_start=True)

        if shape == "shutdown":
            task = asyncio.ensure_future(wm.create_worker(create))
        else:
            task = asyncio.ensure_future(
                asyncio.wait_for(wm.create_worker(create), timeout=0.1)
            )
        await asyncio.sleep(0.2)
        if shape != "timeout":
            # The supervisor's shutdown cancels the command, then stops its workers.
            task.cancel()
            asyncio.get_running_loop().call_later(0.2, release.set)
            await wm.stop()
        else:
            release.set()
        with contextlib.suppress(BaseException):
            await task

        created.stop.assert_called_once()
        created.remove.assert_called_once()
        factory.destroy_worker.assert_called_once_with(adapter)
        assert registry.try_get(adapter.token) is None

    @pytest.mark.asyncio
    async def test_a_create_keeps_its_name_until_its_unwind_ends(self) -> None:
        release = threading.Event()
        wm, registry, adapter, created, _ = _creating_manager(release)
        create = WorkerInitConfig(init_on_start=True)
        task = asyncio.ensure_future(
            asyncio.wait_for(wm.create_worker(create), timeout=0.1)
        )
        await asyncio.sleep(0.2)

        assert registry.try_get_by_name(adapter.name) is adapter
        release.set()
        with contextlib.suppress(BaseException):
            await task
        created.stop.assert_called_once()
        assert registry.try_get_by_name(adapter.name) is None
