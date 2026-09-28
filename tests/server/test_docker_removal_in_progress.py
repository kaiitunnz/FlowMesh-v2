"""Tests for the supervisor Docker adapter's handling of a concurrent remove."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, NotFound
from requests import Response

from server.hooks import PrincipalContext
from server.supervisor.adapters import docker as docker_adapter
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.docker import (
    DockerWorkerAdapter,
    DockerWorkerConfig,
    WorkerType,
    _is_removal_in_progress,
)
from server.supervisor.schemas import WorkerStatus

_IN_PROGRESS = "removal of container gpu_0 is already in progress"


def _api_error(status_code: int, explanation: str | None) -> APIError:
    # APIError.status_code is a read-only property over .response.
    response = Response()
    response.status_code = status_code
    return APIError("boom", response=response, explanation=explanation)


def _adapter(docker_client: MagicMock) -> DockerWorkerAdapter:
    adapter = DockerWorkerAdapter(
        token=WorkerTokenType("worker-token"),
        name="gpu_0",
        container_name="gpu_0",
        cuda_devices=None,
        gpu_arch=None,
        config=DockerWorkerConfig(
            worker_type=WorkerType.CPU,
            results_dir="/results",
            hf_cache_dir="/hf-cache",
            enable_ssh=False,
        ),
        docker_client=docker_client,
        owner=PrincipalContext(
            principal_id="test-user",
            org_id="test-org",
            external_id="test-user",
            principal_type="user",
            scopes=[],
        ),
    )
    adapter._hardware = {}
    return adapter


def _stale_container(remove_error: Exception | None) -> MagicMock:
    container = MagicMock(status="exited")
    container.remove.side_effect = remove_error
    return container


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_POLL", 0)


class TestIsRemovalInProgress:
    def test_matches_the_concurrent_removal_409(self) -> None:
        assert _is_removal_in_progress(_api_error(409, _IN_PROGRESS)) is True

    def test_rejects_other_409s(self) -> None:
        name_taken = _api_error(
            409, 'Conflict. The container name "/gpu_0" is already in use'
        )
        running = _api_error(
            409, "You cannot remove a running container. Stop the container"
        )
        assert _is_removal_in_progress(name_taken) is False
        assert _is_removal_in_progress(running) is False

    def test_rejects_non_409_even_with_matching_text(self) -> None:
        assert _is_removal_in_progress(_api_error(500, _IN_PROGRESS)) is False

    def test_tolerates_missing_explanation(self) -> None:
        assert _is_removal_in_progress(_api_error(409, None)) is False

    def test_rejects_non_api_errors(self) -> None:
        assert _is_removal_in_progress(RuntimeError(_IN_PROGRESS)) is False


class TestWaitContainerGone:
    def test_returns_true_once_the_container_disappears(self) -> None:
        client = MagicMock()
        client.containers.get.side_effect = [
            SimpleNamespace(),
            SimpleNamespace(),
            NotFound("gone"),
        ]
        assert _adapter(client)._wait_container_gone() is True
        assert client.containers.get.call_count == 3

    def test_times_out_when_it_never_goes(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        client = MagicMock()
        client.containers.get.return_value = SimpleNamespace()
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._wait_container_gone() is False
        assert "is still present" in caplog.text

    def test_keeps_waiting_through_a_transient_inspect_error(self) -> None:
        client = MagicMock()
        client.containers.get.side_effect = [
            RuntimeError("transport blip"),
            SimpleNamespace(),
            NotFound("gone"),
        ]
        assert _adapter(client)._wait_container_gone() is True
        assert client.containers.get.call_count == 3

    def test_timeout_reports_the_inspect_error_when_presence_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        client = MagicMock()
        client.containers.get.side_effect = RuntimeError("daemon unreachable")
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._wait_container_gone() is False
        assert "Could not confirm removal" in caplog.text
        assert "daemon unreachable" in caplog.text


class TestStartWithStaleContainer:
    def _client(self, stale: MagicMock, *after_remove: Any) -> MagicMock:
        client = MagicMock()
        client.containers.get.side_effect = [stale, *after_remove]
        return client

    def test_removes_a_stale_container_then_starts(self) -> None:
        stale = _stale_container(None)
        client = self._client(stale)
        assert _adapter(client)._start() is True
        stale.remove.assert_called_once_with(force=True)
        client.containers.run.assert_called_once()

    def test_waits_out_a_concurrent_removal_then_starts(self) -> None:
        stale = _stale_container(_api_error(409, _IN_PROGRESS))
        client = self._client(stale, SimpleNamespace(), NotFound("gone"))
        assert _adapter(client)._start() is True
        assert client.containers.run.call_args.kwargs["name"] == "gpu_0"

    def test_starts_when_the_container_is_gone_before_the_remove(self) -> None:
        client = self._client(_stale_container(NotFound("gone")))
        assert _adapter(client)._start() is True
        client.containers.run.assert_called_once()

    def test_other_conflicts_fail_the_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stale = _stale_container(
            _api_error(409, "You cannot remove a running container")
        )
        client = self._client(stale)
        with caplog.at_level(logging.ERROR, logger="supervisor"):
            assert _adapter(client)._start() is False
        client.containers.run.assert_not_called()
        assert "You cannot remove a running container" in caplog.text

    def test_a_removal_that_never_completes_fails_the_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(docker_adapter, "_REMOVAL_IN_PROGRESS_TIMEOUT", 0)
        stale = _stale_container(_api_error(409, _IN_PROGRESS))
        client = self._client(stale, SimpleNamespace())
        assert _adapter(client)._start() is False
        client.containers.run.assert_not_called()

    def test_a_running_container_is_kept(self) -> None:
        running = MagicMock(status="running")
        client = self._client(running)
        assert _adapter(client)._start() is True
        running.remove.assert_not_called()
        client.containers.run.assert_not_called()


class TestStopOrder:
    def _docker(self, events: list[str], worker: MagicMock | Exception) -> MagicMock:
        docker_client = MagicMock()
        if isinstance(worker, Exception):
            docker_client.containers.get.side_effect = worker
        else:
            docker_client.containers.get.return_value = worker
        ssh_container = MagicMock(status="running")
        ssh_container.name = "ssh_0"
        ssh_container.stop.side_effect = lambda **_: events.append("ssh stopped")
        ssh_container.remove.side_effect = lambda **kwargs: events.append(
            "ssh removed" if kwargs == {"force": True} else "ssh removed unforced"
        )
        docker_client.containers.list.return_value = [ssh_container]
        volume = MagicMock()
        volume.remove.side_effect = lambda **_: events.append("volume removed")
        docker_client.volumes.list.return_value = [volume]
        return docker_client

    def test_the_worker_stops_before_its_ssh_containers_and_volumes(self) -> None:
        events: list[str] = []
        worker = MagicMock()
        worker.stop.side_effect = lambda **_: events.append("worker stopped")
        worker.remove.side_effect = lambda **_: events.append("worker removed")

        assert _adapter(self._docker(events, worker))._stop() is True

        assert events == [
            "worker stopped",
            "worker removed",
            "ssh removed",
            "volume removed",
        ]

    def test_a_worker_that_fails_to_stop_keeps_its_ssh_resources(self) -> None:
        events: list[str] = []
        worker = MagicMock()
        worker.stop.side_effect = APIError("stuck")

        assert _adapter(self._docker(events, worker))._stop() is False

        assert events == []

    def test_a_stopped_worker_that_fails_to_be_removed_has_its_ssh_resources_removed(
        self,
    ) -> None:
        events: list[str] = []
        worker = MagicMock()
        worker.stop.side_effect = lambda **_: events.append("worker stopped")
        worker.remove.side_effect = APIError("busy")

        assert _adapter(self._docker(events, worker))._stop() is False

        assert events == [
            "worker stopped",
            "ssh removed",
            "volume removed",
        ]

    def test_a_missing_worker_has_its_ssh_resources_removed(self) -> None:
        events: list[str] = []

        assert _adapter(self._docker(events, NotFound("gone")))._stop() is True

        assert events == ["ssh removed", "volume removed"]


class TestCancelledStart:
    @pytest.mark.asyncio
    async def test_a_start_cancelled_mid_create_is_stopped_by_the_unwind(
        self,
    ) -> None:
        creating = threading.Event()
        release = threading.Event()
        created = MagicMock(status="running")
        docker_client = MagicMock()
        containers: list[MagicMock] = []

        def get(_name: str) -> MagicMock:
            if not containers:
                raise NotFound("gone")
            return containers[0]

        def run(**_: Any) -> MagicMock:
            creating.set()
            # The create outlives the cancel of the start awaiting it.
            release.wait(timeout=5.0)
            containers.append(created)
            return created

        docker_client.containers.get.side_effect = get
        docker_client.containers.run.side_effect = run
        docker_client.containers.list.return_value = []
        docker_client.volumes.list.return_value = []
        adapter = _adapter(docker_client)

        async def create() -> None:
            try:
                await adapter.start()
            except BaseException:
                await adapter.stop()
                raise

        task = asyncio.ensure_future(create())
        await asyncio.to_thread(creating.wait, 5.0)
        task.cancel()
        # An unwind that runs before the create finishes finds no container to stop.
        await asyncio.wait({task}, timeout=0.3)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        created.stop.assert_called_once()
        created.remove.assert_called_once()


class TestVanishingWorker:
    @pytest.mark.parametrize("step", ["stop", "remove"])
    def test_a_worker_gone_mid_stop_has_stopped_and_its_ssh_resources_go(
        self, step: str
    ) -> None:
        events: list[str] = []
        worker = MagicMock()
        getattr(worker, step).side_effect = NotFound("gone")

        docker_client = TestStopOrder()._docker(events, worker)
        assert _adapter(docker_client)._stop() is True

        assert events == ["ssh removed", "volume removed"]


class TestAbandonedStart:
    @pytest.mark.asyncio
    async def test_a_start_that_fails_after_its_cancel_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        creating = threading.Event()
        release = threading.Event()
        docker_client = MagicMock()
        docker_client.containers.get.side_effect = NotFound("gone")
        docker_client.containers.list.return_value = []
        docker_client.volumes.list.return_value = []
        adapter = _adapter(docker_client)

        def start() -> bool:
            creating.set()
            release.wait(timeout=5.0)
            raise RuntimeError("create failed")

        with (
            patch.object(adapter, "_start", side_effect=start),
            caplog.at_level(logging.WARNING, logger="supervisor"),
        ):
            task = asyncio.ensure_future(adapter.start())
            await asyncio.to_thread(creating.wait, 5.0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await adapter.stop()

        assert "failed to start after its start was cancelled" in caplog.text


class TestConcurrentStops:
    @pytest.mark.asyncio
    async def test_a_second_stop_returns_once_the_first_has_stopped_the_worker(
        self,
    ) -> None:
        stopping = threading.Event()
        release = threading.Event()
        worker = MagicMock()

        def stop(**_: Any) -> None:
            stopping.set()
            release.wait(timeout=5.0)

        worker.stop.side_effect = stop
        docker_client = MagicMock()
        docker_client.containers.get.return_value = worker
        docker_client.containers.list.return_value = []
        docker_client.volumes.list.return_value = []
        adapter = _adapter(docker_client)
        adapter.set_status(WorkerStatus.RUNNING)

        first = asyncio.ensure_future(adapter.stop())
        await asyncio.to_thread(stopping.wait, 5.0)
        second = asyncio.ensure_future(adapter.stop())
        done, _ = await asyncio.wait({second}, timeout=0.2)
        assert not done
        release.set()

        assert await second is True
        assert await first is True
        worker.remove.assert_called_once()
