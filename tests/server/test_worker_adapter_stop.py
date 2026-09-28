"""A worker adapter stops what it started, whatever the worker's status reads.

The status reads STOPPED whenever the worker's event stream closes, as after a worker
crash or a reconnect, while its container or instance still runs.
"""

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
