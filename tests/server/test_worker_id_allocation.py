"""A worker id is never handed out while a worker still holds it."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis
import pytest

from server.clients.redis import WORKER_ID_SEQ_KEY, WORKERS_SET_KEY
from server.registries.worker import WorkerRegistry as RootWorkerRegistry
from server.supervisor.adapters.base import WorkerAdapter, WorkerTokenType
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.redis_helpers import fake_redis_client, fake_sync_client

_LOGGER = logging.getLogger("test.worker_id_allocation")


class _Adapter:
    def __init__(self, token: str) -> None:
        self.token = cast(WorkerTokenType, token)
        self.alias = token

    def set_worker_id(self, worker_id: str) -> None:
        pass


class _Context:
    def __init__(self, token: str) -> None:
        self._token = token

    def invocation_metadata(self) -> list[tuple[str, str]]:
        return [("authorization", f"Bearer {self._token}")]

    async def abort(self, code: Any, details: str) -> None:
        raise AssertionError(details)


def _servicer(
    server: fakeredis.FakeServer, released: list[str]
) -> tuple[SupervisorServicer, WorkerRegistry]:
    client = fake_sync_client(server)
    registry = WorkerRegistry(on_worker_id_released=released.append)
    for token in ("tok-a", "tok-b"):
        registry.add(cast(WorkerAdapter, _Adapter(token)))
    servicer = SupervisorServicer(
        registry,
        client,
        "nod-1",
        "box",
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _LOGGER,
    )
    return servicer, registry


async def _register(servicer: SupervisorServicer, token: str) -> str:
    response = await servicer.RegisterWorker(
        supervisor_pb2.RegisterRequest(), cast(Any, _Context(token))
    )
    return response.worker_id


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def rds(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.mark.asyncio
async def test_a_lost_counter_skips_the_ids_recorded_workers_hold(
    server: fakeredis.FakeServer, rds: fakeredis.FakeRedis
) -> None:
    servicer, _ = _servicer(server, [])
    rds.sadd(WORKERS_SET_KEY, "wkr-1", "wkr-2")

    assert await _register(servicer, "tok-a") == "wkr-3"
    assert rds.get(WORKER_ID_SEQ_KEY) == "3"


@pytest.mark.asyncio
async def test_a_wiped_store_never_hands_out_an_id_bound_here(
    server: fakeredis.FakeServer, rds: fakeredis.FakeRedis
) -> None:
    released: list[str] = []
    servicer, _ = _servicer(server, released)
    first = await _register(servicer, "tok-a")
    rds.flushall()

    second = await _register(servicer, "tok-b")
    again = await _register(servicer, "tok-a")

    assert second != first
    assert released == [first]
    assert again not in (first, second)


@pytest.mark.asyncio
async def test_a_worker_the_root_registers_skips_the_ids_recorded_workers_hold(
    server: fakeredis.FakeServer, rds: fakeredis.FakeRedis
) -> None:
    registry = RootWorkerRegistry(fake_redis_client(server))
    rds.sadd(WORKERS_SET_KEY, "wkr-1", "wkr-2")
    rds.hset("worker:wkr-1", mapping={"node_alias": "box"})

    first = await registry.register_worker_async("nod-1", "box", {})
    second = registry.register_worker("nod-1", "box", {})

    assert (first, second) == ("wkr-3", "wkr-4")
    assert rds.hget("worker:wkr-1", "node_alias") == "box"
    assert rds.smembers(WORKERS_SET_KEY) == {"wkr-1", "wkr-2", "wkr-3", "wkr-4"}
