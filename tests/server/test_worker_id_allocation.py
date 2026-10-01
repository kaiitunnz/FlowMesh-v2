"""A worker id is never handed out while a worker still holds it."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis
import pytest

from server.clients.redis import WORKER_ID_SEQ_KEY, WORKERS_SET_KEY, SyncRedisClient
from server.supervisor.adapters.base import WorkerAdapter, WorkerTokenType
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from shared.grpc.supervisor.v1 import supervisor_pb2

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
    rds: fakeredis.FakeRedis, released: list[str]
) -> tuple[SupervisorServicer, WorkerRegistry]:
    client = SyncRedisClient.__new__(SyncRedisClient)
    client._control = rds
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
def rds() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.mark.asyncio
async def test_a_lost_counter_skips_the_ids_recorded_workers_hold(
    rds: fakeredis.FakeRedis,
) -> None:
    servicer, _ = _servicer(rds, [])
    rds.sadd(WORKERS_SET_KEY, "wkr-1", "wkr-2")

    assert await _register(servicer, "tok-a") == "wkr-3"
    assert rds.get(WORKER_ID_SEQ_KEY) == "3"


@pytest.mark.asyncio
async def test_a_wiped_store_never_hands_out_an_id_bound_here(
    rds: fakeredis.FakeRedis,
) -> None:
    released: list[str] = []
    servicer, _ = _servicer(rds, released)
    first = await _register(servicer, "tok-a")
    rds.flushall()

    second = await _register(servicer, "tok-b")
    again = await _register(servicer, "tok-a")

    assert second != first
    assert released == [first]
    assert again not in (first, second)
