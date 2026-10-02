"""The server WorkerRegistry's sync and async twins behave alike over fakeredis."""

import asyncio

import fakeredis
import pytest

from server.clients.redis import (
    WORKERS_SET_KEY,
    worker_hb_key,
    worker_key,
)
from server.registries.worker import WorkerRegistry
from tests.server.redis_helpers import fake_redis_client

_WORKER = "wkr-1"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def _record(server: fakeredis.FakeServer, node_alias: str) -> fakeredis.FakeRedis:
    rds = fakeredis.FakeRedis(server=server, decode_responses=True)
    rds.sadd(WORKERS_SET_KEY, _WORKER)
    rds.hset(worker_key(_WORKER), mapping={"node_alias": node_alias})
    rds.setex(worker_hb_key(_WORKER), 120, "ts")
    return rds


def _unregister(registry: WorkerRegistry, node_alias: str, use_async: bool) -> bool:
    if use_async:
        return asyncio.run(registry.unregister_node_worker_async(_WORKER, node_alias))
    return registry.unregister_node_worker(_WORKER, node_alias)


@pytest.mark.parametrize("use_async", [False, True])
def test_a_node_unregisters_a_worker_record_it_wrote(
    server: fakeredis.FakeServer, use_async: bool
) -> None:
    rds = _record(server, "box-a")

    assert _unregister(WorkerRegistry(fake_redis_client(server)), "box-a", use_async)

    assert not rds.exists(worker_key(_WORKER), worker_hb_key(_WORKER))
    assert not rds.sismember(WORKERS_SET_KEY, _WORKER)


@pytest.mark.parametrize("use_async", [False, True])
def test_a_node_leaves_a_worker_record_another_node_wrote(
    server: fakeredis.FakeServer, use_async: bool
) -> None:
    rds = _record(server, "box-b")

    assert not _unregister(
        WorkerRegistry(fake_redis_client(server)), "box-a", use_async
    )

    assert rds.hget(worker_key(_WORKER), "node_alias") == "box-b"
    assert rds.sismember(WORKERS_SET_KEY, _WORKER)
