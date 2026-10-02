"""A worker id is never handed out while a worker still holds it."""

from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import WORKER_ID_SEQ_KEY, WORKERS_SET_KEY, worker_key
from server.registries.worker import WorkerRegistry as RootWorkerRegistry
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.redis_helpers import fake_redis_client
from tests.server.servicer_helpers import external_adapter, supervisor_servicer


class _Context:
    def __init__(self, token: str) -> None:
        self._token = token

    def invocation_metadata(self) -> list[tuple[str, str]]:
        return [("authorization", f"Bearer {self._token}")]

    async def abort(self, code: Any, details: str) -> None:
        raise AssertionError(details)


def _servicer(
    server: fakeredis.FakeServer,
    released: list[str],
    node_id: str = "nod-1",
    node_alias: str = "box",
) -> tuple[SupervisorServicer, WorkerRegistry]:
    client = fake_redis_client(server)
    servicer: SupervisorServicer | None = None

    def on_released(worker_id: str) -> None:
        released.append(worker_id)
        assert servicer is not None
        servicer.worker_id_released(worker_id)

    registry = WorkerRegistry(on_worker_id_released=on_released)
    for token in ("tok-a", "tok-b"):
        registry.add(external_adapter(token, token))
    servicer = supervisor_servicer(registry, client, node_id, node_alias)
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


@pytest.mark.asyncio
async def test_after_a_wipe_a_node_releases_the_id_another_node_took(
    server: fakeredis.FakeServer, rds: fakeredis.FakeRedis
) -> None:
    a, _ = _servicer(server, [], "nod-a", "box-a")
    b, _ = _servicer(server, [], "nod-b", "box-b")
    held = await _register(a, "tok-a")
    rds.flushall()

    taken = await _register(b, "tok-a")
    a.reconcile_workers()
    a.rebind_node("nod-a2")

    assert taken == held
    assert taken not in a._registry.bound_worker_ids()
    assert rds.hget(worker_key(taken), "node_id") == "nod-b"


@pytest.mark.asyncio
async def test_a_worker_released_after_a_wipe_registers_under_a_fresh_id(
    server: fakeredis.FakeServer, rds: fakeredis.FakeRedis
) -> None:
    released: list[str] = []
    servicer, _ = _servicer(server, released)
    first = await _register(servicer, "tok-a")
    rds.flushall()

    servicer.reconcile_workers()
    again = await _register(servicer, "tok-a")

    assert released == [first]
    assert again != first
