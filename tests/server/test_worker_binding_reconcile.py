"""A supervisor's worker bindings and the root's worker records end together."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import grpc
import pytest

from server.clients.redis import WORKERS_SET_KEY, worker_key
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.registry import WorkerRegistry
from server.supervisor.schemas import WorkerStatus
from server.supervisor.services import grpc_server as grpc_server_module
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.servicer_helpers import (
    ALIAS,
    NODE_ALIAS,
    Aborted,
    ServicerHarness,
    WorkerContext,
)


async def _events(
    *types: str, worker_id: str = "wkr-1"
) -> AsyncIterator[supervisor_pb2.EventMessage]:
    for event_type in types:
        message = supervisor_pb2.EventMessage()
        message.payload.update({"type": event_type, "worker_id": worker_id})
        yield message
        await asyncio.sleep(0)


def test_retire_unbinds_only_the_binding_still_held() -> None:
    released: list[str] = []
    registry = WorkerRegistry(on_worker_id_released=released.append)
    registry.set_worker_id(cast(WorkerTokenType, "tok"), "wkr-2")

    assert registry.retire("wkr-1") is False
    assert registry.retire("wkr-2") is True
    assert registry.get_worker_id(cast(WorkerTokenType, "tok")) is None
    assert released == ["wkr-2"]


@pytest.mark.asyncio
async def test_a_worker_the_root_no_longer_records_is_released() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    other = cast(WorkerTokenType, "tok-2")
    harness.registry.set_worker_id(other, "wkr-live")
    harness.rds.sadd(WORKERS_SET_KEY, "wkr-live")
    harness.rds.hset(worker_key("wkr-live"), mapping={"node_alias": NODE_ALIAS})
    harness.rds.srem(WORKERS_SET_KEY, worker_id)

    harness.servicer.reconcile_workers()

    assert harness.released == [worker_id]
    assert harness.registry.get_worker_id(harness.adapter.token) is None
    assert harness.registry.get_worker_id(other) == "wkr-live"
    assert harness.adapter.worker_id is None


@pytest.mark.asyncio
async def test_a_binding_whose_record_another_node_wrote_is_released() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    harness.rds.hset(worker_key(worker_id), mapping={"node_alias": "elsewhere"})

    harness.servicer.reconcile_workers()

    assert harness.released == [worker_id]


@pytest.mark.asyncio
async def test_rehoming_leaves_a_record_another_node_wrote() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    harness.rds.hset(
        worker_key(worker_id), mapping={"node_alias": "elsewhere", "node_id": "nod-9"}
    )

    harness.servicer.rebind_node("nod-2")

    assert harness.rds.hget(worker_key(worker_id), "node_id") == "nod-9"


@pytest.mark.asyncio
async def test_an_event_stream_ends_with_its_binding() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()

    async def events() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("HEARTBEAT"):
            yield message
        harness.registry.retire(worker_id)
        async for message in _events("HEARTBEAT"):
            yield message

    with pytest.raises(Aborted) as aborted:
        await harness.servicer.PushEvents(events(), cast(Any, WorkerContext()))

    assert aborted.value.code is grpc.StatusCode.UNAUTHENTICATED
    assert [e["type"] for e in harness.relay.events] == ["HEARTBEAT", "UNREGISTER"]


@pytest.mark.asyncio
async def test_a_destroyed_worker_is_unregistered_once() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()

    harness.registry.try_pop_by_alias(ALIAS)

    assert harness.relay.unregisters() == [worker_id]


@pytest.mark.asyncio
async def test_a_worker_that_unregistered_itself_is_not_unregistered_again() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    await harness.servicer.PushEvents(
        _events("REGISTER", "UNREGISTER"), cast(Any, WorkerContext())
    )

    harness.registry.try_pop_by_alias(ALIAS)

    assert harness.relay.unregisters() == [worker_id]


@pytest.mark.asyncio
async def test_a_reconnecting_event_stream_keeps_its_worker_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server_module, "_REATTACH_GRACE_SEC", 0.2)
    harness = ServicerHarness()
    await harness.register()
    await harness.servicer.PushEvents(_events("REGISTER"), cast(Any, WorkerContext()))

    reattached = asyncio.Event()

    async def held_open() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("HEARTBEAT"):
            yield message
        reattached.set()
        await asyncio.sleep(0.5)

    stream = asyncio.ensure_future(
        harness.servicer.PushEvents(held_open(), cast(Any, WorkerContext()))
    )
    await reattached.wait()
    await asyncio.sleep(0.3)
    assert harness.relay.unregisters() == []
    await stream


@pytest.mark.asyncio
async def test_an_event_stream_that_stays_closed_unregisters_its_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server_module, "_REATTACH_GRACE_SEC", 0.05)
    harness = ServicerHarness()
    worker_id = await harness.register()
    await harness.servicer.PushEvents(_events("REGISTER"), cast(Any, WorkerContext()))
    assert harness.relay.unregisters() == []

    await asyncio.sleep(0.2)

    assert harness.relay.unregisters() == [worker_id]


@pytest.mark.asyncio
async def test_a_reconnected_event_stream_restores_its_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server_module, "_REATTACH_GRACE_SEC", 0.2)
    harness = ServicerHarness()
    worker_id = await harness.register()
    await harness.servicer.PushEvents(_events("REGISTER"), cast(Any, WorkerContext()))
    reattached = asyncio.Event()

    async def held_open() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("REGISTER", "HEARTBEAT"):
            yield message
        reattached.set()
        await asyncio.sleep(0.5)

    stream = asyncio.ensure_future(
        harness.servicer.PushEvents(held_open(), cast(Any, WorkerContext()))
    )
    await reattached.wait()
    info = harness.adapter.get_info()

    assert (info.id, info.status) == (worker_id, WorkerStatus.RUNNING)
    assert [e["type"] for e in harness.relay.events] == ["REGISTER", "HEARTBEAT"]
    await stream


@pytest.mark.asyncio
async def test_a_stream_of_an_earlier_registration_leaves_the_new_one_alone() -> None:
    harness = ServicerHarness()
    await harness.register()
    opened = asyncio.Event()
    release = asyncio.Event()

    async def earlier() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("REGISTER"):
            yield message
        opened.set()
        await release.wait()

    stream = asyncio.ensure_future(
        harness.servicer.PushEvents(earlier(), cast(Any, WorkerContext()))
    )
    await opened.wait()
    new_id = await harness.register()
    release.set()
    await stream

    assert harness.adapter.worker_id == new_id


@pytest.mark.asyncio
async def test_a_log_is_attributed_to_the_worker_its_stream_authenticated() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()

    async def logs() -> AsyncIterator[supervisor_pb2.LogMessage]:
        message = supervisor_pb2.LogMessage()
        message.payload.update({"line": "hello", "worker_id": "wkr-other"})
        yield message

    await harness.servicer.PushLogs(logs(), cast(Any, WorkerContext()))

    assert [log["worker_id"] for log in harness.relay.logs] == [worker_id]
