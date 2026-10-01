"""A supervisor's worker bindings and the root's worker records end together."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock

import grpc
import pytest

from server.clients.redis import WORKERS_SET_KEY, SyncRedisClient
from server.supervisor.adapters.base import WorkerAdapter, WorkerTokenType
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services import grpc_server as grpc_server_module
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.relay_service import RelayService
from shared.grpc.supervisor.v1 import supervisor_pb2

_LOGGER = logging.getLogger("test.worker_binding_reconcile")
_TOKEN = "tok-1"
_ALIAS = "worker-1"


class _Redis:
    def __init__(self) -> None:
        self._seq = 0
        self.workers: set[str] = set()

    def incr(self, key: str) -> int:
        self._seq += 1
        return self._seq

    def sadd(self, key: str, *members: str) -> None:
        assert key == WORKERS_SET_KEY
        self.workers.update(members)

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        pass

    def set_members(self, key: str) -> set[str]:
        assert key == WORKERS_SET_KEY
        return set(self.workers)


class _Relay(RelayService):
    def __init__(self) -> None:
        super().__init__(cast(SyncRedisClient, None), _LOGGER)
        self.events: list[dict[str, Any]] = []

    def add_event(self, event_data: Any) -> None:
        self.events.append(dict(event_data))

    def unregisters(self) -> list[str]:
        return [e["worker_id"] for e in self.events if e["type"] == "UNREGISTER"]


class _Adapter:
    def __init__(self) -> None:
        self.token = cast(WorkerTokenType, _TOKEN)
        self.alias = _ALIAS
        self._streams = 0

    @property
    def has_event_stream(self) -> bool:
        return self._streams > 0

    def attach_event_stream(self) -> None:
        self._streams += 1

    def detach_event_stream(self) -> None:
        self._streams -= 1

    def set_worker_id(self, worker_id: str) -> None:
        pass

    def clear_worker_id(self) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass


class _Aborted(Exception):
    def __init__(self, code: grpc.StatusCode) -> None:
        self.code = code


class _Context:
    def invocation_metadata(self) -> list[tuple[str, str]]:
        return [("authorization", f"Bearer {_TOKEN}")]

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise _Aborted(code)


class _Harness:
    def __init__(self) -> None:
        self.redis = _Redis()
        self.relay = _Relay()
        self.released: list[str] = []
        self.registry = WorkerRegistry(on_worker_id_released=self._released)
        self.adapter = _Adapter()
        self.registry.add(cast(WorkerAdapter, self.adapter))
        listener = MagicMock()
        self.servicer = SupervisorServicer(
            self.registry,
            cast(SyncRedisClient, self.redis),
            "nod-1",
            "box",
            listener,
            self.relay,
            MagicMock(),
            _LOGGER,
        )

    def _released(self, worker_id: str) -> None:
        self.released.append(worker_id)
        self.servicer.worker_id_released(worker_id)

    async def register(self) -> str:
        response = await self.servicer.RegisterWorker(
            supervisor_pb2.RegisterRequest(), cast(Any, _Context())
        )
        return response.worker_id


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
    harness = _Harness()
    worker_id = await harness.register()
    other = cast(WorkerTokenType, "tok-2")
    harness.registry.set_worker_id(other, "wkr-live")
    harness.redis.workers.add("wkr-live")
    harness.redis.workers.discard(worker_id)

    harness.servicer.reconcile_workers()

    assert harness.released == [worker_id]
    assert harness.registry.get_worker_id(harness.adapter.token) is None
    assert harness.registry.get_worker_id(other) == "wkr-live"


@pytest.mark.asyncio
async def test_an_event_stream_ends_with_its_binding() -> None:
    harness = _Harness()
    worker_id = await harness.register()

    async def events() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("HEARTBEAT"):
            yield message
        harness.registry.retire(worker_id)
        async for message in _events("HEARTBEAT"):
            yield message

    with pytest.raises(_Aborted) as aborted:
        await harness.servicer.PushEvents(events(), cast(Any, _Context()))

    assert aborted.value.code is grpc.StatusCode.UNAUTHENTICATED
    assert [e["type"] for e in harness.relay.events] == ["HEARTBEAT", "UNREGISTER"]


@pytest.mark.asyncio
async def test_a_destroyed_worker_is_unregistered_once() -> None:
    harness = _Harness()
    worker_id = await harness.register()

    harness.registry.try_pop_by_alias(_ALIAS)

    assert harness.relay.unregisters() == [worker_id]


@pytest.mark.asyncio
async def test_a_worker_that_unregistered_itself_is_not_unregistered_again() -> None:
    harness = _Harness()
    worker_id = await harness.register()
    await harness.servicer.PushEvents(
        _events("REGISTER", "UNREGISTER"), cast(Any, _Context())
    )

    harness.registry.try_pop_by_alias(_ALIAS)

    assert harness.relay.unregisters() == [worker_id]


@pytest.mark.asyncio
async def test_a_reconnecting_event_stream_keeps_its_worker_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server_module, "_REATTACH_GRACE_SEC", 0.2)
    harness = _Harness()
    await harness.register()
    await harness.servicer.PushEvents(_events("REGISTER"), cast(Any, _Context()))

    reattached = asyncio.Event()

    async def held_open() -> AsyncIterator[supervisor_pb2.EventMessage]:
        async for message in _events("HEARTBEAT"):
            yield message
        reattached.set()
        await asyncio.sleep(0.5)

    stream = asyncio.ensure_future(
        harness.servicer.PushEvents(held_open(), cast(Any, _Context()))
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
    harness = _Harness()
    worker_id = await harness.register()
    await harness.servicer.PushEvents(_events("REGISTER"), cast(Any, _Context()))
    assert harness.relay.unregisters() == []

    await asyncio.sleep(0.2)

    assert harness.relay.unregisters() == [worker_id]
