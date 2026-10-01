"""StreamTasks reads its worker's dispatch queue until the queue is released or taken
over by the worker's next stream."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis
import pytest
from google.protobuf.empty_pb2 import Empty

from server.clients.redis import SyncRedisClient
from server.hooks import PrincipalContext
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.external import (
    ExternalWorkerAdapter,
    ExternalWorkerConfig,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.relay_service import RelayService
from server.supervisor.services.task_listener import TaskListener
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.redis_helpers import fake_sync_client

_LOGGER = logging.getLogger("test.stream_tasks")
_TOKEN = "tok-1"
_NAME = "worker-1"


class _FakeContext:
    def invocation_metadata(self) -> list[tuple[str, str]]:
        return [("authorization", f"Bearer {_TOKEN}")]

    async def abort(self, code: Any, details: str) -> None:
        raise AssertionError(details)


def _servicer(listener: TaskListener) -> tuple[SupervisorServicer, WorkerRegistry]:
    registry = WorkerRegistry(on_worker_id_released=listener.remove_worker)
    registry.add(
        ExternalWorkerAdapter(
            cast(WorkerTokenType, _TOKEN),
            _NAME,
            ExternalWorkerConfig(),
            MagicMock(spec=PrincipalContext),
        )
    )
    servicer = SupervisorServicer(
        registry,
        fake_sync_client(fakeredis.FakeServer()),
        "nde-1",
        "box",
        listener,
        cast(RelayService, MagicMock()),
        MagicMock(),
        _LOGGER,
    )
    return servicer, registry


async def _register(servicer: SupervisorServicer) -> str:
    response = await servicer.RegisterWorker(
        supervisor_pb2.RegisterRequest(), cast(Any, _FakeContext())
    )
    return response.worker_id


def _stream_tasks(
    servicer: SupervisorServicer,
) -> AsyncGenerator[supervisor_pb2.DispatchMessage, None]:
    return cast(
        AsyncGenerator[supervisor_pb2.DispatchMessage, None],
        servicer.StreamTasks(Empty(), cast(Any, _FakeContext())),
    )


async def _read_until_closed(
    stream: AsyncIterator[supervisor_pb2.DispatchMessage],
) -> list[supervisor_pb2.DispatchMessage]:
    return [message async for message in stream]


@pytest.fixture
def listener() -> TaskListener:
    return TaskListener(cast(SyncRedisClient, None), "nde-1", _LOGGER)


@pytest.mark.asyncio
async def test_a_destroyed_worker_s_stream_finishes(listener: TaskListener) -> None:
    listener._loop = asyncio.get_running_loop()
    servicer, registry = _servicer(listener)
    worker_id = await _register(servicer)
    reader = asyncio.ensure_future(_read_until_closed(_stream_tasks(servicer)))
    await asyncio.sleep(0)

    registry.try_pop_by_alias(_NAME)

    assert await asyncio.wait_for(reader, timeout=2) == []
    assert worker_id not in listener._qs


@pytest.mark.asyncio
async def test_a_second_stream_ends_the_first_and_takes_every_frame_kind(
    listener: TaskListener,
) -> None:
    listener._loop = asyncio.get_running_loop()
    servicer, _ = _servicer(listener)
    worker_id = await _register(servicer)
    first = asyncio.ensure_future(_read_until_closed(_stream_tasks(servicer)))
    await asyncio.sleep(0)

    second = _stream_tasks(servicer)
    read = asyncio.ensure_future(anext(second))
    await asyncio.sleep(0)
    listener._deliver(worker_id, {"task_id": "tsk-1"})
    listener._deliver(worker_id, {"kind": "stop", "task_id": "tsk-1", "reason": "r"})
    listener._deliver(
        worker_id,
        {"kind": "mediated_op", "frame_kind": "permit", "payload": {"n": 1}},
    )
    listener._deliver(
        worker_id, {"kind": "revoke", "task_id": "tsk-1", "dispatch_id": "dsp-9"}
    )
    messages = [await asyncio.wait_for(read, timeout=2)]
    messages += [await asyncio.wait_for(anext(second), timeout=2) for _ in range(3)]
    await second.aclose()

    assert await asyncio.wait_for(first, timeout=2) == []
    assert [message.WhichOneof("payload") for message in messages] == [
        "task",
        "stop",
        "mediated_op",
        "revoke",
    ]
    assert messages[0].task.payload["task_id"] == "tsk-1"
    assert messages[2].mediated_op.kind == "permit"
    assert messages[3].revoke.dispatch_id == "dsp-9"


@pytest.mark.asyncio
async def test_re_registering_ends_the_old_id_s_stream(listener: TaskListener) -> None:
    listener._loop = asyncio.get_running_loop()
    servicer, _ = _servicer(listener)
    await _register(servicer)
    old = asyncio.ensure_future(_read_until_closed(_stream_tasks(servicer)))
    await asyncio.sleep(0)

    new_id = await _register(servicer)

    assert await asyncio.wait_for(old, timeout=2) == []
    assert list(listener._qs) == [new_id]
    stream = _stream_tasks(servicer)
    read = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    listener._deliver(new_id, {"task_id": "tsk-1"})
    message = await asyncio.wait_for(read, timeout=2)
    assert message.task.payload["task_id"] == "tsk-1"
    await stream.aclose()


@pytest.mark.asyncio
async def test_a_stream_without_a_queue_finishes(listener: TaskListener) -> None:
    servicer, _ = _servicer(listener)
    worker_id = await _register(servicer)
    del listener._qs[worker_id]

    assert await _read_until_closed(_stream_tasks(servicer)) == []
