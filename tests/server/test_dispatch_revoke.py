"""A revoke withdraws its dispatch from the worker's queue; a cancel or a stop is
delivered behind the frame it ends."""

import asyncio
import json
import logging
from typing import Any, cast

import pytest

from server.clients.redis import SyncRedisClient
from server.supervisor.services.task_listener import TaskListener
from shared.schemas.command import (
    InterruptMessage,
    RevokeMessage,
    StopMessage,
    TaskMessage,
)
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_worker_task_message

_LOGGER = logging.getLogger("test.dispatch_revoke")


def _task(dispatch_id: str) -> dict[str, Any]:
    message = make_worker_task_message(
        {"taskType": "echo"},
        task_type=TaskType.ECHO,
        task_id="tsk-1",
        assigned_worker="wkr-1",
        dispatch_id=dispatch_id,
    )
    return json.loads(
        TaskMessage(
            worker_id="wkr-1",
            payload=message.model_dump(mode="json", exclude_none=True),
        ).model_dump_json()
    )


async def _frames(listener: TaskListener) -> list[dict[str, Any]]:
    stream = listener.attach_stream("wkr-1")
    assert stream is not None
    listener.remove_worker("wkr-1")
    frames = []
    while (frame := await stream.next()) is not None:
        frames.append(frame)
    return frames


def _listener() -> TaskListener:
    listener = TaskListener(cast(SyncRedisClient, None), "nod-1", _LOGGER)
    listener._loop = asyncio.get_running_loop()
    listener.add_worker("wkr-1")
    return listener


async def _publish(listener: TaskListener, *messages: dict[str, Any]) -> None:
    for message in messages:
        listener._handle_message(message)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_revoked_dispatch_queued_while_the_stream_was_down_never_runs() -> None:
    listener = _listener()
    revoke = RevokeMessage(
        task_id="tsk-1", worker_id="wkr-1", dispatch_id="dsp-1"
    ).model_dump()

    await _publish(listener, _task("dsp-1"), _task("dsp-2"), revoke)
    frames = await _frames(listener)

    assert [
        WorkerTaskMessage.wire_dispatch_id(f) for f in frames if "kind" not in f
    ] == ["dsp-2"]
    assert [f["kind"] for f in frames if "kind" in f] == ["revoke"]


@pytest.mark.asyncio
async def test_a_task_keyed_interrupt_leaves_the_queue_alone() -> None:
    listener = _listener()
    cancel = InterruptMessage(task_id="tsk-1", worker_id="wkr-1").model_dump()
    stop = StopMessage(task_id="tsk-1", worker_id="wkr-1").model_dump()

    await _publish(listener, _task("dsp-1"), cancel, stop)
    frames = await _frames(listener)

    assert [
        f["kind"] if "kind" in f else WorkerTaskMessage.wire_dispatch_id(f)
        for f in frames
    ] == ["dsp-1", "interrupt", "stop"]


@pytest.mark.asyncio
async def test_a_dispatch_keyed_stop_or_cancel_reaches_the_frame_it_ends() -> None:
    listener = _listener()
    stop = StopMessage(
        task_id="tsk-1", worker_id="wkr-1", dispatch_id="dsp-1"
    ).model_dump()
    cancel = InterruptMessage(
        task_id="tsk-1", worker_id="wkr-1", dispatch_id="dsp-1"
    ).model_dump()

    await _publish(listener, _task("dsp-1"), stop, cancel)
    frames = await _frames(listener)

    assert [
        f["kind"] if "kind" in f else WorkerTaskMessage.wire_dispatch_id(f)
        for f in frames
    ] == ["dsp-1", "stop", "interrupt"]
