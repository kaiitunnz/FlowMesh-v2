"""TaskListener hands dispatches from other threads to per-worker streams without
occupying executor threads."""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from server.supervisor.services.task_listener import TaskListener
from shared.schemas.command import (
    InterruptMessage,
    MediatedOpMessage,
    StopMessage,
    TaskMessage,
)

_LOGGER = logging.getLogger("test.task_listener")
_ANY_OBJECT: Any = None


def _run_in_thread(target: Callable[[], Any]) -> None:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=2)


class _RecordingExecutor(ThreadPoolExecutor):
    """Records submitted work and never runs it, so nothing can occupy a thread."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.submitted: list[Callable[..., Any]] = []

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        self.submitted.append(fn)
        return Future()


def test_dispatch_after_cancelled_get_events_uses_no_executor() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    executor = _RecordingExecutor()
    loop = asyncio.new_event_loop()
    loop.set_default_executor(executor)
    listener._loop = loop

    async def scenario() -> dict[str, Any] | None:
        dead = []
        for i in range(10):
            worker_id = f"wkr-dead-{i}"
            listener.add_worker(worker_id)
            dead.append(asyncio.ensure_future(listener.get_event(worker_id)))
        await asyncio.sleep(0.01)
        for getter in dead:
            getter.cancel()
        await asyncio.gather(*dead, return_exceptions=True)

        listener.add_worker("wkr-live")
        live = asyncio.ensure_future(listener.get_event("wkr-live"))
        await asyncio.sleep(0)
        message = TaskMessage(worker_id="wkr-live", payload={"task_id": "tsk-1"})
        _run_in_thread(lambda: listener._handle_message(message.model_dump()))
        try:
            return await asyncio.wait_for(live, timeout=2)
        except TimeoutError:
            return None

    try:
        result = loop.run_until_complete(scenario())
    finally:
        loop.close()
    assert executor.submitted == []
    assert result == {"task_id": "tsk-1"}


def test_every_frame_kind_from_another_thread_arrives_in_order() -> None:
    listener = TaskListener(_ANY_OBJECT, "nde-1", _LOGGER)
    messages = [
        TaskMessage(worker_id="wkr-1", payload={"task_id": "tsk-1"}),
        InterruptMessage(worker_id="wkr-1", task_id="tsk-1", reason="cancel"),
        StopMessage(worker_id="wkr-1", task_id="tsk-1", reason="stop"),
        MediatedOpMessage(worker_id="wkr-1", frame_kind="permit", payload={"n": 1}),
        MediatedOpMessage(worker_id="wkr-unknown", frame_kind="reap", payload={}),
    ]

    async def scenario() -> list[dict[str, Any]]:
        listener._loop = asyncio.get_running_loop()
        listener.add_worker("wkr-1")

        def publish() -> None:
            for message in messages:
                listener._handle_message(message.model_dump(mode="json"))

        _run_in_thread(publish)
        await asyncio.sleep(0)  # the frames handed over run before the local one
        local = await listener.enqueue_local(
            "wkr-1", {"kind": "mediated_op", "frame_kind": "resident_frame"}
        )
        assert local
        assert not await listener.enqueue_local("wkr-unknown", {"kind": "task"})
        return [
            await asyncio.wait_for(listener.get_event("wkr-1"), timeout=2)
            for _ in range(5)
        ]

    events = asyncio.run(scenario())
    assert [(event.get("kind"), event.get("frame_kind")) for event in events] == [
        (None, None),
        ("interrupt", None),
        ("stop", None),
        ("mediated_op", "permit"),
        ("mediated_op", "resident_frame"),
    ]
