"""Await a condition another task or thread brings about."""

import asyncio
import threading
from collections.abc import Callable

from server.task.runtime import TaskRuntime


async def until(
    condition: Callable[[], bool], timeout: float = 2.0, interval: float = 0.01
) -> None:
    """Poll ``condition`` on the running loop until it holds; fail past ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "condition not reached in time"
        await asyncio.sleep(interval)


def pop_ready(runtime: TaskRuntime, timeout: float = 1.0) -> str | None:
    """The next ready task, or None once ``timeout`` passes with none ready."""
    stop = threading.Event()
    timer = threading.Timer(timeout, stop.set)
    timer.start()
    try:
        return runtime.next_ready(stop, timeout=0.01)
    finally:
        timer.cancel()
