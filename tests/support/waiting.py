"""Await a condition another task or thread brings about."""

import asyncio
from collections.abc import Callable


async def until(
    condition: Callable[[], bool], timeout: float = 2.0, interval: float = 0.01
) -> None:
    """Poll ``condition`` on the running loop until it holds; fail past ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "condition not reached in time"
        await asyncio.sleep(interval)
