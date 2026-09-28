"""A thread run to completion finishes before a cancel of the coroutine awaiting it
propagates."""

import asyncio
import threading

import pytest

from server.utils.concurrent import run_to_completion


@pytest.mark.asyncio
async def test_a_cancel_waits_for_the_thread_to_finish() -> None:
    started = threading.Event()
    release = threading.Event()
    finished: list[bool] = []

    def work() -> None:
        started.set()
        release.wait(timeout=5.0)
        finished.append(True)

    task = asyncio.ensure_future(run_to_completion(work))
    await asyncio.to_thread(started.wait, 5.0)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=0.2)
    assert not done
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]


@pytest.mark.asyncio
async def test_the_thread_result_and_error_are_returned() -> None:
    assert await run_to_completion(lambda: 7) == 7

    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_to_completion(fail)
