"""The dispatch loop must survive a control-Redis outage.

A dropped Redis connection surfaces as a redis-py error inside `dispatch_once`
and, worse, inside the `except`-branch requeue. `_safe_requeue` swallows the
Redis error so the dispatcher thread keeps running (the connection pool
reconnects on the next command and the watchdog re-surfaces the task).
"""

import logging
from typing import Any
from unittest import mock

import pytest

from server.dispatcher.base import Dispatcher
from tests.server.dispatcher.helpers import make_capturing_dispatcher
from tests.server.task.test_task_merge import (
    _next,
    _register,
    _Registry,
    _runtime,
    _siblings,
)


@pytest.mark.anyio
async def test_safe_requeue_leaves_the_task_queued_when_its_persist_fails() -> None:
    registry = _Registry()
    runtime = _runtime(registry)
    _, _ids = await _register(runtime, _siblings(names=["a"]))
    task_id = _next(runtime)
    registry.down = True

    # Must not raise (a requeue that also hits the outage cannot kill the loop), and
    # the task must stay queued: nothing else re-surfaces it while the server stays up.
    Dispatcher(runtime, mock.Mock(), logging.getLogger("resilience"))._safe_requeue(
        task_id
    )

    assert task_id in runtime._ready_index


def test_safe_requeue_propagates_non_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = make_capturing_dispatcher()

    def _boom(task_id: str, **kwargs: Any) -> None:
        raise ValueError("a real bug, not an outage")

    monkeypatch.setattr(dispatcher, "requeue_task", _boom)
    # A non-Redis error is a genuine defect and must not be silently swallowed.
    with pytest.raises(ValueError, match="real bug"):
        dispatcher._safe_requeue("tsk-1")
