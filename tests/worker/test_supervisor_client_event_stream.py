"""A worker's event stream runs its ready callback on each connect, before anything
queued is sent."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

from tests.worker.test_supervisor_client_dispatch_id import _client
from worker import supervisor_client as supervisor_module


def test_the_ready_callback_runs_before_the_stream_sends() -> None:
    client = _client()
    order: list[str] = []
    client._channel = cast(Any, object())
    client._shutdown.clear()

    def push(*_: Any, **__: Any) -> None:
        order.append("sent")
        client._shutdown.set()

    client._stub = cast(Any, MagicMock())
    client._stub.PushEvents.side_effect = push
    client.on_event_stream_ready(lambda: order.append("ready"))

    with patch.object(supervisor_module.grpc, "channel_ready_future"):
        client._run_event_stream()

    assert order == ["ready", "sent"]
