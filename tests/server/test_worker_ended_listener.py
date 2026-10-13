"""A supervisor tells its listener each worker id that ends on its node."""

import asyncio
from typing import Any, cast

import pytest

from server.supervisor.services import grpc_server
from tests.server.servicer_helpers import ServicerHarness


def test_a_released_worker_id_is_announced_as_ended() -> None:
    harness = ServicerHarness()
    ended: list[str] = []
    harness.servicer.on_worker_ended(ended.append)

    harness.servicer.worker_id_released("wkr-gone")

    assert ended == ["wkr-gone"]


def test_a_worker_whose_stream_closed_for_good_is_announced_as_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grpc_server, "_REATTACH_GRACE_SEC", 0.0)
    harness = ServicerHarness()
    ended: list[str] = []
    harness.servicer.on_worker_ended(ended.append)

    async def run() -> None:
        worker_id = await harness.register()
        await harness.servicer._unregister_unless_reattached(
            cast(Any, harness.adapter), worker_id
        )
        assert ended == [worker_id]

    asyncio.run(run())


def test_a_failing_listener_never_fails_the_release() -> None:
    harness = ServicerHarness()

    def broken(worker_id: str) -> None:
        raise RuntimeError("listener down")

    harness.servicer.on_worker_ended(broken)
    harness.servicer.worker_id_released("wkr-gone")
    assert harness.relay.unregisters() == ["wkr-gone"]
