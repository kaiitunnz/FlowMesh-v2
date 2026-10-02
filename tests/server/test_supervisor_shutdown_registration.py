"""A stopping supervisor admits no worker, so a worker that registers while it stops
does so with the next supervisor and holds exactly one new id."""

from typing import Any, cast

import grpc
import pytest

from server.clients.redis import WORKERS_SET_KEY
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.servicer_helpers import Aborted, ServicerHarness, WorkerContext


@pytest.mark.asyncio
async def test_a_stopping_supervisor_refuses_a_registration() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    harness.registry.retire(worker_id)
    harness.relay.events.clear()

    harness.servicer.begin_shutdown()
    with pytest.raises(Aborted) as refused:
        await harness.servicer.RegisterWorker(
            supervisor_pb2.RegisterRequest(), cast(Any, WorkerContext())
        )

    assert refused.value.code is grpc.StatusCode.UNAVAILABLE
    assert harness.registry.get_worker_id(harness.adapter.token) is None
    assert harness.relay.events == []


@pytest.mark.asyncio
async def test_a_worker_refused_while_its_supervisor_stops_gets_one_new_id() -> None:
    stopping = ServicerHarness()
    first = await stopping.register()
    stopping.registry.retire(first)
    stopping.servicer.begin_shutdown()
    with pytest.raises(Aborted):
        await stopping.servicer.RegisterWorker(
            supervisor_pb2.RegisterRequest(), cast(Any, WorkerContext())
        )

    # The worker retries against the next supervisor on the same store, which takes
    # the next id: the refused registration allocated none.
    after = ServicerHarness(stopping.server)
    second = await after.register()

    assert (first, second) == ("wkr-1", "wkr-2")


@pytest.mark.asyncio
async def test_a_stopping_supervisor_releases_no_binding() -> None:
    harness = ServicerHarness()
    worker_id = await harness.register()
    harness.rds.srem(WORKERS_SET_KEY, worker_id)

    harness.servicer.begin_shutdown()
    harness.servicer.reconcile_workers()

    assert harness.released == []
    assert harness.registry.get_worker_id(harness.adapter.token) == worker_id
