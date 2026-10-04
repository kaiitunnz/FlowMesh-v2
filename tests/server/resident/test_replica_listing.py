"""The replica listing names the worker running each replica and whether it stands."""

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.resident.state import ReplicaIncarnation
from server.routers.v1 import resident as resident_router
from tests.server.resident.node_harness import Node


async def _replicas(node: Node, **params: str) -> dict[str, dict]:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.replica_listing")
    app.state.resident_control = node.control
    app.include_router(resident_router.router, prefix="/api/v1")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/resident/replicas", params=params)
    assert response.status_code == 200
    return {replica["replica_id"]: replica for replica in response.json()}


@pytest.mark.anyio
async def test_a_listed_replica_names_the_worker_running_its_serve_task() -> None:
    node = Node()
    demand = await node.warm_async()
    serve_task_id = await node.submit_serve_async()
    node.serve(serve_task_id)
    node.delivery.serve_workers[serve_task_id] = "wkr-2"
    standing = node.adopt_standing(serve_task_id)

    listed = await _replicas(node)

    assert listed[demand.replica_id]["worker_id"] == "wkr-replica"
    assert listed[demand.replica_id]["standing"] is False
    assert listed[standing.replica_id]["worker_id"] == "wkr-2"
    assert listed[standing.replica_id]["standing"] is True
    assert set(await _replicas(node, worker_id="wkr-2")) == {standing.replica_id}


def test_a_replica_stored_with_a_worker_field_still_loads() -> None:
    stored = {
        "replica_id": "rpl-1",
        "family": "fam",
        "incarnation": 1,
        "worker_id": None,
    }
    assert ReplicaIncarnation.model_validate(stored).replica_id == "rpl-1"
