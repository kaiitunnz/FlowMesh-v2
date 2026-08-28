import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app_state import get_runtime
from server.auth.security import authenticate_connection
from server.config import OrchestrationConfig
from server.routers.v1 import workflows as workflows_router
from server.task.runtime import TaskRuntime

_V1_WF = """
apiVersion: flowmesh/v1
kind: EchoTask
metadata: {name: t}
spec:
  taskType: echo
  data: {type: list, items: [hi]}
"""

_V2_DAG = _V1_WF.replace("flowmesh/v1", "flowmesh/v2")

_V2_BAD = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {recovery: recompute}
"""

_V2_GUARD_ON_REGION = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: gate
        region: {kind: branch, selection: "x", ports: [p, q]}
      - name: gated
        dependsOn: [gate]
        spec:
          taskType: echo
          condition: {node: gate, field: items.0.output, equals: run}
          data: {type: list, items: [x]}
"""

_V2_REGIONS = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: route
        dependsOn: [a]
        region: {kind: branch, selection: "x", ports: [p, q]}
"""


def _runtime() -> TaskRuntime:
    worker_stub = SimpleNamespace(
        get_worker=lambda wid: SimpleNamespace(id=wid, node_id="nde-1"),
        publish_interrupt=lambda *a: 0,
    )
    registry = SimpleNamespace(
        register_workflow_async=None,
        save_task_states_async=None,
        save_workflow_sched_async=None,
    )
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, worker_stub),
        OrchestrationConfig(),
        logging.getLogger("v2-endpoint-test"),
    )


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(workflows_router.router, prefix="/api/v1")
    app.dependency_overrides[authenticate_connection] = lambda: SimpleNamespace(
        principal_id="p", org_id="o"
    )
    app.dependency_overrides[get_runtime] = _runtime
    return TestClient(app)


def _post(client: TestClient, body: str) -> Any:
    return client.post(
        "/api/v1/workflows/validate",
        content=body,
        headers={"content-type": "text/plain"},
    )


def test_v1_validate_has_no_inspection(client: TestClient) -> None:
    resp = _post(client, _V1_WF)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["inspection"] is None


def test_v2_validate_returns_inspection(client: TestClient) -> None:
    resp = _post(client, _V2_DAG)
    assert resp.status_code == 200
    data = resp.json()
    assert data["inspection"] is not None
    assert data["inspection"]["template"]["operators"]


def test_v2_region_bearing_is_inspectable(client: TestClient) -> None:
    resp = _post(client, _V2_REGIONS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["inspection"]["region_bearing"] is True


def test_v2_invalid_returns_422_with_diagnostics(client: TestClient) -> None:
    resp = _post(client, _V2_BAD)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any("recovery.illegal-recompute" in d for d in detail["diagnostics"])


def test_v2_guard_on_region_returns_422_with_location(client: TestClient) -> None:
    # A structural guard error is a 422 with a readable location, not a 500.
    resp = _post(client, _V2_GUARD_ON_REGION)
    assert resp.status_code == 422
    diagnostics = resp.json()["detail"]["diagnostics"]
    assert any(
        "guard.unknown-node" in d and "graph node 'gated'" in d for d in diagnostics
    )
