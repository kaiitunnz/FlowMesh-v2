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
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_validation import (
    _CALL,
    _FAN,
    _REGION_CONSUMER,
    _SPAWNED_WORKER,
    SPAWN_DEPENDENTS,
)

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
        region: {kind: merge}
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
        region: {kind: merge}
"""


def _region(region: str) -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: t}}
spec:
  graph:
    nodes:
      - name: a
        spec: {{taskType: echo, data: {{type: list, items: [x]}}}}
      - name: route
        dependsOn: [a]
        region: {region}
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
        make_result_reader(),
        logging.getLogger("v2-endpoint-test"),
        credential_vault=InMemoryCredentialVault(),
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


_V2_TEMPLATES = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    templates:
      - name: body
        inputs: [{name: state, role: carried}]
        nodes:
          - name: step
            dependsOn: [{node: $ingress, port: state, input: state}]
            spec: {taskType: echo, data: {type: list, items: ["${state.output}", done]}}
          - name: judge
            dependsOn: [{node: step, input: input}]
            region:
              kind: branch
              inputs: [{name: input}]
              outputs: [{name: again}, {name: done}]
              selection: {input: input, field: [items, 0, output]}
        edges:
          - from: {node: judge, port: again}
            to: {node: $feedback, port: state}
            project: [items, 1]
          - from: {node: judge, port: done}
            to: {node: $egress, port: state}
    nodes:
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [again, b]}}
      - name: refine
        dependsOn: [{node: seed, input: state, project: [items, 0]}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [seed]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled}
"""


def test_v2_validate_lists_only_the_tasks_a_submission_registers(
    client: TestClient,
) -> None:
    resp = _post(client, _V2_TEMPLATES)
    assert resp.status_code == 200
    data = resp.json()
    assert [task["graph_node_name"] for task in data["tasks"]] == ["seed"]
    assert data["count"] == 1
    template = data["inspection"]["template"]
    operators = {op["operator_id"] for op in template["operators"]}
    assert {task["task_id"] for task in data["tasks"]} <= operators


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


@pytest.mark.parametrize(
    ("region", "code"),
    [
        ("{kind: frobnicate}", "region.unknown-kind"),
        ('{kind: branch, selection: "x", ports: [p, q]}', "region.unknown-field"),
        ("{kind: loop, coordinate: t}", "region.unknown-field"),
    ],
)
def test_v2_malformed_region_returns_422(
    client: TestClient, region: str, code: str
) -> None:
    resp = _post(client, _region(region))
    assert resp.status_code == 422
    diagnostics = resp.json()["detail"]["diagnostics"]
    assert any(code in d for d in diagnostics)


def test_v2_region_input_from_a_spawned_agent_returns_422(client: TestClient) -> None:
    body = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
""" + _SPAWNED_WORKER + _REGION_CONSUMER % "worker"
    resp = _post(client, body)
    assert resp.status_code == 422
    diagnostics = resp.json()["detail"]["diagnostics"]
    assert any("dataflow.spawned-region-output" in d for d in diagnostics)


def test_v2_a_node_depending_on_a_spawn_returns_422(client: TestClient) -> None:
    body = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
""" + _FAN + SPAWN_DEPENDENTS["task"]
    resp = _post(client, body)
    assert resp.status_code == 422
    diagnostics = resp.json()["detail"]["diagnostics"]
    assert any("dataflow.spawn-dependent" in d for d in diagnostics)


def test_v2_a_region_fed_by_a_call_returns_422(client: TestClient) -> None:
    body = (
        """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
"""
        + _CALL
        + """      - name: r
        dependsOn: [c]
        region: {kind: spawn, child: kid}
"""
    )
    resp = _post(client, body)
    assert resp.status_code == 422
    diagnostics = resp.json()["detail"]["diagnostics"]
    assert any("dataflow.region-input" in d for d in diagnostics)
