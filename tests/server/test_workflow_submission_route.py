"""A submission answers from the tasks it registered, without the store."""

import logging
from types import SimpleNamespace
from typing import Any, cast

import fakeredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from server.app_state import get_logger, get_metrics, get_runtime
from server.auth.security import authenticate_connection
from server.config import OrchestrationConfig
from server.registries.workflow import WorkflowRegistry
from server.routers.v1 import workflows as workflows_router
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.redis_helpers import fake_redis_client
from tests.server.result_store import make_result_reader
from tests.server.task.test_v2_orchestration import _WorkerRegistryStub

_LOGGER = logging.getLogger("test.workflow_submission")

_CHAIN = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: chain}
spec:
  graph:
    nodes:
      - {name: a, spec: {taskType: echo, data: {type: list, items: [x]}}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""


def test_a_submission_answers_without_reading_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TaskRuntime(
        WorkflowRegistry(fake_redis_client(fakeredis.FakeServer())),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        _LOGGER,
        credential_vault=InMemoryCredentialVault(),
    )

    def outage(*args: Any) -> Any:
        raise RedisConnectionError("store unavailable")

    monkeypatch.setattr(runtime, "describe_task", outage)
    monkeypatch.setattr(runtime, "_stored_task", outage)
    app = FastAPI()
    app.include_router(workflows_router.router, prefix="/api/v1")
    app.dependency_overrides[authenticate_connection] = lambda: SimpleNamespace(
        principal_id="p", org_id="o"
    )
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.dependency_overrides[get_logger] = lambda: _LOGGER
    app.dependency_overrides[get_metrics] = lambda: SimpleNamespace(
        record_task_event=lambda event: None
    )

    response = TestClient(app).post(
        "/api/v1/workflows", content=_CHAIN, headers={"content-type": "text/plain"}
    )

    assert response.status_code == 201, response.text
    tasks = response.json()["tasks"]
    assert [task["status"] for task in tasks] == ["PENDING", "PENDING"]
    assert tasks[1]["depends_on"] == [tasks[0]["task_id"]]
