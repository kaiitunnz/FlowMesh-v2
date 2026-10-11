"""A workflow's source is stored once beside its tasks, which name it by digest."""

import asyncio
import json
import logging
from typing import Any, cast

import fakeredis
import pydantic
import pytest

from server.clients.redis import task_state_key, workflow_sources_key
from server.config import OrchestrationConfig
from server.registries.workflow import WorkflowRegistry
from server.task import models
from server.task.models import TaskStatus, source_digest
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import fake_redis_client, raw_control
from tests.server.result_store import make_result_reader
from tests.server.task.test_runtime_control_flow import _CONSUMED, _LOOP, _workflow
from tests.server.task.test_v2_orchestration import _WorkerRegistryStub

_V1 = """apiVersion: flowmesh/v1
kind: EchoTask
metadata: {name: sources}
spec:
  taskType: echo
  stages:
    - name: first
      spec:
        data: {type: list, items: [a]}
    - name: second
      spec:
        data: {type: list, items: [b]}
"""


def _runtime(registry: WorkflowRegistry) -> TaskRuntime:
    return TaskRuntime(
        registry,
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        make_result_reader(),
        logging.getLogger("test.workflow_sources"),
        credential_vault=InMemoryCredentialVault(),
        redrive=lambda fire, logger: StoreRedriveScheduler(
            fire, logger, run_thread=False
        ),
    )


def _registry() -> WorkflowRegistry:
    return WorkflowRegistry(fake_redis_client(fakeredis.FakeServer()))


def _register(runtime: TaskRuntime, text: str) -> tuple[str, list[str]]:
    workflow_id, results = asyncio.run(
        runtime.register("owner", "org", text, format="native")
    )
    return workflow_id, [result.task_id for result in results]


@pytest.mark.parametrize("text", [_V1, _workflow(_CONSUMED, _LOOP)])
def test_a_workflow_stores_its_source_once_beside_its_tasks(text: str) -> None:
    registry = _registry()
    workflow_id, task_ids = _register(_runtime(registry), text)
    raw = raw_control(registry)

    sources = raw.hgetall(workflow_sources_key(workflow_id))
    (source,) = sources.values()
    assert list(sources) == [source_digest(source)]
    for task_id in task_ids:
        stored = json.loads(raw.get(task_state_key(task_id)))
        assert "raw_yaml" not in stored["record"]
        assert stored["source_digest"] == source_digest(source)

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1
    records = [restored.get_record(task_id) for task_id in task_ids]
    assert all(r is not None and r.raw_yaml == source for r in records)


def test_a_task_naming_a_source_the_store_lost_does_not_load() -> None:
    registry = _registry()
    workflow_id, task_ids = _register(_runtime(registry), _V1)
    raw_control(registry).delete(workflow_sources_key(workflow_id))

    with pytest.raises(pydantic.ValidationError, match="is not stored"):
        registry.load_task_states(workflow_id, *task_ids)
    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 0
    stored = registry.get_workflow_record(workflow_id)
    assert stored is not None and stored.control_failure.startswith(
        "UnsupportedWorkflowVersion"
    )


def test_a_task_stored_with_its_source_inline_loads_and_names_it_once_written() -> None:
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, task_ids = _register(runtime, _V1)
    raw = raw_control(registry)
    # A store written before sources moved out of the task states.
    for task_id in task_ids:
        record = runtime.get_record(task_id)
        assert record is not None
        stored = json.loads(raw.get(task_state_key(task_id)))
        del stored["source_digest"]
        stored["record"]["raw_yaml"] = record.raw_yaml
        raw.set(task_state_key(task_id), json.dumps(stored))
    raw.delete(workflow_sources_key(workflow_id))

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1
    first = task_ids[0]
    source = restored.get_record(first)
    assert source is not None
    record_dispatch(restored, first)

    loaded = registry.load_task_states(workflow_id, first)[0]
    assert loaded is not None and loaded.record.status == TaskStatus.DISPATCHED
    assert loaded.record.raw_yaml == source.raw_yaml
    assert "source_digest" in json.loads(raw.get(task_state_key(first)))


def test_unregistering_a_workflow_removes_its_sources() -> None:
    registry = _registry()
    workflow_id, _ = _register(_runtime(registry), _V1)

    registry.unregister_workflows(workflow_id)

    assert not raw_control(registry).exists(workflow_sources_key(workflow_id))


def test_a_workflow_digests_its_source_once_however_often_its_tasks_are_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digested: list[str] = []

    def counted(source: str) -> str:
        digested.append(source)
        return source_digest(source)

    monkeypatch.setattr(models, "source_digest", counted)
    registry = _registry()
    runtime = _runtime(registry)
    workflow_id, task_ids = _register(runtime, _workflow(_CONSUMED, _LOOP))
    for task_id in task_ids:
        record_dispatch(runtime, task_id)
    assert len(digested) == 1

    restored = _runtime(registry)
    assert asyncio.run(restored.rehydrate()) == 1
    for task_id in task_ids:
        record = restored.get_record(task_id)
        assert record is not None and record.status == TaskStatus.DISPATCHED
    restored.cancel_workflow(workflow_id)
    assert len(digested) == 1


def test_interleaved_workflows_each_digest_their_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digested: list[str] = []

    def counted(source: str) -> str:
        digested.append(source)
        return source_digest(source)

    monkeypatch.setattr(models, "source_digest", counted)
    runtime = _runtime(_registry())
    workflows = [_register(runtime, _V1.replace("sources", f"w{i}")) for i in range(70)]
    for stage in (0, 1):
        for _, task_ids in workflows:
            record_dispatch(runtime, task_ids[stage])

    assert len(digested) == len(set(digested)) == 70
