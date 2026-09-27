"""A task reads upstream results only from the workflow it was submitted in."""

import asyncio
import logging
from typing import Any, cast
from unittest import mock

import pytest

from server.config import OrchestrationConfig
from server.registries.workflow import PersistedTask
from server.task.parser import parse_workflow
from server.task.runtime import TaskRuntime
from tests.server.dispatch_helpers import record_dispatch
from tests.server.dispatcher.helpers import CapturingDispatcher
from tests.server.dispatcher.test_result_availability import _worker
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import FakeRegistry, _NoopSecretVault

_SECRET = "org-a-secret"

_PRODUCER = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: producer}
spec:
  graph:
    nodes:
      - name: extract
        spec: {taskType: echo, data: {type: list, items: [x]}}
"""

_CONSUMER = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: consumer}
spec:
  graph:
    nodes:
      - name: summarize
        spec:
          taskType: echo
          data: {type: list, items: ["${extract.items.0.output}"]}
"""


def _runtime(registry: FakeRegistry, reader: Any) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, mock.Mock()),
        OrchestrationConfig(),
        reader,
        logging.getLogger("foreign-dependency"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _register(runtime: TaskRuntime, org: str, payload: str) -> str:
    _, results = asyncio.run(runtime.register("owner", org, payload, format="native"))
    return results[0].task_id


def _produce(runtime: TaskRuntime, reader: Any) -> str:
    extract = _register(runtime, "org-a", _PRODUCER)
    record_dispatch(runtime, extract, cast(Any, _worker()))
    runtime.mark_succeeded(
        extract,
        "wkr-1",
        result_payload(reader, extract, {"items": [{"output": _SECRET}]}, "org-a"),
        "t",
    )
    return extract


def _consumer_naming(extract: str) -> str:
    return _CONSUMER.replace(
        "      - name: summarize\n",
        f"      - name: summarize\n        dependsOn: [{extract}]\n",
    )


@pytest.mark.parametrize("dependency", ["foreign", "typo"])
def test_a_dependency_outside_the_workflow_is_refused_at_submit(
    dependency: str,
) -> None:
    reader = make_result_reader()
    runtime = _runtime(FakeRegistry(), reader)
    extract = _produce(runtime, reader)
    name = extract if dependency == "foreign" else "extrct"

    with pytest.raises(ValueError, match="names no node or stage of this workflow"):
        _register(runtime, "org-b", _consumer_naming(name))
    with pytest.raises(ValueError, match="names no node or stage of this workflow"):
        runtime.validate(_consumer_naming(name))


def test_a_stored_foreign_dependency_never_renders_its_result() -> None:
    registry = FakeRegistry()
    reader = make_result_reader()
    runtime = _runtime(registry, reader)
    extract = _produce(runtime, reader)
    summarize = _register(runtime, "org-b", _CONSUMER)
    # A stored record whose dependency lies outside its workflow, which submission
    # refuses.
    stored = PersistedTask.model_validate_json(registry.task_blobs[summarize])
    registry.task_blobs[summarize] = stored.model_copy(
        update={"depends_on": {extract}}
    ).model_dump_json()

    restored = _runtime(registry, reader)
    assert asyncio.run(restored.rehydrate()) == 2

    worker_registry = mock.Mock()
    worker_registry.idle_satisfying_pool.return_value = [_worker()]
    worker_registry.satisfying_workers.return_value = [_worker()]
    worker_registry.publish_task.return_value = 1
    worker_registry.get_worker.return_value = _worker()
    dispatcher = CapturingDispatcher(
        runtime=restored,
        worker_registry=worker_registry,
        logger=logging.getLogger("dispatch"),
    )
    dispatcher.dispatch_once(summarize)

    (published,) = worker_registry.publish_task.call_args_list
    assert _SECRET not in str(published)
    assert restored.upstream_task_ids(summarize) == set()


def test_a_stage_naming_a_later_stage_is_refused_as_out_of_order() -> None:
    stages = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: stages}
spec:
  stages:
    - name: b
      dependsOn: [a]
      spec: {taskType: echo}
    - name: a
      spec: {taskType: echo}
"""
    with pytest.raises(ValueError, match="names a stage declared after it"):
        parse_workflow(stages, "native")
