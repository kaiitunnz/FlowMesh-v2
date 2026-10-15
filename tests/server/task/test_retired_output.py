"""A task spec carries no output block: a submission naming one is refused, and a task
stored with one, a destination included, still loads and runs."""

import asyncio
import json
from typing import Any

import pytest

from server.clients.redis import task_state_key
from server.task.parser import parse_workflow
from shared.schemas.artifact import ArtifactContext
from tests.server.dispatch_helpers import record_dispatch
from tests.server.redis_helpers import raw_control
from tests.server.result_store import result_payload
from tests.server.task.test_settled_history import _TS, _registry, _runtime
from tests.support.waiting import pop_ready

_V1 = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: retired}
spec:
  graph:
    nodes:
      - {name: a, spec: {taskType: echo}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo}
"""

_V2 = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: retired}
spec:
  graph:
    nodes:
      - name: a
        spec:
          taskType: echo
          data: {type: list, items: [x]}
      - name: b
        dependsOn: [a]
        spec:
          taskType: echo
          data: {type: list, items: [y]}
"""

_DESTINATION = {"type": "http", "headers": {"X-Api-Key": "msk-stored"}}
_POINTER = "/output/destination/headers/X-Api-Key"


@pytest.mark.parametrize("text", [_V1, _V2])
@pytest.mark.parametrize(
    "output",
    ["{artifacts: [results.json]}", "{destination: {type: http}}"],
)
def test_a_submission_naming_an_output_block_is_refused(text: str, output: str) -> None:
    named = text.replace("taskType: echo", f"taskType: echo, output: {output}", 1)
    if named == text:
        named = text.replace(
            "          taskType: echo\n",
            f"          taskType: echo\n          output: {output}\n",
            1,
        )
    runtime = _runtime(_registry())

    with pytest.raises(Exception, match="output"):
        if "flowmesh/v1" in named:
            parse_workflow(named, "native")
        else:
            asyncio.run(runtime.register("owner", "org", named, format="native"))


def _name_a_destination(registry: Any, task_id: str) -> None:
    """Rewrite a stored task as one stored before destinations were retired: its spec
    names one, with a vaulted credential pointing into it."""
    raw = raw_control(registry)
    stored = json.loads(raw.get(task_state_key(task_id)))
    stored["record"]["task"]["spec"]["output"] = {
        "artifacts": ["results.json"],
        "destination": _DESTINATION,
    }
    stored["record"]["credential_refs"] = {_POINTER: "msk-stored"}
    raw.set(task_state_key(task_id), json.dumps(stored))


@pytest.mark.parametrize("text", [_V1, _V2])
def test_a_restart_loads_and_runs_tasks_stored_with_an_output_block(
    text: str,
) -> None:
    registry = _registry()
    runtime = _runtime(registry)
    asyncio.run(runtime.register("owner", "org", text, format="native"))
    first = pop_ready(runtime)
    assert first is not None
    record_dispatch(runtime, first, "wkr-1", "dsp-1")
    runtime.mark_started(first, "wkr-1", {}, _TS)
    old_envelope = {
        "ok": True,
        "_artifacts": {"base_dir": f"/results/{first}", "base_url": "http://h"},
    }
    payload = result_payload(runtime._results, first, old_envelope, "org")
    runtime.mark_succeeded(first, "wkr-1", payload, _TS, "dsp-1")
    second = pop_ready(runtime)
    assert second is not None
    for task_id in (first, second):
        _name_a_destination(registry, task_id)

    restored = _runtime(registry)
    restored._results = runtime._results
    assert asyncio.run(restored.rehydrate()) == 1

    for task_id in (first, second):
        record = restored.get_record(task_id)
        assert record is not None
        assert "output" not in record.task.spec.model_dump()
        assert not record.credential_refs
    envelope = restored.read_result(first)
    assert envelope is not None and envelope.result.artifacts_ == ArtifactContext()
    record = restored.get_record(second)
    assert record is not None
    assert restored.credentialed_task(record) is not None
