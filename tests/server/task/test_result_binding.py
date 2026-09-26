"""A task's success binds its stored result once, and its result reads back from it."""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration import PublicationOutcome
from server.task.models import EventEffect, TaskStatus
from server.task.runtime import TaskRuntime
from shared.schemas.result import BaseExecutorResult
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import (
    _TS,
    AUTORESEARCH,
    LINEAR,
    FakeRegistry,
    _NoopSecretVault,
    _planned,
    _pop_ready,
    _register,
    _worker,
    _WorkerRegistryStub,
)

V1_LINEAR = """
apiVersion: flowmesh/v1
kind: Workflow
metadata: {name: linear-v1}
spec:
  graph:
    nodes:
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


def _runtime(registry: FakeRegistry, reader: Any = None) -> TaskRuntime:
    return TaskRuntime(
        cast(Any, registry),
        cast(Any, _WorkerRegistryStub()),
        OrchestrationConfig(),
        reader or make_result_reader(),
        logging.getLogger("result-binding"),
        secret_vault=cast(Any, _NoopSecretVault()),
    )


def _stored(runtime: TaskRuntime, task_id: str, value: str) -> dict[str, Any]:
    scope = runtime._tasks[task_id].org_id
    return result_payload(runtime._results, task_id, {"value": value}, scope)


def _value(runtime: TaskRuntime, task_id: str) -> Any:
    envelope = runtime.read_result(task_id)
    return None if envelope is None else envelope.result.model_dump().get("value")


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [LINEAR, V1_LINEAR])
async def test_a_retried_success_never_re_points_the_bound_result(
    payload: str,
) -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _register(runtime, payload)
    a = ids["a"]
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", _stored(runtime, a, "first"), _TS)
    # A duplicate or speculative success converges on the result already bound.
    duplicate = runtime.mark_succeeded(a, "wkr-1", _stored(runtime, a, "second"), _TS)

    assert duplicate.effect is EventEffect.SETTLED

    assert _value(runtime, a) == "first"


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [LINEAR, V1_LINEAR])
async def test_a_result_outside_the_task_scope_is_not_bound(payload: str) -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _register(runtime, payload)
    a = ids["a"]
    foreign = result_payload(runtime._results, a, {"value": "x"}, "another-org")
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", foreign, _TS)

    assert runtime._tasks[a].status == TaskStatus.DONE
    assert runtime.result_binding(a) is None


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [LINEAR, V1_LINEAR])
async def test_a_bound_result_survives_a_restart(payload: str) -> None:
    registry = FakeRegistry()
    reader = make_result_reader()
    runtime = _runtime(registry, reader)
    _, ids = await _register(runtime, payload)
    a = ids["a"]
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", _stored(runtime, a, "kept"), _TS)

    restored = _runtime(registry, make_result_reader(reader.store))
    assert await restored.rehydrate() == 1
    assert _value(restored, a) == "kept"


@pytest.mark.anyio
async def test_a_spawned_child_reads_as_the_value_it_settled_with() -> None:
    registry = FakeRegistry()
    reader = make_result_reader()
    runtime = _runtime(registry, reader)
    _, ids = await _register(runtime, AUTORESEARCH)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", _planned(runtime, planner, ["h1"]), _TS)
    (child,) = _pop_ready(runtime)
    record_dispatch(runtime, child, cast(Any, _worker()))
    runtime.mark_succeeded(child, "wkr-1", _stored(runtime, child, "child"), _TS)

    assert _value(runtime, child) == "child"
    restored = _runtime(registry, make_result_reader(reader.store))
    assert await restored.rehydrate() == 1
    assert _value(restored, child) == "child"


@pytest.mark.anyio
async def test_a_replayed_skip_republishes_as_explicit_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeRegistry()
    runtime = _runtime(registry)
    workflow_id, ids = await _register(runtime, LINEAR)
    a = ids["a"]
    skip = {"skipped": True, "reason": "condition_not_met"}
    # A crash after the task's terminal persist and before the ledger save leaves the
    # skip only on the record; the restart replays it from there.
    monkeypatch.setattr(runtime, "_save_ledger_locked", lambda _workflow_id: None)
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, None, {}, _TS, skip=skip)

    restored = _runtime(registry)
    assert await restored.rehydrate() == 1
    pub = restored.resolve_v2_legacy_result(workflow_id, a)
    assert pub is not None and pub.outcome is PublicationOutcome.EXPLICIT_EMPTY
    envelope = restored.read_result(a)
    assert envelope is not None and envelope.metadata == skip
    assert envelope.result == BaseExecutorResult()


@pytest.mark.anyio
async def test_a_merged_child_binds_only_its_own_result() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _register(runtime, V1_LINEAR)
    parent, own, unreported = ids["a"], ids["b"], "tsk-unreported"
    record = runtime._tasks[own].model_copy(update={"task_id": unreported})
    runtime._tasks[unreported] = record
    runtime._merge_children_map[parent] = [own, unreported]
    payload = _stored(runtime, parent, "parent")
    payload["child_result_references"] = {
        own: _stored(runtime, own, "own")["result_reference"]
    }
    record_dispatch(runtime, parent, cast(Any, _worker()))
    runtime.mark_succeeded(parent, "wkr-1", payload, _TS)

    assert _value(runtime, own) == "own"
    assert runtime._tasks[unreported].status != TaskStatus.DONE
    assert runtime.result_binding(unreported) is None


@pytest.mark.anyio
async def test_a_success_with_nothing_bound_is_unreadable_to_its_consumers() -> None:
    runtime = _runtime(FakeRegistry())
    _, ids = await _register(runtime, V1_LINEAR)
    a = ids["a"]
    # Unsettled: a consumer defers rather than failing.
    assert not runtime._settled_unbound_locked(a)
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", {}, _TS)

    assert runtime.result_binding(a) is None
    assert runtime._settled_unbound_locked(a)
