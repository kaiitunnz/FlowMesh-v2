"""A failed input fails the control region it feeds and everything downstream of it."""

import asyncio
from typing import Any, cast

import pytest

from server.orchestration import OrchestrationEngine, PublicationOutcome
from server.orchestration.state import ProgressAxis
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _drain,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _worker,
)
from tests.server.task.test_workflow_finalizer import _wired

_HEAD = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: region-failure}
spec:
  graph:
    nodes:
"""

_MERGE = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [a]}}
      - name: b
        spec: {taskType: echo, data: {type: list, items: [b]}}
      - name: m
        dependsOn: [a, b]
        region: {kind: merge, combination: concat}
      - name: after
        dependsOn: [m]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""

_CALL = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [a]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: c
        dependsOn: [a]
        region: {kind: call, child: kid, returns: [out]}
      - name: after
        dependsOn: [c]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


def _spawn_join(join: str) -> str:
    return f"""
      - name: a
        spec: {{taskType: echo, data: {{type: list, items: [x, y]}}}}
      - name: kid
        spec: {{taskType: echo, data: {{type: list, items: [k]}}}}
      - name: fan
        dependsOn: [a]
        region: {{kind: spawn, child: kid, result: {{visibility: published}}}}
      - name: collect
        dependsOn: [fan]
        region: {join}
      - name: after
        dependsOn: [collect]
        spec:
          taskType: echo
          data: {{type: list, items: [z]}}
          v2: {{result: {{visibility: published}}}}
"""


_JOINS = {
    "all_settled": "{kind: join, completion: all_settled}",
    "all_succeed": "{kind: join, completion: all_succeed}",
    "any": "{kind: join, completion: any, residual: cancel}",
}


def _fail(runtime: TaskRuntime, task_id: str) -> None:
    record_dispatch(runtime, task_id, cast(Any, _worker()))
    runtime.mark_failed(task_id, "wkr-1", {}, _TS, error="boom")


def _engine(runtime: TaskRuntime, workflow_id: str) -> OrchestrationEngine:
    engine = runtime.orchestration_engine(workflow_id)
    assert engine is not None
    return engine


def _assert_failed_downstream(
    runtime: TaskRuntime, ids: dict[str, str], failed: str, *names: str
) -> None:
    for name in names:
        record = runtime.get_record(ids[name])
        assert record is not None and record.status == TaskStatus.FAILED, name
        assert record.error == f"Dependency {ids[failed]} failed", name
        assert ids[name] in runtime._failed


@pytest.mark.parametrize(
    "body",
    [_MERGE, _CALL, *(_spawn_join(join) for join in _JOINS.values())],
    ids=["merge", "call", *(f"spawn_join_{name}" for name in _JOINS)],
)
def test_a_failed_input_fails_the_region_and_closes_the_workflow(body: str) -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _live_runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _HEAD + body)
        finalizer, redis, emitter = _wired(runtime, registry, workflow_id)

        _fail(runtime, ids["a"])
        _drain(runtime)
        finalizer.drain()

        downstream = ["after"] + (["kid"] if "kid" in ids else [])
        _assert_failed_downstream(runtime, ids, "a", *downstream)
        assert runtime.workflow_settlement(workflow_id).settled
        assert registry.remaining_of(workflow_id) == set()
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys
        assert emitter.emitted == [workflow_id]

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_failed_producer_is_not_an_empty_spawn() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _spawn_join(_JOINS["any"]))
    _fail(runtime, ids["a"])
    engine = _engine(runtime, workflow_id)
    snapshot = engine.to_snapshot()

    # The spawn never fired: no scope, no child-init capability, no seal, no child, and
    # no join aggregate -- nothing an empty spawn would record.
    assert engine.scope_for("fan") is None
    assert not any(
        c.axis is ProgressAxis.CHILD_INIT for c in snapshot.progress_capabilities
    )
    assert not snapshot.region_aggregates
    kinds = {kind for kind, _ in engine.contract_trace()}
    assert "child_init_sealed" not in kinds and "join_released" not in kinds
    assert sorted(snapshot.failed_regions) == ["collect", "fan"]
    assert engine.region_closed("fan") and engine.region_closed("collect")
    assert not engine.spawn_awaits_children("fan")

    publications = {p.output_id: p for p in snapshot.result_publications}
    collection = publications["collection:fan"]
    assert collection.outcome is PublicationOutcome.DECLARED_FAILURE
    slot = engine.output_slot("collection:fan")
    assert slot is not None and slot.scope_id is None and slot.logical_key is None
    assert all(
        p.outcome is PublicationOutcome.DECLARED_FAILURE
        for p in snapshot.result_publications
    )


@pytest.mark.anyio
async def test_a_late_input_never_fires_a_failed_merge() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _MERGE)
    ready = _pop_ready(runtime)
    assert set(ready) == {ids["a"], ids["b"]}
    _fail(runtime, ids["a"])
    record_dispatch(runtime, ids["b"], cast(Any, _worker()))
    runtime.mark_succeeded(ids["b"], "wkr-1", {}, _TS)

    engine = _engine(runtime, workflow_id)
    assert "merge_combined" not in {kind for kind, _ in engine.contract_trace()}
    assert not [r for r in engine.to_snapshot().records if r.operator_id == "m"]
    assert _pop_ready(runtime) == []
    _assert_failed_downstream(runtime, ids, "a", "after")


_EXTRA_INPUT_JOIN = """
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: x
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [planner]
        region: {kind: spawn, child: kid, result: {visibility: published}}
      - name: collect
        dependsOn: [fan, x]
        region: {kind: join, completion: all_settled}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


@pytest.mark.anyio
async def test_a_join_failed_by_another_input_keeps_its_children_members() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _EXTRA_INPUT_JOIN)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS
    )
    children = [t for t in _pop_ready(runtime) if t != ids["x"]]
    assert len(children) == 2

    _fail(runtime, ids["x"])
    for child in children:
        record_dispatch(runtime, child, cast(Any, _worker()))
        runtime.mark_succeeded(child, "wkr-1", {}, _TS)

    engine = _engine(runtime, workflow_id)
    snapshot = engine.to_snapshot()
    _assert_failed_downstream(runtime, ids, "x", "after")
    # The spawn itself ran, so its members stand and no collection-level member exists.
    members = [
        p for p in snapshot.result_publications if p.output_id == "collection:fan"
    ]
    assert len(members) == 2
    assert all(p.outcome is PublicationOutcome.SUCCESS for p in members)
    assert engine.output_slot("collection:fan") is None
    assert "join_released" not in {kind for kind, _ in engine.contract_trace()}
    assert runtime.workflow_settlement(workflow_id).settled


_TWO_SPAWNS = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [a]}}
      - name: b
        spec: {taskType: echo, data: {type: list, items: [b]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan_a
        dependsOn: [a]
        region: {kind: spawn, child: kid}
      - name: fan_b
        dependsOn: [b]
        region: {kind: spawn, child: kid}
"""


@pytest.mark.anyio
async def test_a_shared_child_template_fails_only_with_every_spawn() -> None:
    runtime = _live_runtime(FakeRegistry())
    _, ids = await _register(runtime, _HEAD + _TWO_SPAWNS)
    _fail(runtime, ids["a"])
    kid = runtime.get_record(ids["kid"])
    assert kid is not None and kid.status == TaskStatus.PENDING

    _fail(runtime, ids["b"])
    kid = runtime.get_record(ids["kid"])
    assert kid is not None and kid.status == TaskStatus.FAILED


@pytest.mark.anyio
async def test_a_succeeded_spawn_retires_its_template() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(runtime, _HEAD + _spawn_join(_JOINS["any"]))
    a = ids["a"]
    record_dispatch(runtime, a, cast(Any, _worker()))
    runtime.mark_succeeded(a, "wkr-1", _planned(runtime, a, ["h1"]), _TS)
    _drain(runtime)

    kid = runtime.get_record(ids["kid"])
    assert kid is not None and kid.status == TaskStatus.PENDING
    assert ids["kid"] not in registry.remaining_of(workflow_id)
    engine = _engine(runtime, workflow_id)
    assert "region_failed" not in {kind for kind, _ in engine.contract_trace()}
    assert engine.output_slot("collection:fan") is None
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_failed_region_stays_failed_and_closed_across_a_restart() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(
        runtime, _HEAD + _spawn_join(_JOINS["all_settled"])
    )
    _fail(runtime, ids["a"])

    restored = _live_runtime(registry, "restored", reader=runtime._results)
    assert await restored.rehydrate() == 1
    engine = _engine(restored, workflow_id)
    assert sorted(engine.to_snapshot().failed_regions) == ["collect", "fan"]
    assert _pop_ready(restored) == []
    _assert_failed_downstream(restored, ids, "a", "kid", "after")
    assert restored.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_crash_before_the_ledger_save_converges_on_restart() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(
        runtime, _HEAD + _spawn_join(_JOINS["all_settled"])
    )
    stale = registry.ledger_blobs[workflow_id]
    _fail(runtime, ids["a"])
    # The task records committed; the ledger save that follows them did not.
    registry.ledger_blobs[workflow_id] = stale

    restored = _live_runtime(registry, "restored", reader=runtime._results)
    await restored.rehydrate()
    engine = _engine(restored, workflow_id)
    assert sorted(engine.to_snapshot().failed_regions) == ["collect", "fan"]
    publication = engine.output_publication("collection:fan")
    assert publication is not None
    assert publication.outcome is PublicationOutcome.DECLARED_FAILURE
    _assert_failed_downstream(restored, ids, "a", "kid", "after")
    assert restored.workflow_settlement(workflow_id).settled
