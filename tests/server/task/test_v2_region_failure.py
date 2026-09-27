"""A failed input fails the control region it feeds and everything downstream of it."""

import asyncio
import json
from typing import Any, cast

import pytest

from server.orchestration import OrchestrationEngine, PublicationOutcome
from server.orchestration.state import BoundaryEvent, ProgressAxis
from server.task.models import TaskStatus
from server.task.runtime import TaskRuntime
from server.task.v2.representations.operators import BoundaryEventKind
from shared.harness.adapter import HarnessResult, HarnessResultKind
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


async def _pre_upgrade_hang(
    registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> tuple[TaskRuntime, str, dict[str, str]]:
    """A workflow stored by a release whose failures stopped at a control region."""
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(
        runtime, _HEAD + _spawn_join(_JOINS["all_settled"])
    )
    engine = _engine(runtime, workflow_id)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_fail_region", lambda *_args: None)
        _fail(runtime, ids["a"])
    blob = json.loads(registry.ledger_blobs[workflow_id])
    del blob["failed_regions"]
    registry.ledger_blobs[workflow_id] = json.dumps(blob)
    assert not runtime.workflow_settlement(workflow_id).settled
    return runtime, workflow_id, ids


def test_a_restart_closes_a_workflow_stored_hung_behind_a_failed_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        registry = FakeRegistry()
        registry.submitted_at = _TS
        runtime, workflow_id, ids = await _pre_upgrade_hang(registry, monkeypatch)

        restored = _live_runtime(registry, "restored", reader=runtime._results)
        finalizer, redis, emitter = _wired(restored, registry, workflow_id)
        await restored.rehydrate()
        finalizer.drain()

        _assert_failed_downstream(restored, ids, "a", "kid", "after")
        persisted = registry.load_task_states(ids["after"])[0]
        assert persisted is not None and persisted.record.status == TaskStatus.FAILED
        assert registry.remaining_of(workflow_id) == set()
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys
        assert emitter.emitted == [workflow_id]
        publication = _engine(restored, workflow_id).output_publication(
            "collection:fan"
        )
        assert publication is not None
        assert publication.outcome is PublicationOutcome.DECLARED_FAILURE

    asyncio.run(run())


@pytest.mark.anyio
async def test_a_restart_leaves_a_cancelled_hung_workflow_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeRegistry()
    runtime, workflow_id, ids = await _pre_upgrade_hang(registry, monkeypatch)
    runtime.cancel_workflow(workflow_id)

    restored = _live_runtime(registry, "restored", reader=runtime._results)
    await restored.rehydrate()
    for name in ("kid", "after"):
        record = restored.get_record(ids[name])
        assert record is not None and record.status == TaskStatus.CANCELLED
    assert _engine(restored, workflow_id).to_snapshot().failed_regions == []


_SSH_PRODUCER = _HEAD + """
      - name: a
        spec:
          taskType: ssh
          interactive: false
          image: alpine:3
          command: ["sh", "-c", "exit 3"]
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [a]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


@pytest.mark.anyio
async def test_an_ambiguous_producer_fails_its_region_downstream_as_dependents() -> (
    None
):
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _SSH_PRODUCER)
    record_dispatch(runtime, ids["a"], cast(Any, _worker()))

    runtime.mark_v2_uncertain(ids["a"])

    record = runtime.get_record(ids["a"])
    assert record is not None and record.error == "ambiguity-terminal effect"
    _assert_failed_downstream(runtime, ids, "a", "kid", "after")
    assert runtime.workflow_settlement(workflow_id).settled


def _last_child_fails(completion: str) -> str:
    return f"""
      - name: planner
        spec: {{taskType: echo, data: {{type: list, items: [seed]}}}}
      - name: kid
        spec: {{taskType: echo, data: {{type: list, items: [k]}}}}
      - name: fan
        dependsOn: [planner]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: {completion}}}
      - name: after
        dependsOn: [collect]
        spec: {{taskType: echo, data: {{type: list, items: [z]}}}}
"""


async def _fail_the_last_child(
    completion: str,
) -> tuple[TaskRuntime, str, dict[str, str], str]:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _last_child_fails(completion))
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS
    )
    first, last = _pop_ready(runtime)
    record_dispatch(runtime, first, cast(Any, _worker()))
    runtime.mark_succeeded(first, "wkr-1", {}, _TS)
    _fail(runtime, last)
    return runtime, workflow_id, ids, last


@pytest.mark.anyio
async def test_a_join_released_by_a_failed_last_child_readies_its_downstream() -> None:
    runtime, workflow_id, ids, _ = await _fail_the_last_child("all_settled")

    assert _pop_ready(runtime) == [ids["after"]]
    record_dispatch(runtime, ids["after"], cast(Any, _worker()))
    runtime.mark_succeeded(ids["after"], "wkr-1", {}, _TS)
    assert runtime.workflow_settlement(workflow_id).settled


_NESTED = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: worker
        spec:
          taskType: agent
          task: research
          v2:
            inputs: [facet]
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            boundary: [spawn, spawn_seal, yield]
            child: [{name: sub, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: sub
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [a]
        region: {kind: spawn, child: worker}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


def test_a_failed_spawn_fails_the_templates_nested_under_its_child() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        registry.submitted_at = _TS
        runtime = _live_runtime(registry)
        workflow_id, ids = await _register(runtime, _HEAD + _NESTED)
        finalizer, redis, _ = _wired(runtime, registry, workflow_id)

        _fail(runtime, ids["a"])
        _drain(runtime)
        finalizer.drain()

        _assert_failed_downstream(runtime, ids, "a", "worker", "sub", "after")
        assert registry.remaining_of(workflow_id) == set()
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys

    asyncio.run(run())


def test_an_empty_spawn_retires_the_templates_nested_under_its_child() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        registry.submitted_at = _TS
        runtime = _live_runtime(registry)
        workflow_id, ids = await _register(runtime, _HEAD + _NESTED)
        finalizer, redis, _ = _wired(runtime, registry, workflow_id)
        a = ids["a"]
        record_dispatch(runtime, a, cast(Any, _worker()))
        runtime.mark_succeeded(a, "wkr-1", _planned(runtime, a, []), _TS)
        _drain(runtime)
        finalizer.drain()

        assert registry.remaining_of(workflow_id) == set()
        assert runtime.workflow_settlement(workflow_id).settled
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys

    asyncio.run(run())


_AGENT_NESTED = """
      - name: lead
        spec:
          taskType: agent
          task: spawn reviewers
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            boundary: [spawn, spawn_seal, yield]
            child: [{name: reviewer, authority: {invoke: [model], delegate: [model]}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec:
          taskType: agent
          task: review
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            boundary: [spawn, spawn_seal, yield]
            child: [{name: deep, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: deep
        spec: {taskType: echo, data: {type: list, items: [k]}}
"""


@pytest.mark.anyio
async def test_a_sealed_agent_region_retires_the_templates_nested_under_it() -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(runtime, _HEAD + _AGENT_NESTED)
    lead = ids["lead"]
    assert _pop_ready(runtime) == [lead]
    record_dispatch(runtime, lead, cast(Any, _worker()))
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION)
    runtime.mark_succeeded(
        lead, "wkr-1", {"agent_episode": completion.model_dump(mode="json")}, _TS
    )

    assert registry.remaining_of(workflow_id) == set()
    assert runtime.workflow_settlement(workflow_id).settled


_AGENT_REGION = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: lead
        dependsOn: [a]
        spec:
          taskType: agent
          task: spawn reviewers
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            boundary: [spawn, spawn_seal, yield]
            child: [{name: reviewer, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: reviewer
        spec:
          taskType: agent
          task: research the facet
          v2:
            inputs: [facet]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: merge
        spec:
          taskType: agent
          task: merge the reviews
          v2:
            inputs: [{name: reviews, from: lead, region: reviewer}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _agent_failure(runtime: TaskRuntime, lead: str) -> None:
    record_dispatch(runtime, lead, cast(Any, _worker()))
    failure = HarnessResult(kind=HarnessResultKind.FAILURE, error="agent blew up")
    runtime.mark_succeeded(
        lead, "wkr-1", {"agent_episode": failure.model_dump(mode="json")}, _TS
    )


@pytest.mark.parametrize("path", ["reported", "episode", "cascade"])
def test_a_failed_agent_fails_its_region_and_closes_the_workflow(path: str) -> None:
    async def run() -> None:
        registry = FakeRegistry()
        registry.submitted_at = _TS
        runtime = _live_runtime(registry)
        workflow_id, ids = await _register(runtime, _HEAD + _AGENT_REGION)
        finalizer, redis, _ = _wired(runtime, registry, workflow_id)
        a, lead = ids["a"], ids["lead"]
        if path == "cascade":
            _fail(runtime, a)
        else:
            record_dispatch(runtime, a, cast(Any, _worker()))
            runtime.mark_succeeded(a, "wkr-1", {}, _TS)
            assert _pop_ready(runtime) == [lead]
            if path == "reported":
                _fail(runtime, lead)
            else:
                _agent_failure(runtime, lead)
        _drain(runtime)
        finalizer.drain()

        primary = "a" if path == "cascade" else "lead"
        downstream = ["reviewer", "merge"] + (["lead"] if path == "cascade" else [])
        _assert_failed_downstream(runtime, ids, primary, *downstream)
        assert _pop_ready(runtime) == []
        engine = _engine(runtime, workflow_id)
        kinds = {kind for kind, _ in engine.contract_trace()}
        assert "child_init_sealed" not in kinds and "join_released" not in kinds
        assert registry.remaining_of(workflow_id) == set()
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys

    asyncio.run(run())


_SOLO_AGENT = """
      - name: solo
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: after
        dependsOn: [solo]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


@pytest.mark.anyio
async def test_a_survived_denial_never_names_a_later_ambiguity_failure() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _SOLO_AGENT)
    solo = ids["solo"]
    _pop_ready(runtime)
    record_dispatch(runtime, solo, cast(Any, _worker()))
    engine = _engine(runtime, workflow_id)
    # A boundary outside the declared face is denied, and the agent carries on past it.
    runtime.apply_boundary_event(
        solo,
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="search"
        ),
    )
    engine.mark_pending_outcome(solo, "c0")
    engine.deliver_boundary_outcome(solo, "c0")
    with runtime._cv:
        runtime._reenqueue_episode_locked(solo)
    _pop_ready(runtime)
    record_dispatch(runtime, solo, cast(Any, _worker()))
    runtime.apply_boundary_event(
        solo,
        BoundaryEvent(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c1",
            interface="model",
            request_digest="sha256:abc",
        ),
    )

    runtime.mark_v2_uncertain(solo)

    record = runtime.get_record(solo)
    assert record is not None and record.error == "ambiguity-terminal effect"
    _assert_failed_downstream(runtime, ids, "solo", "after")


@pytest.mark.anyio
async def test_a_failed_child_fails_an_all_succeed_join_downstream() -> None:
    runtime, workflow_id, ids, last = await _fail_the_last_child("all_succeed")

    assert _pop_ready(runtime) == []
    _assert_failed_downstream_of(runtime, ids, last, "after")
    engine = _engine(runtime, workflow_id)
    assert "collect" in engine.to_snapshot().failed_regions
    assert "join_released" not in {kind for kind, _ in engine.contract_trace()}
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_the_first_failed_child_names_an_all_succeed_join_failure() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(
        runtime, _HEAD + _last_child_fails("all_succeed")
    )
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS
    )
    first, last = _pop_ready(runtime)
    _fail(runtime, first)
    record_dispatch(runtime, last, cast(Any, _worker()))
    runtime.mark_succeeded(last, "wkr-1", {}, _TS)

    _assert_failed_downstream_of(runtime, ids, first, "after")
    assert runtime.workflow_settlement(workflow_id).settled


_NO_WINNER = """
      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [planner]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: any, residual: cancel, no_winner_failure: true}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
"""


@pytest.mark.anyio
async def test_a_join_with_no_winner_fails_its_downstream() -> None:
    runtime = _live_runtime(FakeRegistry())
    workflow_id, ids = await _register(runtime, _HEAD + _NO_WINNER)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(planner, "wkr-1", _planned(runtime, planner, []), _TS)
    _drain(runtime)

    assert _pop_ready(runtime) == []
    record = runtime.get_record(ids["after"])
    assert record is not None and record.status == TaskStatus.FAILED
    assert record.error == "join collect resolved no winner"
    assert runtime.workflow_settlement(workflow_id).settled


@pytest.mark.anyio
async def test_a_failed_join_stays_failed_across_a_crash_before_its_ledger_save() -> (
    None
):
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(
        runtime, _HEAD + _last_child_fails("all_succeed")
    )
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["h1", "h2"]), _TS
    )
    first, last = _pop_ready(runtime)
    record_dispatch(runtime, first, cast(Any, _worker()))
    runtime.mark_succeeded(first, "wkr-1", {}, _TS)
    save = registry.save_ledger_snapshot

    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("root crashed")

    record_dispatch(runtime, last, cast(Any, _worker()))
    registry.save_ledger_snapshot = crash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        runtime.mark_failed(last, "wkr-1", {}, _TS, error="boom")
    registry.save_ledger_snapshot = save  # type: ignore[method-assign]

    restored = _live_runtime(registry, "restored", reader=runtime._results)
    assert await restored.rehydrate() == 1
    _assert_failed_downstream_of(restored, ids, last, "after")
    assert restored.workflow_settlement(workflow_id).settled


def _assert_failed_downstream_of(
    runtime: TaskRuntime, ids: dict[str, str], failed: str, *names: str
) -> None:
    for name in names:
        record = runtime.get_record(ids[name])
        assert record is not None and record.status == TaskStatus.FAILED, name
        assert record.error == f"Dependency {failed} failed", name
