"""Region work across a crash at each durable write, and the workflow status every
write leaves behind."""

import copy
from typing import Any, cast

import pytest

from server.orchestration import LedgerSnapshot, WorkItemStatus
from server.orchestration.state import BoundaryEvent
from server.registries.workflow import WorkflowRecord, WorkflowRegistry
from server.task.models import TaskStatus
from server.task.runtime import control_reads
from server.task.runtime import facade as runtime_facade
from server.task.v2 import PersistedV2Workflow
from shared.harness.boundary import BoundaryEventKind
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import result_payload
from tests.server.task.test_runtime_control_flow import (
    _DIAMOND,
    _ECHO,
    _LOOP,
    _LOOP_NODES,
    _Run,
    _workflow,
)
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _live_runtime,
    _planned,
    _pop_ready,
    _register,
    _worker,
)

_DURABLE = ("task_blobs", "ledger_blobs", "remaining", "dynamic_task_ids", "control")


def _succeed(
    run: _Run, task_id: str, result: dict[str, Any], *, dispatched: bool = False
) -> None:
    """Settle a task's success without running the re-drives it leaves due."""
    record = run.runtime.get_record(task_id)
    assert record is not None
    payload = result_payload(run.reader, task_id, {"ok": True, **result}, record.org_id)
    if not dispatched:
        record_dispatch(run.runtime, task_id, cast(Any, _worker()))
    run.runtime.mark_succeeded(task_id, "wkr-1", payload, _TS)


def _capture(registry: FakeRegistry) -> dict[str, Any]:
    return {name: copy.deepcopy(getattr(registry, name)) for name in _DURABLE}


def _crash_to(registry: FakeRegistry, durable: dict[str, Any]) -> None:
    for name, value in durable.items():
        setattr(registry, name, value)


_WRITES = ("commit_transition", "commit_dynamic_tasks", "save_ledger_snapshot")


def _writes(
    registry: FakeRegistry, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """The durable state after each write the registry takes."""
    states: list[dict[str, Any]] = []
    for method in _WRITES:
        real = getattr(registry, method)

        def write(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            out = _real(*args, **kwargs)
            states.append(_capture(registry))
            return out

        monkeypatch.setattr(registry, method, write)
    return states


@pytest.mark.anyio
async def test_a_skip_whose_ledger_was_lost_never_replays_as_a_success() -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    (classify,) = run.ready
    run.ready.clear()
    _succeed(run, classify, {"label": "yes"})
    before = run.registry.ledger_blobs[run.workflow_id]
    run.drive()
    right = run.ids["right_work"]
    stored = run.registry.load_task_states(right)[0]
    assert stored is not None and stored.record.result_skip is not None
    # The skipped record landed; the ledger save holding the decision did not.
    run.registry.ledger_blobs[run.workflow_id] = before

    restored = await run.restart()
    wi = restored.engine.work_item(right)
    assert wi is not None and wi.status is WorkItemStatus.SKIPPED
    restored.run("left_work")
    restored.run("after")
    assert restored.settled()
    assert restored.engine.control_failure() is None


_SPAWN_AND_LOOP = f"""
      - name: seed
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [seed]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
"""


@pytest.mark.anyio
async def test_work_a_transition_readies_survives_a_crash_after_each_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(_SPAWN_AND_LOOP, _LOOP))
    (seed,) = run.ready
    run.ready.clear()
    record_dispatch(run.runtime, seed, cast(Any, _worker()))
    registry = run.registry
    writes = _writes(registry, monkeypatch)
    _succeed(run, seed, {"items": ["a", "b"]}, dispatched=True)
    run.drive()
    assert sorted(run.name(t) for t in run.ready) == ["kid", "kid", "step"]
    monkeypatch.undo()

    for durable in writes:
        _crash_to(registry, durable)
        restored = await run.restart()
        ready = sorted(restored.name(t) for t in restored.ready)
        assert ready == ["kid", "kid", "step"], ready


_GATED_FAN = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: input, field: [label]}}
      - name: left_work
        dependsOn: [{{node: decide, port: left}}]
        spec: {_ECHO}
      - name: right_work
        dependsOn: [{{node: decide, port: right}}]
        spec: {_ECHO}
      - name: p2
        spec: {_ECHO}
      - name: gate
        dependsOn: [{{node: p2, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: go}}, {{name: stop}}]
          selection: {{input: input, field: [label]}}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [{{node: gate, port: go}}]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""


@pytest.mark.anyio
async def test_a_skip_and_a_fan_out_in_one_redrive_survive_a_crash_after_each_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(_GATED_FAN))
    by_name = {run.name(t): t for t in run.ready}
    run.ready.clear()
    real_read = control_reads.read_control_value
    reads: list[str] = []

    def read(results: Any, value: Any, binding: Any) -> Any:
        reads.append(str(value.legacy_task_id))
        # The fan-out's read of p2 finds the store away once, so it shares the next
        # re-drive with decide's read.
        if value.legacy_task_id == by_name["p2"] and reads.count(by_name["p2"]) == 2:
            return control_reads.ControlRead(unavailable=True)
        return real_read(results, value, binding)

    monkeypatch.setattr(runtime_facade.control_reads, "read_control_value", read)
    _succeed(run, by_name["p2"], {"label": "go", "items": ["a", "b"]})
    run.drive()
    _succeed(run, by_name["classify"], {"label": "left"})
    registry = run.registry
    writes = _writes(registry, monkeypatch)
    run.schedulers[0].settle(run.workflow_id)
    run.runtime._redrive_workflow(run.workflow_id)
    monkeypatch.undo()
    right = run.ids["right_work"]

    for durable in writes:
        _crash_to(registry, durable)
        restored = await run.restart()
        record = restored.runtime.get_record(right)
        wi = restored.engine.work_item(right)
        assert record is not None and wi is not None
        assert not (
            record.status == TaskStatus.PENDING and wi.status is WorkItemStatus.SKIPPED
        )
        ledger = LedgerSnapshot.model_validate_json(
            durable["ledger_blobs"][run.workflow_id]
        )
        for item in ledger.work_items:
            if item.legacy_task_id and item.status is not WorkItemStatus.BLOCKED:
                assert item.legacy_task_id in durable["task_blobs"], item


def _durable_status(registry: FakeRegistry, workflow_id: str) -> str:
    """The workflow status the registry derives from what is durable now."""
    control = registry.control.get(workflow_id)
    ids = [
        *registry.workflow_task_ids[workflow_id],
        *registry.dynamic_task_ids.get(workflow_id, ()),
    ]
    records = [p.record for t in ids if (p := registry.load_task_states(t)[0])]
    record = WorkflowRecord(
        workflow_id=workflow_id,
        task_ids=list(registry.workflow_task_ids[workflow_id]),
        control_open=bool(control and control.open),
        control_failure=(control.failure or "") if control else "",
        control_cancelled=bool(control and control.cancelled),
    )
    built = WorkflowRegistry._build_workflow(
        WorkflowRegistry.__new__(WorkflowRegistry),
        record,
        set(),
        {r.task_id for r in records if r.status == TaskStatus.FAILED},
        {r.task_id for r in records if r.status == TaskStatus.CANCELLED},
        registry.remaining_of(workflow_id),
    )
    return str(built.status.value).lower()


def _statuses(
    registry: FakeRegistry, workflow_id: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """The durable workflow status after every write."""
    seen: list[str] = []
    for method in _WRITES:
        real = getattr(registry, method)

        def write(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            out = _real(*args, **kwargs)
            seen.append(_durable_status(registry, workflow_id))
            return out

        monkeypatch.setattr(registry, method, write)
    return seen


@pytest.mark.anyio
async def test_a_loop_never_reads_done_between_its_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(_LOOP_NODES, _LOOP))
    seen = _statuses(run.registry, run.workflow_id, monkeypatch)
    run.run("seed")
    run.run("step", {"route": "again"})
    run.run("step", {"route": "done"})
    assert run.settled()
    assert "done" not in seen[:-1], seen
    assert seen[-1] == "done"


@pytest.mark.anyio
async def test_a_spawn_producer_success_never_reads_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FakeRegistry()
    runtime = _live_runtime(registry)
    workflow_id, ids = await _register(
        runtime,
        _workflow(f"""
      - name: planner
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [planner]
        region: {{kind: spawn, child: kid, result: {{visibility: published}}}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""),
    )
    seen = _statuses(registry, workflow_id, monkeypatch)
    planner = ids["planner"]
    record_dispatch(runtime, planner, cast(Any, _worker()))
    runtime.mark_succeeded(
        planner, "wkr-1", _planned(runtime, planner, ["a", "b"]), _TS
    )
    assert len(_pop_ready(runtime)) == 2
    assert "done" not in seen, seen


@pytest.mark.anyio
async def test_a_cancel_while_only_a_read_holds_the_workflow_reads_cancelled() -> None:
    run = await _Run().start(_workflow(_LOOP_NODES, _LOOP))
    run.run("seed")
    (step,) = run.ready
    run.ready.clear()
    _succeed(run, step, {"route": "again"})
    assert _durable_status(run.registry, run.workflow_id) == "pending"
    run.runtime.cancel_workflow(run.workflow_id)
    assert _durable_status(run.registry, run.workflow_id) == "cancelled"


_RECURSIVE = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: recursive}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2: {child: writer}
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


@pytest.mark.anyio
async def test_a_recursive_agent_stored_before_blueprints_makes_its_children() -> None:
    run = await _Run().start(_RECURSIVE)
    (writer,) = run.ready
    run.ready.clear()
    # A workflow stored before blueprints keeps only its live tasks.
    run.registry.blueprints.pop(run.workflow_id)
    record_dispatch(run.runtime, writer, cast(Any, _worker()))
    with run.runtime._lock:
        (child,) = run.engine.route_boundary_event(
            writer,
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN,
                call_correlation="c0",
                child_region_ref="writer",
            ),
        ).ready
        run.runtime._committer.save_ledger_locked(run.workflow_id)

    restored = await run.restart()
    record = restored.runtime.get_record(child)
    assert record is not None and record.status == TaskStatus.PENDING
    assert child in restored.ready


@pytest.mark.anyio
async def test_work_with_no_blueprint_fails_its_workflow_by_name() -> None:
    run = await _Run().start(_workflow(_SPAWN_AND_LOOP, _LOOP))
    run.run("seed", {"items": ["a", "b"]})
    kids = [t for t in run.ready if run.name(t) == "kid"]
    registry = run.registry
    for kid in kids:
        del registry.task_blobs[kid]
        registry.dynamic_task_ids[run.workflow_id].discard(kid)
    blueprints = registry.blueprints[run.workflow_id]
    registry.blueprints[run.workflow_id] = [
        p for p in blueprints if p.record.task_id != run.ids["kid"]
    ]

    restored = await run.restart()
    failure = restored.engine.control_failure()
    assert failure is not None and failure.startswith("BlueprintMissing")
    assert restored.registry.control[run.workflow_id].failure == failure


@pytest.mark.anyio
async def test_a_workflow_this_server_cannot_read_fails_alone() -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    registry = run.registry
    other = run.workflow_id
    broken, _ = await _register(run.runtime, _workflow(_DIAMOND))
    registry.v2_blobs[broken] = "{}"

    restored = _Run(registry, run.reader)
    assert await restored.runtime.rehydrate() == 1
    assert restored.runtime.orchestration_engine(other) is not None
    assert restored.runtime.orchestration_engine(broken) is None
    failure = registry.control[broken].failure
    assert failure is not None and failure.startswith("UnsupportedWorkflowVersion")


@pytest.mark.anyio
async def test_a_workflow_this_server_cannot_read_settles_failed_and_closes(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    registry = run.registry
    broken, ids = await _register(run.runtime, _workflow(_DIAMOND))
    classify = ids["classify"]
    record_dispatch(run.runtime, classify, cast(Any, _worker()), "dsp-1")
    run.runtime.mark_started(classify, "wkr-1", {}, _TS)
    registry.v2_blobs[broken] = "{}"

    restored = _Run(registry, run.reader)
    closed: list[str] = []
    restored.runtime.set_completion_notifier(closed.append)
    revoked: list[Any] = []

    def publish_revoke(*args: Any) -> int:
        revoked.append(args)
        return 1

    monkeypatch.setattr(
        restored.runtime._worker_registry, "publish_revoke", publish_revoke
    )
    await restored.runtime.rehydrate()
    restored.runtime._act_after_commit()

    failure = registry.control[broken].failure
    assert failure is not None
    assert all(
        persisted.record.status == TaskStatus.FAILED
        and persisted.record.error == failure
        for task_id in registry.workflow_task_ids[broken]
        if (persisted := registry.load_task_states(task_id)[0]) is not None
    )
    assert not registry.remaining_of(broken)
    assert restored.runtime.workflow_settlement(broken).settled
    assert broken in closed
    assert len(revoked) == 1

    caplog.clear()
    again = _Run(registry, run.reader)
    await again.runtime.rehydrate()
    assert not [r for r in caplog.records if r.exc_info is not None]


@pytest.mark.anyio
async def test_a_running_workflow_through_a_pre_contract_branch_fails_on_restart() -> (
    None
):
    run = await _Run().start(_workflow(_DIAMOND))
    run.run("classify", {"label": "yes"})
    registry = run.registry
    bundle = PersistedV2Workflow.model_validate_json(registry.v2_blobs[run.workflow_id])
    decide = next(
        e.logical_ref for e in bundle.template.source_map if e.source_id == "decide"
    )
    operators = tuple(
        op.model_copy(update={"rule": None}) if op.operator_id == decide else op
        for op in bundle.template.operators
    )
    registry.v2_blobs[run.workflow_id] = bundle.model_copy(
        update={"template": bundle.template.model_copy(update={"operators": operators})}
    ).model_dump_json()

    restored = await run.restart()
    failure = restored.engine.control_failure()
    assert failure is not None
    assert failure.startswith("LegacyControlRegionUnsupported")
    record = restored.runtime.get_record(run.ids["left_work"])
    assert record is not None and record.status == TaskStatus.FAILED


_ZERO_FAN = f"""
      - name: planner
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [planner]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""


@pytest.mark.anyio
async def test_a_fan_out_read_again_that_settles_its_workflow_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(_ZERO_FAN))
    closed: list[str] = []
    run.runtime.set_completion_notifier(closed.append)
    reads: list[str] = []
    read_fanout = runtime_facade.fanout.read_fanout

    def read(results: Any, task_id: str, binding: Any) -> Any:
        reads.append(task_id)
        if len(reads) == 1:
            return runtime_facade.fanout.FanoutRead(error="away", unavailable=True)
        return read_fanout(results, task_id, binding)

    monkeypatch.setattr(runtime_facade.fanout, "read_fanout", read)
    (planner,) = run.ready
    run.ready.clear()
    _succeed(run, planner, {"items": []})
    assert not run.settled()
    closed.clear()

    run.runtime._redrive_workflow(run.workflow_id)

    assert run.settled() and closed == [run.workflow_id]
