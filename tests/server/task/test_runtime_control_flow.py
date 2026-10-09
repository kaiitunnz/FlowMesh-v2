"""Branches, loops and region-definition children through the task runtime: tasks
made from blueprints, routes read off the lock, dead routes settled without running,
and workflow status that reflects what no task holds."""

import logging
from typing import Any, cast

import pytest

from server.config import OrchestrationConfig
from server.orchestration import OrchestrationEngine, PublicationOutcome
from server.registries.workflow import PersistedTask
from server.task.models import TaskLoopTime, TaskOccurrence, TaskStatus
from server.task.redrive import StoreRedriveScheduler
from server.task.runtime import TaskRuntime
from tests.server.credential_vault_helpers import InMemoryCredentialVault
from tests.server.dispatch_helpers import record_dispatch
from tests.server.result_store import make_result_reader, result_payload
from tests.server.task.test_v2_orchestration import (
    _TS,
    FakeRegistry,
    _register,
    _worker,
)
from tests.support.waiting import pop_ready

_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _workflow(nodes: str, templates: str = "") -> str:
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: cf}}
spec:
  graph:
{templates}
    nodes:
{nodes}
"""


class _Run:
    """One workflow on a real runtime, with its read re-drives run on demand."""

    def __init__(
        self,
        registry: FakeRegistry | None = None,
        reader: Any = None,
        config: OrchestrationConfig | None = None,
    ) -> None:
        self.registry = registry or FakeRegistry()
        self.reader = reader or make_result_reader()
        self.schedulers: list[StoreRedriveScheduler] = []

        def redrive(fire: Any, logger: logging.Logger) -> StoreRedriveScheduler:
            self.schedulers.append(
                StoreRedriveScheduler(fire, logger, clock=lambda: 0.0, run_thread=False)
            )
            return self.schedulers[-1]

        self.runtime = TaskRuntime(
            cast(Any, self.registry),
            cast(Any, _StubWorkers()),
            config or OrchestrationConfig(),
            self.reader,
            logging.getLogger("control-flow"),
            credential_vault=InMemoryCredentialVault(),
            redrive=redrive,
        )
        self.workflow_id = ""
        self.ids: dict[str, str] = {}
        self.ready: list[str] = []
        self.succeeded: list[str] = []

    async def start(self, text: str) -> "_Run":
        self.workflow_id, self.ids = await _register(self.runtime, text)
        self.drive()
        return self

    async def restart(self) -> "_Run":
        restored = _Run(self.registry, self.reader)
        assert await restored.runtime.rehydrate() == 1
        restored.workflow_id, restored.ids = self.workflow_id, self.ids
        restored.drive()
        return restored

    @property
    def engine(self) -> OrchestrationEngine:
        engine = self.runtime.orchestration_engine(self.workflow_id)
        assert engine is not None
        return engine

    def drive(self) -> None:
        """Run every due re-drive, then take every ready task."""
        while any(scheduler.run_due() for scheduler in self.schedulers):
            pass
        while self.runtime.ready_queue_length() > 0:
            if (task_id := pop_ready(self.runtime)) is None:
                break
            self.ready.append(task_id)

    def name(self, task_id: str) -> str:
        wi = self.engine.work_item(task_id)
        assert wi is not None
        names = {op: name for name, op in self.ids.items()}
        return names[wi.operator_id]

    def run(self, name: str, result: dict[str, Any] | None = None) -> str:
        """Dispatch the one ready task of ``name`` and settle it with ``result``."""
        task_id = next(t for t in self.ready if self.name(t) == name)
        self.ready.remove(task_id)
        record = self.runtime.get_record(task_id)
        assert record is not None
        payload = (
            result_payload(self.reader, task_id, {"ok": True, **result}, record.org_id)
            if result is not None
            else {}
        )
        record_dispatch(self.runtime, task_id, cast(Any, _worker()))
        self.runtime.mark_started(task_id, "wkr-1", {}, _TS)
        self.runtime.mark_succeeded(task_id, "wkr-1", payload, _TS)
        self.succeeded.append(name)
        self.drive()
        return task_id

    def settled(self) -> bool:
        return self.runtime.workflow_settlement(self.workflow_id).settled


class _StubWorkers:
    def get_worker(self, worker_id: str) -> Any:
        return _worker(worker_id)

    def publish_interrupt(self, *args: Any) -> int:
        return 0

    def publish_revoke(self, *args: Any) -> int:
        return 0

    def release_worker(self, *args: Any) -> bool:
        return False

    def reservations(self) -> list[Any]:
        return []


_DIAMOND = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: input, field: [label], cases: {{yes: left, no: right}}}}
      - name: left_work
        dependsOn: [{{node: decide, port: left, input: in}}]
        spec: {_ECHO}
      - name: right_work
        dependsOn: [{{node: decide, port: right}}]
        spec: {_ECHO}
      - name: merged
        dependsOn: [{{node: left_work, input: l}}, {{node: right_work, input: r}}]
        region: {{kind: merge, combination: one_live}}
      - name: after
        dependsOn: [{{node: merged, input: m}}]
        spec: {_ECHO}
""".replace("yes: left, no: right", "'yes': left, 'no': right")


@pytest.mark.anyio
async def test_a_branch_routes_by_its_stored_selector_and_skips_the_dead_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    reported: list[str] = []
    on_succeeded = OrchestrationEngine.on_succeeded

    def _record(engine: OrchestrationEngine, task_id: str, **kw: Any) -> Any:
        reported.append(task_id)
        return on_succeeded(engine, task_id, **kw)

    monkeypatch.setattr(OrchestrationEngine, "on_succeeded", _record)
    run.run("classify", {"label": "yes"})

    right = run.ids["right_work"]
    record = run.runtime.get_record(right)
    # The dead arm settles done without running, an empty result saying why, and
    # never as a reported success that would make its routes live.
    assert record is not None and record.status is TaskStatus.DONE
    assert record.result_skip == {"skipped": True, "reason": "route_not_taken"}
    assert right not in reported
    assert right not in run.registry.remaining_of(run.workflow_id)
    assert [run.name(t) for t in run.ready] == ["left_work"]
    run.run("left_work")
    run.run("after")
    assert run.succeeded == ["classify", "left_work", "after"]
    assert run.settled()


@pytest.mark.anyio
async def test_a_workflow_waiting_on_a_selector_read_stays_open() -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    classify = run.ready.pop()
    record = run.runtime.get_record(classify)
    assert record is not None
    payload = result_payload(
        run.reader, classify, {"ok": True, "label": "no"}, record.org_id
    )
    record_dispatch(run.runtime, classify, cast(Any, _worker()))
    run.runtime.mark_succeeded(classify, "wkr-1", payload, _TS)
    # Every task but the arms has settled; the branch still waits on its read.
    with run.runtime._lock:
        assert run.engine.awaits_control_reads()
    assert run.registry.control[run.workflow_id].open
    assert not run.settled()
    run.drive()
    with run.runtime._lock:
        assert not run.engine.awaits_control_reads()
    assert [run.name(t) for t in run.ready] == ["right_work"]
    run.run("right_work")
    run.run("after")
    assert not run.registry.control[run.workflow_id].open


_LOOP = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: route
            dependsOn: [{{node: step, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_LOOP_NODES = f"""
      - name: seed
        spec: {_ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
"""

_CONSUMED = _LOOP_NODES + f"""      - name: consume
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {_ECHO}
"""


@pytest.mark.anyio
async def test_a_loop_runs_its_body_at_each_time_until_it_exits() -> None:
    run = await _Run().start(_workflow(_CONSUMED, _LOOP))
    run.run("seed")
    run.run("step", {"route": "again"})
    run.run("step", {"route": "again"})
    assert not run.settled()
    run.run("step", {"route": "done"})
    run.run("consume")
    assert run.succeeded == ["seed", "step", "step", "step", "consume"]
    # A body member is a blueprint, never one of the workflow's own tasks.
    assert run.runtime.get_record(run.ids["step"]) is None
    assert run.settled()


@pytest.mark.anyio
async def test_a_loop_resumes_after_a_restart_between_times() -> None:
    run = await _Run().start(_workflow(_CONSUMED, _LOOP))
    run.run("seed")
    run.run("step", {"route": "again"})
    restored = await run.restart()
    assert [restored.name(t) for t in restored.ready] == ["step"]
    restored.run("step", {"route": "done"})
    restored.run("consume")
    assert restored.settled()


@pytest.mark.anyio
async def test_a_control_failure_no_task_holds_fails_the_workflow() -> None:
    run = await _Run(config=OrchestrationConfig(max_loop_iterations=1)).start(
        _workflow(_LOOP_NODES, _LOOP)
    )
    run.run("seed")
    run.run("step", {"route": "again"})
    # Every task succeeded; the loop failed on its budget and that is the verdict.
    failure = run.registry.control[run.workflow_id].failure
    assert failure is not None and "LoopIterationBudgetExceeded" in failure
    assert run.settled()


_CHILD = f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: work
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {_ECHO}
        edges:
          - from: {{node: work}}
            to: {{node: $return, port: out}}
"""

_ARM_FANOUT = f"""
      - name: classify
        spec: {_ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: go}}, {{name: stop}}]
          selection: {{input: input, field: [label]}}
      - name: fan
        dependsOn: [{{node: decide, port: go}}]
        region: {{kind: spawn, child: one, result: {{visibility: published}}}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: after
        dependsOn: [collect]
        spec: {_ECHO}
"""


@pytest.mark.anyio
async def test_a_spawn_fans_out_over_a_branch_arm_into_template_children() -> None:
    run = await _Run().start(_workflow(_ARM_FANOUT, _CHILD))
    run.run("classify", {"label": "go", "items": ["a", "b"]})
    assert [run.name(t) for t in run.ready] == ["work", "work"]
    run.run("work")
    assert not run.settled()
    run.run("work")
    run.run("after")
    members = [
        p
        for p in run.engine.to_snapshot().result_publications
        if p.output_id == "collection:fan"
    ]
    assert len(members) == 2
    assert all(p.outcome is PublicationOutcome.SUCCESS for p in members)
    assert run.settled()


@pytest.mark.anyio
async def test_an_unselected_spawn_makes_no_children_and_the_workflow_closes() -> None:
    run = await _Run().start(_workflow(_ARM_FANOUT, _CHILD))
    run.run("classify", {"label": "stop", "items": ["a", "b"]})
    assert run.ready == []
    record = run.runtime.get_record(run.ids["after"])
    assert record is not None and record.result_skip is not None
    assert run.settled()


@pytest.mark.anyio
async def test_a_template_stored_as_a_task_reads_as_a_blueprint_after_a_restart() -> (
    None
):
    run = await _Run().start(_workflow(_ARM_FANOUT, _CHILD))
    work = run.ids["work"]
    blueprint = run.runtime._occurrences.blueprint_locked(run.workflow_id, work)
    assert blueprint is not None
    # A workflow stored before blueprints held its template as one of its tasks.
    registry = run.registry
    registry.blueprints.pop(run.workflow_id)
    registry.workflow_task_ids[run.workflow_id].append(work)
    registry.task_blobs[work] = PersistedTask(record=blueprint).model_dump_json()
    registry.remaining[run.workflow_id].add(work)

    restored = await run.restart()
    assert restored.runtime.get_record(work) is None
    assert work not in registry.remaining_of(run.workflow_id)
    restored.run("classify", {"label": "go", "items": ["a"]})
    assert [restored.name(t) for t in restored.ready] == ["work"]


_CAPTURING = f"""
    templates:
      - name: researcher
        inputs:
          - {{name: topic, role: param}}
          - {{name: dataset, role: capture}}
        returns: [{{name: answer}}]
        nodes:
          - name: work
            dependsOn:
              - {{node: $ingress, port: topic, input: topic}}
              - {{node: $ingress, port: dataset, input: dataset}}
            spec: {_ECHO}
        edges:
          - from: {{node: work}}
            to: {{node: $return, port: answer}}
"""

_CAPTURE_FAN = f"""
      - name: plan
        spec: {_ECHO}
      - name: dataset_source
        spec: {_ECHO}
      - name: fan
        dependsOn: [plan, {{node: dataset_source, input: dataset}}]
        region: {{kind: spawn, child: researcher}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""


@pytest.mark.anyio
@pytest.mark.parametrize("capture_last", [False, True])
async def test_a_spawn_fans_out_over_its_param_never_its_capture(
    capture_last: bool,
) -> None:
    run = await _Run().start(_workflow(_CAPTURE_FAN, _CAPTURING))
    results = {
        "plan": {"items": ["t1", "t2"]},
        "dataset_source": {"items": ["d1", "d2", "d3"]},
    }
    order = ["plan", "dataset_source"] if capture_last else ["dataset_source", "plan"]
    for name in order:
        run.run(name, results[name])
    assert [run.name(t) for t in run.ready] == ["work", "work"]


_PROJECTED_FAN = f"""
      - name: plan
        spec: {_ECHO}
      - name: fan
        dependsOn: [{{node: plan, project: [nested]}}]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""


@pytest.mark.anyio
async def test_a_spawn_over_a_projection_fans_out_over_the_projected_list() -> None:
    run = await _Run().start(_workflow(_PROJECTED_FAN, _CHILD))
    run.run("plan", {"items": ["a", "b", "c"], "nested": ["x", "y"]})
    assert [run.name(t) for t in run.ready] == ["work", "work"]


@pytest.mark.anyio
async def test_a_branch_reads_its_selector_through_its_inputs_projection() -> None:
    nodes = _DIAMOND.replace(
        "dependsOn: [{node: classify, input: input}]",
        "dependsOn: [{node: classify, input: input, project: [inner]}]",
    )
    run = await _Run().start(_workflow(nodes))
    run.run("classify", {"label": "no", "inner": {"label": "yes"}})
    assert [run.name(t) for t in run.ready] == ["left_work"]


@pytest.mark.anyio
async def test_a_child_of_a_projected_fan_out_runs_on_its_element_of_the_part() -> None:
    nodes = f"""
      - name: plan
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [{{node: plan, project: [nested]}}]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""
    run = await _Run().start(_workflow(nodes))
    run.run("plan", {"items": ["a", "b", "c"], "nested": ["x", "y"]})
    kids = [t for t in run.ready if run.name(t) == "kid"]
    elements = [run.runtime.input_element(kid) for kid in kids]
    assert [(e.element, e.path) for e in elements if e is not None] == [
        (None, ("nested", 0)),
        (None, ("nested", 1)),
    ]


@pytest.mark.anyio
async def test_a_loop_body_task_names_its_member_and_loop_time() -> None:
    run = await _Run().start(_workflow(_LOOP_NODES, _LOOP))
    seed = run.run("seed")
    first = run.run("step", {"route": "again"})
    (second,) = [t for t in run.ready if run.name(t) == "step"]

    places = [run.runtime.describe_task(t) for t in (first, second)]
    assert [p.occurrence for p in places if p is not None] == [
        TaskOccurrence(
            member="body/step", time=[TaskLoopTime(loop="refine", iteration=0)]
        ),
        TaskOccurrence(
            member="body/step", time=[TaskLoopTime(loop="refine", iteration=1)]
        ),
    ]
    root = run.runtime.describe_task(seed)
    assert root is not None and root.occurrence is None
