"""Loops through the orchestration engine: routed feedback and exit, pipelined
logical times, frontier-gated release, budgets, failure and restart."""

import pytest

from server.orchestration import ScopeBudget
from server.orchestration.state import (
    ControlStatus,
    IterationKind,
    LoopInstanceStatus,
    PublicationOutcome,
    ValueRef,
    WorkItemStatus,
)
from server.task.v2.compiler import region_checks

from .control_flow import ECHO, Driver, workflow

_BODY = f"""
    templates:
      - name: body
        inputs:
          - {{name: state, role: carried}}
          - {{name: dataset, role: invariant}}
        nodes:
          - name: step
            dependsOn:
              - {{node: $ingress, port: state, input: state}}
              - {{node: $ingress, port: dataset, input: dataset}}
            spec: {ECHO}
          - name: side
            dependsOn: [{{node: $ingress, port: state, input: s}}]
            spec: {ECHO}
          - name: route
            dependsOn: [{{node: step, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: continue}}, {{name: finish}}]
              selection: {{input: input, field: [route]}}
        edges:
          - from: {{node: route, port: continue}}
            project: [state]
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: finish}}
            project: [state]
            to: {{node: $egress, port: state}}
"""

_NODES = f"""
      - name: seed
        spec: {ECHO}
      - name: data
        spec: {ECHO}
      - name: refine
        dependsOn:
          - {{node: seed, input: state}}
          - {{node: data, input: dataset}}
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
          invariants: [{{name: dataset}}]
          result: {{visibility: published}}
      - name: consume
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {ECHO}
"""


def _loop(budget: ScopeBudget | None = None) -> Driver:
    run = Driver(workflow(_NODES, _BODY), budget=budget)
    run.run_one("seed")
    run.run_one("data")
    return run


def _time(run: Driver, task_id: str) -> int:
    occurrence = run.engine.occurrence_of(task_id)
    assert occurrence is not None
    return occurrence.time[-1].iteration


def test_every_time_runs_its_body_and_the_exit_publishes_the_exiting_value() -> None:
    run = _loop()
    steps: list[str] = []
    for decision in ("continue", "continue", "continue", "finish"):
        (step,) = run.ready_named("step")
        steps.append(step)
        run.run(step)
        run.select(decision)
    # The body ran at times 0..3; each time has its own task and occurrence.
    assert [_time(run, s) for s in steps] == [0, 1, 2, 3]
    assert len(set(steps)) == 4
    # Side work at every time still holds the loop open.
    assert run.ready_named("consume") == []
    for side in list(run.ready_named("side")):
        run.run(side)
    (consume,) = run.ready_named("consume")
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.RELEASED
    exited = instance.exit_bundle["state"]
    # The exit carries the value routed at the exit time, not any later record.
    assert exited.legacy_task_id == steps[-1]
    assert exited.projection == ("state",)
    (_, decl), *_ = run.engine.published_outputs()
    publication = run.engine.output_publication(decl.output_id)
    assert publication is not None and publication.value_ref == exited
    for time in range(3):
        resolution = run.engine.iteration("refine", time)
        assert resolution is not None and resolution.kind is IterationKind.FEEDBACK
    exit_resolution = run.engine.iteration("refine", 3)
    assert exit_resolution is not None and exit_resolution.kind is IterationKind.EXIT
    run.run(consume)


def test_feedback_starts_the_next_time_while_earlier_work_is_outstanding() -> None:
    run = _loop()
    (step0,) = run.ready_named("step")
    (side0,) = run.ready_named("side")
    run.run(step0)
    run.select("continue")
    # Time 1 runs at once, though time 0's side task is still outstanding.
    (step1,) = run.ready_named("step")
    assert _time(run, step1) == 1
    run.run(step1)
    run.select("finish")
    # The exit is accepted but cannot leave while time 0's work can still arrive.
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.EXITED
    for side in [s for s in run.ready_named("side") if s != side0]:
        run.run(side)
    assert run.ready_named("consume") == []
    run.run(side0)
    assert run.ready_named("consume") != []


def test_invariants_are_read_at_every_time_and_carried_values_are_replaced() -> None:
    run = _loop()
    (step0,) = run.ready_named("step")
    run.run(step0)
    run.select("continue")
    (step1,) = run.ready_named("step")
    occurrence = run.engine.occurrence_of(step1)
    assert occurrence is not None
    instance = run.engine.loop_instance("refine")
    assert instance is not None
    # Time 1 starts from the feedback bundle; the invariant is the ingress binding.
    feedback = run.engine.iteration("refine", 0)
    assert feedback is not None
    assert feedback.bundle["state"].legacy_task_id == step0
    assert instance.invariants["dataset"].legacy_task_id == run.ops["data"]


def test_a_time_routing_neither_feedback_nor_exit_fails_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Submission refuses an arm leading to neither; a stored plan can still hold one.
    monkeypatch.setattr(region_checks, "_check_return_routes", lambda *_: [])
    body = _BODY.replace("{name: finish}]", "{name: finish}, {name: neither}]")
    run = Driver(workflow(_NODES, body))
    run.run_one("seed")
    run.run_one("data")
    run.run_one("step")
    run.run_one("side")
    run.select("neither")
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.FAILED
    state = run.engine.control_state("refine")
    assert state is not None and state.status is ControlStatus.FAILED
    assert "LoopControlViolation" in (state.reason or "")
    assert run.status("consume") is WorkItemStatus.SETTLED
    assert run.ops["consume"] in run.failed


def test_the_iteration_budget_admits_times_below_it_and_fails_the_next() -> None:
    run = _loop(ScopeBudget(max_loop_iterations=1))
    run.run_one("step")
    run.run_one("side")
    run.select("continue")
    state = run.engine.control_state("refine")
    assert state is not None and state.status is ControlStatus.FAILED
    assert "LoopIterationBudgetExceeded" in (state.reason or "")
    # No time-1 work exists.
    assert run.ready_named("step") == []
    assert run.ops["consume"] in run.failed

    exits = _loop(ScopeBudget(max_loop_iterations=1))
    exits.run_one("step")
    exits.run_one("side")
    exits.select("finish")
    assert exits.ready_named("consume") != []


def test_the_iteration_budget_is_pinned_across_restart() -> None:
    run = _loop(ScopeBudget(max_loop_iterations=2))
    run.run_one("step")
    run.run_one("side")
    run.select("continue")
    run.restore()  # rebuilt with the default budget
    run.run_one("step")
    run.run_one("side")
    run.select("continue")
    state = run.engine.control_state("refine")
    assert state is not None and state.status is ControlStatus.FAILED


def test_a_body_failure_fails_the_loop_and_withdraws_later_work() -> None:
    run = _loop()
    run.run_one("step")
    run.select("continue")
    (side0,) = [s for s in run.ready_named("side") if _time(run, s) == 0]
    run.run(side0, fail=True)
    state = run.engine.control_state("refine")
    assert state is not None and state.status is ControlStatus.FAILED
    # Time 1's outstanding work is withdrawn, not left running.
    for task_id in list(run.ready):
        wi = run.engine.work_item(task_id)
        if run.engine.occurrence_of(task_id) is not None:
            assert wi is not None and wi.status is WorkItemStatus.CANCELLED
    assert run.ops["consume"] in run.failed
    # The accepted feedback at time 0 stays recorded.
    feedback = run.engine.iteration("refine", 0)
    assert feedback is not None and feedback.kind is IterationKind.FEEDBACK


def test_cancellation_withdraws_an_open_loop() -> None:
    run = _loop()
    run.run_one("step")
    run.apply(run.engine.cancel_instance())
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.status is LoopInstanceStatus.CANCELLED
    assert run.engine.pending_branch_reads() == []


def test_a_loop_survives_restart_mid_iteration_without_rereading_its_inputs() -> None:
    run = _loop()
    run.run_one("step")
    run.select("continue")
    run.restore()
    instance = run.engine.loop_instance("refine")
    assert instance is not None and instance.times == 2
    (step1,) = run.ready_named("step")
    run.run(step1)
    run.select("finish")
    for side in list(run.ready_named("side")):
        run.run(side)
    assert run.ready_named("consume") != []


def test_a_dead_ingress_skips_the_loop_without_entering_its_body() -> None:
    nodes = f"""
      - name: classify
        spec: {ECHO}
      - name: decide
        dependsOn: [{{node: classify, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: go}}, {{name: skip}}]
          selection: {{input: input}}
      - name: data
        spec: {ECHO}
      - name: refine
        dependsOn:
          - {{node: decide, port: go, input: state}}
          - {{node: data, input: dataset}}
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
          invariants: [{{name: dataset}}]
"""
    run = Driver(workflow(nodes, _BODY))
    run.run_one("classify")
    run.run_one("data")
    run.select("skip")
    state = run.engine.control_state("refine")
    assert state is not None and state.status is ControlStatus.DEAD
    assert run.engine.loop_instance("refine") is None
    assert run.ready == []


_CHILD = f"""
    templates:
      - name: body
        inputs:
          - {{name: state, role: carried}}
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {ECHO}
          - name: route
            dependsOn: [{{node: step, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: continue}}, {{name: finish}}]
              selection: {{input: input}}
        edges:
          - from: {{node: route, port: continue}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: finish}}
            to: {{node: $egress, port: state}}
      - name: researcher
        inputs: [{{name: topic, role: param}}]
        returns: [{{name: answer}}]
        nodes:
          - name: inner
            dependsOn: [{{node: $ingress, port: topic, input: state}}]
            region:
              kind: loop
              body_ref: body
              loop_coordinate: round
              carried: [{{name: state}}]
        edges:
          - from: {{node: inner, port: state}}
            to: {{node: $return, port: answer}}
"""

_SPAWN = f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: researcher}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: after
        dependsOn: [{{node: collect, input: all}}]
        spec: {ECHO}
"""


def test_loops_inside_two_spawned_children_keep_their_contexts_apart() -> None:
    run = Driver(workflow(_SPAWN, _CHILD))
    run.run_one("plan")
    for element in ("a", "b"):
        run.apply(
            run.engine.enter_definition_child(
                "fan", ValueRef(kind="inline", literal=element)
            )
        )
    run.apply(run.engine.seal_spawn("fan"))
    steps = run.ready_named("step")
    assert len(steps) == 2
    contexts = {run.engine.occurrence_of(s).context_id for s in steps}  # type: ignore[union-attr]
    assert len(contexts) == 2
    # Both children are at their own time 0 of their own loop instance.
    loops = {run.engine.occurrence_of(s).time[-1].loop for s in steps}  # type: ignore[union-attr]
    assert len(loops) == 2

    def answer(context: str, decision: str) -> None:
        task = next(
            t
            for t in run.ready_named("step")
            if run.engine.occurrence_of(t).context_id == context  # type: ignore[union-attr]
        )
        run.run(task)
        key = next(k for k, _ in run.engine.pending_branch_reads() if context in k)
        run.apply(run.engine.accept_branch_selection(key, decision))

    first, second = sorted(contexts)
    answer(first, "continue")
    answer(second, "finish")
    assert run.ready_named("after") == []
    answer(first, "finish")
    (after,) = run.ready_named("after")
    collect = run.engine.control_state("collect")
    assert collect is not None and collect.status is ControlStatus.LIVE
    members = collect.outputs[""].members
    assert [m.outcome for m in members] == [PublicationOutcome.SUCCESS] * 2
    run.run(after)


_SPAWN_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {ECHO}
          - name: kid
            spec: {ECHO}
          - name: fan
            dependsOn: [step]
            region: {{kind: spawn, child: kid}}
          - name: collect
            dependsOn: [fan]
            region: {{kind: join, completion: all_settled}}
          - name: route
            dependsOn: [{{node: collect, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input}}
        edges:
          - from: {{node: route, port: again}}
            to: {{node: $feedback, port: state}}
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_SPAWN_LOOP = f"""
      - name: seed
        spec: {ECHO}
      - name: refine
        dependsOn: [{{node: seed, input: state}}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{{name: state}}]
      - name: consume
        dependsOn: [{{node: refine, port: state, input: in}}]
        spec: {ECHO}
"""


def _fan_at(run: Driver, time: int) -> str:
    op = run.ops["body/fan"]
    (occurrence,) = [
        o for o in run.engine.occurrences(op) if o.time[-1].iteration == time
    ]
    return occurrence.activation_id


def test_a_spawn_and_join_inside_a_loop_close_per_time() -> None:
    run = Driver(workflow(_SPAWN_LOOP, _SPAWN_BODY))
    run.run_one("seed")
    run.run_one("step")
    fan0 = _fan_at(run, 0)
    for element in ("a", "b"):
        run.apply(
            run.engine.materialize_child(
                fan0, value_ref=ValueRef(kind="inline", literal=element)
            )
        )
    run.apply(run.engine.seal_spawn(fan0))
    kids = run.ready_named("kid")
    assert len(kids) == 2
    # The join waits for its own children, not for the loop's feedback.
    run.run(kids[0])
    assert run.engine.pending_branch_reads() == []
    run.run(kids[1])
    run.select("again")
    run.run_one("step")
    fan1 = _fan_at(run, 1)
    assert fan1 != fan0
    # A later time's spawn is a distinct occurrence with its own child-init scope;
    # a zero-child spawn closes once sealed.
    run.apply(run.engine.seal_spawn(fan1))
    run.select("done")
    assert run.ready_named("consume") != []


def test_an_agent_in_a_loop_body_runs_as_a_fresh_activation_each_time() -> None:
    body = _BODY.replace(
        f"""          - name: side
            dependsOn: [{{node: $ingress, port: state, input: s}}]
            spec: {ECHO}
""",
        """          - name: side
            dependsOn: [{node: $ingress, port: state, input: s}]
            spec:
              taskType: agent
              task: think
              harness: {backend: scripted, version: v1, params: {script: []}}
""",
    )
    run = Driver(workflow(_NODES, body))
    run.run_one("seed")
    run.run_one("data")
    run.run_one("step")
    run.select("continue")
    sides = run.ready_named("side")
    assert len(sides) == 2
    activations = {run.engine.work_item(s).activation_id for s in sides}  # type: ignore[union-attr]
    assert len(activations) == 2
    for side in sides:
        accepted = run.engine.accepted_inputs_for_task(side)
        assert [a.target_port for a in accepted] == ["s"]


def test_an_inner_loop_reenters_at_time_zero_at_each_outer_time() -> None:
    templates = f"""
    templates:
      - name: inner_body
        inputs: [{{name: x, role: carried}}]
        nodes:
          - name: work
            dependsOn: [{{node: $ingress, port: x, input: x}}]
            spec: {ECHO}
          - name: inner_route
            dependsOn: [{{node: work, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input}}
        edges:
          - from: {{node: inner_route, port: again}}
            to: {{node: $feedback, port: x}}
          - from: {{node: inner_route, port: done}}
            to: {{node: $egress, port: x}}
      - name: outer_body
        inputs: [{{name: y, role: carried}}]
        nodes:
          - name: inner
            dependsOn: [{{node: $ingress, port: y, input: x}}]
            region:
              kind: loop
              body_ref: inner_body
              loop_coordinate: inner
              carried: [{{name: x}}]
          - name: outer_route
            dependsOn: [{{node: inner, port: x, input: input}}]
            region:
              kind: branch
              inputs: [{{name: input}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: input}}
        edges:
          - from: {{node: outer_route, port: again}}
            to: {{node: $feedback, port: y}}
          - from: {{node: outer_route, port: done}}
            to: {{node: $egress, port: y}}
"""
    nodes = f"""
      - name: seed
        spec: {ECHO}
      - name: outer
        dependsOn: [{{node: seed, input: y}}]
        region:
          kind: loop
          body_ref: outer_body
          loop_coordinate: outer
          carried: [{{name: y}}]
      - name: consume
        dependsOn: [{{node: outer, port: y, input: in}}]
        spec: {ECHO}
"""
    run = Driver(workflow(nodes, templates))
    run.run_one("seed")
    seen: list[tuple[int, int]] = []
    for outer_decision in ("again", "done"):
        for inner_decision in ("again", "done"):
            (work,) = run.ready_named("work")
            occurrence = run.engine.occurrence_of(work)
            assert occurrence is not None
            seen.append(tuple(f.iteration for f in occurrence.time))  # type: ignore[arg-type]
            run.run(work)
            run.select(inner_decision)
        run.select(outer_decision)
    # The inner time restarts at 0 under each outer time.
    assert seen == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert run.ready_named("consume") != []


_EFFECT = (
    "{taskType: ssh, interactive: false, image: 'alpine:3', command: [sh, -c, 'true']}"
)


def _effect_admission(nodes: str, templates: str, *, definition: bool) -> list[str]:
    """Spawn one child running an effect leaf; return the child's authority
    decisions."""
    run = Driver(workflow(nodes, templates))
    run.run_one("plan")
    element = ValueRef(kind="inline", literal="e")
    if definition:
        run.apply(run.engine.enter_definition_child("fan", element))
    else:
        run.apply(run.engine.materialize_child("fan", value_ref=element))
    run.apply(run.engine.seal_spawn("fan"))
    (kid,) = run.ready
    wi = run.engine.work_item(kid)
    assert wi is not None and wi.status is WorkItemStatus.READY
    snapshot = run.engine.to_snapshot()
    return [
        d.kind.value
        for d in snapshot.authority_decisions
        if d.work_item_id == wi.work_item_id
    ]


def test_an_effect_leaf_is_admitted_alike_as_a_child_and_in_a_child_definition() -> (
    None
):
    shorthand = _effect_admission(
        f"""
      - name: plan
        spec: {ECHO}
      - name: kid
        spec: {_EFFECT}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
""",
        "",
        definition=False,
    )
    defined = _effect_admission(
        f"""
      - name: plan
        spec: {ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: one}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
""",
        f"""
    templates:
      - name: one
        inputs: [{{name: e, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: kid
            dependsOn: [{{node: $ingress, port: e, input: e}}]
            spec: {_EFFECT}
        edges:
          - from: {{node: kid}}
            to: {{node: $return, port: out}}
""",
        definition=True,
    )
    assert shorthand == defined == ["granted"]
