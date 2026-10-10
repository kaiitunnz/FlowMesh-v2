"""A branch selects on one input and passes on the value of another it declares."""

from typing import Any, cast

import pytest

from server.orchestration.state import ControlStatus
from server.task.models import TaskStatus
from server.task.v2.representations.operators import BranchRegion
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_file_inputs import _file, _trained
from tests.server.task.test_runtime_control_flow import _Run, _workflow
from tests.server.task.test_runtime_control_flow_faults import _succeed
from tests.server.task.test_scoped_inputs import _dispatch
from tests.server.task.test_v2_control_flow_compile import _codes
from tests.server.task.test_v2_orchestration import _TS, _worker

_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _items(read: str) -> str:
    return f'{{taskType: echo, data: {{type: list, items: ["${{{read}}}"]}}}}'


_TUNE_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: adapter, role: carried}}]
        nodes:
          - name: train
            dependsOn: [{{node: $ingress, port: adapter, input: prev}}]
            spec: {_items("prev.final_lora_archive")}
          - name: evaluate
            dependsOn: [{{node: train, input: model}}]
            spec: {_ECHO}
          - name: judge
            dependsOn:
              - {{node: evaluate, input: verdict}}
              - {{node: train, input: adapter}}
            region:
              kind: branch
              inputs: [{{name: verdict}}, {{name: adapter}}]
              outputs: [{{name: again}}, {{name: done}}]
              selection: {{input: verdict, field: [verdict]}}
              forward: adapter
        edges:
          - from: {{node: judge, port: again}}
            to: {{node: $feedback, port: adapter}}
          - from: {{node: judge, port: done}}
            to: {{node: $egress, port: adapter}}
"""

_TUNE = f"""
      - name: seed
        spec: {_ECHO}
      - name: tune
        dependsOn: [{{node: seed, input: adapter}}]
        region:
          kind: loop
          body_ref: body
          carried: [{{name: adapter}}]
      - name: serve
        dependsOn: [{{node: tune, port: adapter, input: final}}]
        spec: {_items("final.final_lora_archive")}
"""


@pytest.mark.anyio
async def test_a_loop_carries_the_trained_adapter_while_a_judge_decides() -> None:
    run = await _Run().start(_workflow(_TUNE, _TUNE_BODY))
    run.run("seed", _trained("tsk-seed"))
    assert _dispatch(run, "train")[0]["data"]["items"] == [_file("tsk-seed")]
    run.run("train", _trained("tsk-t0"))
    run.run("evaluate", {"verdict": "again"})

    assert _dispatch(run, "train")[0]["data"]["items"] == [_file("tsk-t0")]
    run.run("train", _trained("tsk-t1"))
    run.run("evaluate", {"verdict": "done"})

    assert _dispatch(run, "serve")[0]["data"]["items"] == [_file("tsk-t1")]


_DIAMOND = f"""
      - name: judge_src
        spec: {_ECHO}
      - name: payload
        spec: {_ECHO}
      - name: judge
        dependsOn:
          - {{node: judge_src, input: verdict}}
          - {{node: payload, input: value}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: value}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: value
      - name: after
        dependsOn: [{{node: judge, port: left, input: got}}]
        spec: {_items("got.v")}
"""


def _op_id(run: _Run, name: str) -> str:
    return next(
        op for op, source in run.engine._topology.source_ids.items() if source == name
    )


def _branch(run: _Run, name: str = "judge") -> BranchRegion:
    op = run.engine._topology.operators[_op_id(run, name)]
    assert isinstance(op, BranchRegion)
    return op


@pytest.mark.anyio
@pytest.mark.parametrize("payload_first", [True, False])
async def test_the_selected_port_carries_the_forwarded_value(
    payload_first: bool,
) -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    for name, result in (
        [("payload", {"v": "carried"}), ("judge_src", {"route": "left"})]
        if payload_first
        else [("judge_src", {"route": "left"}), ("payload", {"v": "carried"})]
    ):
        run.run(name, result)

    assert _dispatch(run, "after")[0]["data"]["items"] == ["carried"]
    decision = run.engine._ledger.branch_decisions[_op_id(run, "judge")]
    inspected = decision.input_ref
    assert inspected is not None
    assert run.name(inspected.legacy_task_id or "") == "judge_src"


@pytest.mark.anyio
async def test_a_branch_without_forward_passes_on_what_it_selects_on() -> None:
    run = await _Run().start(
        _workflow(_DIAMOND.replace("          forward: value\n", ""))
    )
    assert _branch(run).forward == "verdict"
    run.run("judge_src", {"route": "left", "v": "verdict"})
    run.run("payload", {"v": "carried"})

    assert _dispatch(run, "after")[0]["data"]["items"] == ["verdict"]


def test_forward_naming_no_input_of_the_branch_is_refused() -> None:
    text = _workflow(_DIAMOND.replace("forward: value", "forward: nope"))
    assert "branch.unknown-forward" in _codes(text)


def test_forward_and_selection_on_exclusive_arms_are_refused() -> None:
    text = _workflow(f"""
      - name: src
        spec: {_ECHO}
      - name: gate
        dependsOn: [{{node: src, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: a}}, {{name: b}}]
          selection: {{input: input, field: [route]}}
      - name: on_a
        dependsOn: [{{node: gate, port: a}}]
        spec: {_ECHO}
      - name: on_b
        dependsOn: [{{node: gate, port: b}}]
        spec: {_ECHO}
      - name: judge
        dependsOn:
          - {{node: on_a, input: verdict}}
          - {{node: on_b, input: value}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: value}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: value
""")
    assert "reads.exclusive" in _codes(text)


def _fail(run: _Run, name: str) -> None:
    task_id = next(t for t in run.ready if run.name(t) == name)
    run.ready.remove(task_id)
    record_dispatch(run.runtime, task_id, cast(Any, _worker()))
    run.runtime.fail_dispatch(task_id, "wkr-1", {}, _TS, error="boom", retryable=False)
    run.drive()


@pytest.mark.anyio
@pytest.mark.parametrize("failed_first", [True, False])
async def test_a_failed_forwarded_input_fails_the_branch(failed_first: bool) -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    if failed_first:
        _fail(run, "payload")
        run.run("judge_src", {"route": "left"})
    else:
        run.run("judge_src", {"route": "left"})
        _fail(run, "payload")

    judge = _op_id(run, "judge")
    state = run.engine._ledger.control_states.get(judge)
    assert run.engine._failures.region_failed(judge) or (
        state is not None and state.status is ControlStatus.FAILED
    )
    assert run.engine.control_failure() is None
    after = run.runtime.get_record(run.ids["after"])
    assert after is not None and after.status == TaskStatus.FAILED


_GATED = f"""
      - name: gate_src
        spec: {_ECHO}
      - name: gate
        dependsOn: [{{node: gate_src, input: input}}]
        region:
          kind: branch
          inputs: [{{name: input}}]
          outputs: [{{name: open}}, {{name: shut}}]
          selection: {{input: input, field: [route]}}
      - name: payload
        dependsOn: [{{node: gate, port: shut}}]
        spec: {_ECHO}
      - name: judge_src
        spec: {_ECHO}
      - name: judge
        dependsOn:
          - {{node: judge_src, input: verdict}}
          - {{node: payload, input: value}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: value}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: value
"""


@pytest.mark.anyio
@pytest.mark.parametrize("dead_first", [True, False])
async def test_a_dead_forwarded_input_leaves_the_branch_inactive(
    dead_first: bool,
) -> None:
    run = await _Run().start(_workflow(_GATED))
    steps = [("gate_src", {"route": "open"}), ("judge_src", {"route": "left"})]
    for name, result in steps if dead_first else steps[::-1]:
        run.run(name, result)

    state = run.engine._ledger.control_states[_op_id(run, "judge")]
    assert state.status is ControlStatus.DEAD


@pytest.mark.anyio
async def test_an_empty_forwarded_input_forwards_an_empty_value() -> None:
    run = await _Run().start(_workflow(f"""
      - name: judge_src
        spec: {_ECHO}
      - name: payload
        dependsOn: [judge_src]
        spec:
          taskType: echo
          data: {{type: list, items: [x]}}
          condition: {{node: judge_src, field: run, equals: "yes"}}
      - name: judge
        dependsOn:
          - {{node: judge_src, input: verdict}}
          - {{node: payload, input: value}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: value}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: value
"""))
    run.run("judge_src", {"route": "left", "run": "no"})
    payload = next(t for t in run.ready if run.name(t) == "payload")
    record_dispatch(run.runtime, payload, cast(Any, _worker()))
    run.runtime.mark_succeeded(
        payload,
        worker_id=None,
        payload={"finished_at": _TS, "started_at": _TS},
        ts=_TS,
        skip={"skipped": True, "reason": "condition_not_met"},
    )
    run.drive()

    state = run.engine._ledger.control_states[_op_id(run, "judge")]
    assert state.status is ControlStatus.LIVE
    assert state.outputs["left"].kind == "empty"


@pytest.mark.anyio
@pytest.mark.parametrize("after_decision", [False, True])
async def test_a_restart_keeps_the_exact_forwarded_binding(
    after_decision: bool,
) -> None:
    run = await _Run().start(_workflow(_DIAMOND))
    payload = run.run("payload", {"v": "carried"})
    judge_src = next(t for t in run.ready if run.name(t) == "judge_src")
    run.ready.remove(judge_src)
    _succeed(run, judge_src, {"route": "left"})
    if after_decision:
        run.drive()
    forwarded = run.engine._ledger.control_states[_op_id(run, "judge")].inputs["value"]
    assert forwarded.legacy_task_id == payload

    restored = await run.restart()

    state = restored.engine._ledger.control_states[_op_id(run, "judge")]
    assert state.inputs["value"] == forwarded
    assert state.status is ControlStatus.LIVE
    assert state.outputs["left"] == forwarded
    assert _dispatch(restored, "after")[0]["data"]["items"] == ["carried"]


@pytest.mark.anyio
async def test_a_forwarded_file_reads_as_its_producers_url() -> None:
    run = await _Run().start(
        _workflow(_DIAMOND.replace(_items("got.v"), _items("got.final_lora_archive")))
    )
    run.run("payload", _trained("tsk-payload"))
    run.run("judge_src", {"route": "left"})

    assert _dispatch(run, "after")[0]["data"]["items"] == [_file("tsk-payload")]


@pytest.mark.anyio
async def test_a_one_live_merge_and_a_fan_out_read_the_forwarded_value() -> None:
    run = await _Run().start(_workflow(f"""
      - name: judge_src
        spec: {_ECHO}
      - name: payload
        spec: {_ECHO}
      - name: judge
        dependsOn:
          - {{node: judge_src, input: verdict}}
          - {{node: payload, input: value}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: value}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: value
      - name: either
        dependsOn: [{{node: judge, port: left}}, {{node: judge, port: right}}]
        region: {{kind: merge, combination: one_live}}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [either]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
"""))
    run.run("payload", {"items": ["a", "b", "c"]})
    run.run("judge_src", {"route": "right", "items": ["only"]})

    assert [run.name(t) for t in run.ready] == ["kid", "kid", "kid"]


def test_a_branch_forwarding_an_aggregate_releases_no_single_value() -> None:
    text = _workflow(f"""
      - name: plan
        spec: {_ECHO}
      - name: kid
        spec: {_ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: kid}}
      - name: collect
        dependsOn: [fan]
        region: {{kind: join, completion: all_settled}}
      - name: judge_src
        spec: {_ECHO}
      - name: judge
        dependsOn:
          - {{node: judge_src, input: verdict}}
          - {{node: collect, input: all}}
        region:
          kind: branch
          inputs: [{{name: verdict}}, {{name: all}}]
          outputs: [{{name: left}}, {{name: right}}]
          selection: {{input: verdict, field: [route]}}
          forward: all
      - name: either
        dependsOn: [{{node: judge, port: left}}, {{node: judge, port: right}}]
        region: {{kind: merge, combination: one_live}}
      - name: kid2
        spec: {_ECHO}
      - name: fan2
        dependsOn: [either]
        region: {{kind: spawn, child: kid2}}
""")
    assert "dataflow.region-input" in _codes(text)


_CHILD = f"""
    templates:
      - name: per_item
        inputs: [{{name: item, role: param}}]
        returns: [{{name: out}}]
        nodes:
          - name: judge_src
            dependsOn: [{{node: $ingress, port: item, input: item}}]
            spec: {_ECHO}
          - name: payload
            dependsOn: [{{node: $ingress, port: item, input: item}}]
            spec: {_ECHO}
          - name: judge
            dependsOn:
              - {{node: judge_src, input: verdict}}
              - {{node: payload, input: value}}
            region:
              kind: branch
              inputs: [{{name: verdict}}, {{name: value}}]
              outputs: [{{name: keep}}, {{name: drop}}]
              selection: {{input: verdict, field: [route]}}
              forward: value
        edges:
          - from: {{node: judge, port: keep}}
            to: {{node: $return, port: out}}
            project: [v]
          - from: {{node: judge, port: drop}}
            to: {{node: $return, port: out}}
            project: [v]
"""


@pytest.mark.anyio
async def test_each_child_forwards_its_own_value() -> None:
    run = await _Run().start(
        _workflow(
            f"""
      - name: plan
        spec: {_ECHO}
      - name: fan
        dependsOn: [plan]
        region: {{kind: spawn, child: per_item}}
      - name: collect
        dependsOn: [fan]
        region:
          kind: join
          completion: all_settled
          result: {{visibility: published}}
""",
            _CHILD,
        )
    )
    run.run("plan", {"items": ["a", "b"]})
    for v in ("p0", "p1"):
        run.run("payload", {"v": v})
    run.run("judge_src", {"route": "keep"})
    run.run("judge_src", {"route": "keep"})

    assert run.settled()
    listed = run.runtime.published_outputs(run.workflow_id, "collect")
    assert listed is not None
    assert sorted(
        run.runtime.read_output(m).model_dump()["routed_value"] for m in listed.members
    ) == ["p0", "p1"]
