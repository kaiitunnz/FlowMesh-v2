"""Published region outputs read as the value an input reading them would read."""

import logging
from typing import Any

import pytest
from flowmesh.models import RoutedValue as SdkRoutedValue
from flowmesh.models import WorkflowOutputValue as SdkWorkflowOutputValue
from lumid_hooks import PrincipalContext

from server.orchestration import PublicationOutcome
from server.routers.v1 import outputs as outputs_router
from server.task.outputs import OutputMember
from shared.schemas.result import RoutedValue
from tests.server.dispatch_helpers import record_dispatch
from tests.server.task.test_runtime_control_flow import _Run, _worker, _workflow
from tests.server.task.test_v2_orchestration import _TS


def _served(run: _Run, member: OutputMember) -> Any:
    value = run.runtime.read_output(member)
    assert isinstance(value, RoutedValue)
    return value.routed_value


def _singleton(run: _Run, name: str) -> OutputMember:
    found = run.runtime.published_output(run.workflow_id, name, None, None, None)
    assert found is not None and found.member is not None
    assert found.member.publication is not None
    return found.member


def _members(run: _Run, name: str) -> dict[str, OutputMember]:
    listed = run.runtime.published_outputs(run.workflow_id, name)
    assert listed is not None
    return {member.key or "": member for member in listed.members}


_LOOP_BODY = """
    templates:
      - name: body
        inputs: [{name: state, role: carried}]
        nodes:
          - name: step
            dependsOn: [{node: $ingress, port: state, input: state}]
            spec: {taskType: echo, data: {type: list, items: ["${state.draft}"]}}
          - name: route
            dependsOn: [{node: step, input: input}]
            region:
              kind: branch
              inputs: [{name: input}]
              outputs: [{name: again}, {name: done}]
              selection: {input: input, field: [route]}
        edges:
          - from: {node: route, port: again}
            to: {node: $feedback, port: state}
            project: [next]
          - from: {node: route, port: done}
            to: {node: $egress, port: state}
            project: [next]
"""

_LOOP = """
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: refine
        dependsOn: [{node: seed, input: state}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
          result: {visibility: published}
"""


@pytest.mark.anyio
async def test_a_published_loop_serves_its_projected_exit_value() -> None:
    run = await _Run().start(_workflow(_LOOP, _LOOP_BODY))
    run.run("seed", {"draft": "a"})
    run.run("step", {"route": "again", "next": {"draft": "b"}})
    run.run("step", {"route": "done", "next": {"draft": "z"}, "noise": 1})

    assert _served(run, _singleton(run, "refine")) == {"draft": "z"}
    fetched = await outputs_router.get_output(
        run.workflow_id,
        "refine",
        scope=None,
        key=None,
        sequence=None,
        principal=PrincipalContext(
            principal_id="p-1",
            org_id="org",
            external_id="ext",
            principal_type="user",
            scopes=[],
        ),
        runtime=run.runtime,
        logger=logging.getLogger("test.region_outputs"),
    )
    client = SdkWorkflowOutputValue.model_validate(fetched.model_dump(mode="json"))
    assert isinstance(client.value, SdkRoutedValue)
    assert client.value.routed_value == {"draft": "z"}


_CONCAT = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: both
        dependsOn: [{node: a, input: l, project: [inner]}, {node: b, input: r}]
        region: {kind: merge, combination: concat, result: {visibility: published}}
"""


@pytest.mark.anyio
async def test_a_published_concat_merge_serves_each_members_projected_value() -> None:
    run = await _Run().start(_workflow(_CONCAT))
    run.run("a", {"inner": {"v": 1}, "outer": 2})
    run.run("b", {"v": 3})

    members = _members(run, "both")
    assert _served(run, members["l"]) == {"v": 1}
    whole = run.runtime.read_output(members["r"])
    assert whole.model_dump()["v"] == 3


_TWO_RETURNS = """
    templates:
      - name: child
        inputs: [{name: item, role: param}]
        returns: [{name: out}, {name: extra}]
        nodes:
          - name: work
            dependsOn: [{node: $ingress, port: item, input: item}]
            spec: {taskType: echo, data: {type: list, items: ["${item}"]}}
        edges:
          - from: {node: work}
            to: {node: $return, port: out}
          - from: {node: work}
            to: {node: $return, port: extra}
            project: [value]
"""

_FAN_OVER_CHILD = """
      - name: plan
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: fan
        dependsOn: [plan]
        region: {kind: spawn, child: child}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled, result: {visibility: published}}
"""


@pytest.mark.anyio
async def test_a_published_join_serves_each_childs_returns_by_port() -> None:
    run = await _Run().start(_workflow(_FAN_OVER_CHILD, _TWO_RETURNS))
    run.run("plan", {"items": ["x", "y"]})
    run.run("work", {"value": "one"})
    run.run("work", {"value": "two"})

    members = _members(run, "collect")
    assert _served(run, members["0"]) == {
        "out": {"ok": True, "value": "one"},
        "extra": "one",
    }
    assert _served(run, members["1"])["extra"] == "two"


_WORK_OR_FAN = """
      - name: classify
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: route
        dependsOn: [{node: classify, input: input}]
        region:
          kind: branch
          inputs: [{name: input}]
          outputs: [{name: work}, {name: fan}]
          selection: {input: input, field: [label]}
      - name: simple
        dependsOn: [{node: route, port: work, input: in}]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: spawn
        dependsOn: [{node: route, port: fan}]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [spawn]
        region: {kind: join, completion: all_settled}
      - name: either
        dependsOn: [{node: simple, input: w}, {node: collect, input: s}]
        region: {kind: merge, combination: one_live, result: {visibility: published}}
"""


@pytest.mark.anyio
async def test_a_published_one_live_merge_of_a_join_serves_its_members() -> None:
    run = await _Run().start(_workflow(_WORK_OR_FAN))
    run.run("classify", {"label": "fan", "items": ["p", "q"]})
    run.run("kid", {"value": "one"})
    run.run("kid", {"value": "two"})

    assert _served(run, _singleton(run, "either")) == [
        {"key": "0", "outcome": "success", "value": {"ok": True, "value": "one"}},
        {"key": "1", "outcome": "success", "value": {"ok": True, "value": "two"}},
    ]


_HIT_OR_MISS = """
      - name: classify
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: route
        dependsOn: [{node: classify, input: input}]
        region:
          kind: branch
          inputs: [{name: input}]
          outputs: [{name: hit}, {name: miss}]
          selection: {input: input, field: [label]}
      - name: on_hit
        dependsOn: [{node: route, port: hit}]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: on_miss
        dependsOn: [{node: route, port: miss}]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: either
        dependsOn: [{node: on_hit, input: h}, {node: on_miss, input: m}]
        region: {kind: merge, combination: one_live, result: {visibility: published}}
"""


@pytest.mark.anyio
async def test_a_one_live_merge_of_a_skipped_arm_publishes_an_explicit_empty() -> None:
    run = await _Run().start(_workflow(_HIT_OR_MISS))
    run.run("classify", {"label": "hit"})
    task_id = next(t for t in run.ready if run.name(t) == "on_hit")
    record_dispatch(run.runtime, task_id, _worker())
    run.runtime.mark_succeeded(
        task_id,
        worker_id=None,
        payload={"finished_at": _TS, "started_at": _TS},
        ts=_TS,
        skip={"skipped": True, "reason": "condition_not_met"},
    )
    run.drive()

    publication = _singleton(run, "either").publication
    assert publication is not None
    assert publication.outcome is PublicationOutcome.EXPLICIT_EMPTY


_EARLY = """
      - name: plan
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: fan
        dependsOn: [plan]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region:
          kind: join
          completion: any
          residual: cancel
          result: {visibility: published}
"""


@pytest.mark.anyio
async def test_a_published_early_join_lists_the_members_it_released_with() -> None:
    run = await _Run().start(_workflow(_EARLY))
    run.run("plan", {"items": ["x", "y"]})
    run.run("kid", {"value": "first"})

    members = _members(run, "collect")
    assert list(members) == ["0"]
    assert run.runtime.read_output(members["0"]).model_dump()["value"] == "first"


def _fail_children(run: _Run) -> None:
    for task_id in list(run.ready):
        run.ready.remove(task_id)
        record_dispatch(run.runtime, task_id, _worker())
        run.runtime.fail_dispatch(
            task_id, "wkr-1", {}, _TS, error="boom", retryable=False
        )
        run.drive()


def _join(completion: str) -> str:
    return _EARLY.replace(
        "completion: any\n          residual: cancel", f"completion: {completion}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("children", [[], ["x", "y"]])
async def test_an_early_join_with_no_winner_publishes_one_empty_member(
    children: list[str],
) -> None:
    run = await _Run().start(_workflow(_EARLY))
    run.run("plan", {"items": children})
    _fail_children(run)

    assert run.settled()
    publication = _singleton(run, "collect").publication
    assert publication is not None
    assert publication.outcome is PublicationOutcome.EXPLICIT_EMPTY


@pytest.mark.anyio
@pytest.mark.parametrize(
    "completion",
    [
        "any\n          residual: cancel\n          no_winner_failure: true",
        "all_succeed",
    ],
)
async def test_a_join_resolving_a_failure_publishes_one_failed_member(
    completion: str,
) -> None:
    run = await _Run().start(_workflow(_join(completion)))
    run.run("plan", {"items": ["x", "y"]})
    _fail_children(run)

    assert run.settled()
    assert {
        member.key: member.publication.outcome
        for member in _members(run, "collect").values()
        if member.publication is not None
    } == {None: PublicationOutcome.DECLARED_FAILURE}


_PENDING_OUTPUTS = """
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: refine
        dependsOn: [{node: seed, input: state}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
          result: {visibility: published}
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [seed]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: either
        dependsOn: [{node: a, input: l}, {node: b, input: r}]
        region: {kind: merge, combination: concat, result: {visibility: published}}
      - name: plan
        dependsOn: [seed]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: fan
        dependsOn: [plan]
        region: {kind: spawn, child: kid}
      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled, result: {visibility: published}}
"""


@pytest.mark.anyio
async def test_a_cancel_publishes_every_pending_output_empty() -> None:
    run = await _Run().start(_workflow(_PENDING_OUTPUTS, _LOOP_BODY))
    run.run("a", {"v": 1})

    run.runtime.cancel_workflow(run.workflow_id)

    listed = run.runtime.published_outputs(run.workflow_id)
    assert listed is not None
    assert {
        member.name: member.publication.outcome
        for member in listed.members
        if member.publication is not None
    } == {
        "refine": PublicationOutcome.EXPLICIT_EMPTY,
        "either": PublicationOutcome.EXPLICIT_EMPTY,
        "collect": PublicationOutcome.EXPLICIT_EMPTY,
    }
    assert all(member.publication is not None for member in listed.members)
