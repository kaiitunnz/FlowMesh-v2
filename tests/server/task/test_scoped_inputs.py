"""A task inside a region definition reads every input as a value, by one spelling.

Each case runs a workflow on a real runtime, renders a ready member's placeholders the
way its dispatch does, and hydrates its upstream values the way its worker does, both
over the one content store the runtime's results live in.
"""

import logging
from typing import Any, cast

import pytest

from server.dispatcher.base import Dispatcher
from server.registries.worker import WorkerRegistry
from server.task.runtime import ScopedInput
from server.task.v2.compiler.diagnostics import CompileError
from shared.schemas.result import RoutedValue
from shared.tasks.worker_message import WorkerTaskMessage
from shared.utils.json import normalize_numbers
from tests.server.orchestration.control_flow import compile_text
from tests.server.task.test_runtime_control_flow import _TS, _Run, _workflow
from tests.worker.factories import FakeContentPlane
from worker.content.inputs import TaskInputHydrator
from worker.executors.utils.expressions import project_expression

_ECHO = "{taskType: echo, data: {type: list, items: [x]}}"


def _dispatch(run: _Run, name: str) -> tuple[dict[str, Any], WorkerTaskMessage]:
    """The spec a ready member of ``name`` renders to, and its message as its worker
    hydrates it."""
    task_id = next(t for t in run.ready if run.name(t) == name)
    record = run.runtime.get_record(task_id)
    assert record is not None
    dispatcher = Dispatcher(
        runtime=run.runtime,
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("scoped-inputs"),
    )
    rendered, upstream = dispatcher._resolve_stage_references(
        task_id, record.task, {}, run.runtime.scoped_inputs(task_id)
    )
    message = WorkerTaskMessage(
        task_id=task_id,
        workflow_id=record.workflow_id,
        owner_id=record.owner_id,
        content_scope=record.org_id,
        task=rendered,
        assigned_worker="wkr-1",
        dispatched_at=_TS,
        upstream_results=upstream,
    )
    wire = message.model_dump(mode="json", exclude_none=True, by_alias=True)
    delivered = WorkerTaskMessage.model_validate(normalize_numbers(wire))
    plane = FakeContentPlane(run.reader.store)
    TaskInputHydrator(cast(Any, plane), backoff_sec=0.0).hydrate(delivered)
    spec = rendered.spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    return spec, delivered


def _worker_reads(message: WorkerTaskMessage, expr: str) -> Any:
    return project_expression(expr, message.task.spec.upstreamResults or {})


_DRAFT = """{taskType: echo, data: {type: list, items: ["${state.draft}"]}}"""

_REFINE_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_DRAFT}
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
            project: [next]
          - from: {{node: route, port: done}}
            to: {{node: $egress, port: state}}
"""

_REFINE = """
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: refine
        dependsOn: [{node: seed, input: state}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}]
"""


@pytest.mark.anyio
async def test_a_carried_value_reads_by_one_spelling_whole_or_projected() -> None:
    run = await _Run().start(_workflow(_REFINE, _REFINE_BODY))
    run.run("seed", {"draft": "a"})

    # Time 0 reads the seed's whole result.
    spec, message = _dispatch(run, "step")
    assert spec["data"]["items"] == ["a"]
    assert _worker_reads(message, "state.draft") == "a"
    run.run("step", {"route": "again", "next": {"draft": "b"}})

    # Time 1 reads the part of the previous step its feedback projects.
    spec, message = _dispatch(run, "step")
    assert spec["data"]["items"] == ["b"]
    assert _worker_reads(message, "state.draft") == "b"
    assert isinstance((message.task.spec.upstreamResults or {})["state"], RoutedValue)


@pytest.mark.anyio
async def test_a_routed_value_entitles_its_reader_to_the_object_it_reads() -> None:
    run = await _Run().start(_workflow(_REFINE, _REFINE_BODY))
    run.run("seed", {"draft": "a"})
    step = run.run("step", {"route": "again", "next": {"draft": "b"}})
    _, message = _dispatch(run, "step")
    binding = (message.upstream_results or {})["state"]
    assert binding.path == ("next",) and binding.task_id == step
    assert binding.reference is not None

    record = run.runtime.get_record(message.task_id)
    assert record is not None
    with run.runtime._lock:
        assert run.runtime._content_bindings.consumes_locked(record, binding.reference)


_COUNTER_BODY = """
    templates:
      - name: body
        inputs: [{name: state, role: carried}]
        nodes:
          - name: step
            dependsOn: [{node: $ingress, port: state, input: state}]
            spec: {taskType: echo, data: {type: list, items: ["round ${state}"]}}
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
            project: [count]
          - from: {node: route, port: done}
            to: {node: $egress, port: state}
"""


@pytest.mark.anyio
async def test_a_bare_reference_reads_a_scalar_value_whole() -> None:
    run = await _Run().start(_workflow(_REFINE, _COUNTER_BODY))
    run.run("seed", {"draft": "a"})
    run.run("step", {"route": "again", "count": 1})

    spec, message = _dispatch(run, "step")
    assert spec["data"]["items"] == ["round 1"]
    assert _worker_reads(message, "state") == 1


_OUTCOME = """{taskType: echo, data: {type: list, items: ["${members.1.outcome}"]}}"""

_TALLY_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: kid
            spec: {_ECHO}
          - name: fan
            dependsOn: [step]
            region: {{kind: spawn, child: kid}}
          - name: collect
            dependsOn: [fan]
            region: {{kind: join, completion: all_settled}}
          - name: tally
            dependsOn: [{{node: collect, input: members}}]
            spec: {_OUTCOME}
          - name: route
            dependsOn: [{{node: tally, input: input}}]
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


@pytest.mark.anyio
async def test_a_member_reads_an_aggregate_with_each_outcome() -> None:
    run = await _Run().start(_workflow(_REFINE, _TALLY_BODY))
    run.run("seed", {"draft": "a"})
    run.run("step", {"items": ["x", "y"]})
    first, second = (t for t in run.ready if run.name(t) == "kid")
    run.run("kid", {"value": "ok"})
    assert run.ready.count(second) == 1 and first not in run.ready
    run.ready.remove(second)
    run.runtime.mark_failed(second, "wkr-1", {}, _TS, error="boom")
    run.drive()

    spec, message = _dispatch(run, "tally")
    assert spec["data"]["items"] == ["declared_failure"]
    members = _worker_reads(message, "members")
    assert [(m["key"], m["outcome"]) for m in members] == [
        ("0", "success"),
        ("1", "declared_failure"),
    ]
    assert members[0]["value"]["value"] == "ok"
    assert members[1]["value"] is None


_SIBLING_BODY = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: check
            dependsOn: [step]
            spec: {{taskType: echo, data: {{type: list, items: ["${{step.task_id}}"]}}}}
          - name: route
            dependsOn: [{{node: check, input: input}}]
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


@pytest.mark.anyio
async def test_a_task_id_read_names_the_sibling_in_the_readers_time() -> None:
    run = await _Run().start(_workflow(_REFINE, _SIBLING_BODY))
    run.run("seed", {"draft": "a"})
    first = run.run("step")
    spec, _ = _dispatch(run, "check")
    assert spec["data"]["items"] == [first]
    run.run("check", {"route": "again"})

    second = run.run("step")
    assert second != first
    spec, _ = _dispatch(run, "check")
    assert spec["data"]["items"] == [second]


@pytest.mark.parametrize(
    "read",
    [
        "${state.task_id}",
        "${input.task_id}",
        "${collect.task_id}",
    ],
)
def test_a_task_id_read_of_a_value_is_refused(read: str) -> None:
    body = f"""
    templates:
      - name: body
        inputs: [{{name: state, role: carried}}]
        nodes:
          - name: step
            dependsOn: [{{node: $ingress, port: state, input: state}}]
            spec: {_ECHO}
          - name: kid
            spec: {_ECHO}
          - name: fan
            dependsOn: [step]
            region: {{kind: spawn, child: kid}}
          - name: collect
            dependsOn: [fan]
            region: {{kind: join, completion: all_settled}}
          - name: reader
            dependsOn:
              - {{node: $ingress, port: state, input: state}}
              - {{node: step, input: input, project: [items]}}
              - collect
            spec: {{taskType: echo, data: {{type: list, items: ["{read}"]}}}}
        edges:
          - from: {{node: reader}}
            to: {{node: $egress, port: state}}
"""
    with pytest.raises(CompileError) as raised:
        compile_text(_workflow(_REFINE, body))
    assert "reads.task-id-of-value" in str(raised.value)


def test_a_bare_reference_stays_malformed_at_the_root() -> None:
    nodes = f"""
      - name: seed
        spec: {_ECHO}
      - name: after
        dependsOn: [seed]
        spec: {{taskType: echo, data: {{type: list, items: ["${{seed}}"]}}}}
"""
    with pytest.raises(CompileError) as raised:
        compile_text(_workflow(nodes))
    assert "reads.unresolved" in str(raised.value)


_ROOT_LOOP = _REFINE + """      - name: consume
        dependsOn: [{node: refine, port: state, input: final}]
        spec:
          taskType: echo
          data: {type: list, items: ["${final.draft}", "${refine.draft}"]}
"""


async def _exited(run: _Run) -> _Run:
    run.run("seed", {"draft": "a"})
    run.run("step", {"route": "again", "next": {"draft": "b"}})
    run.run("step", {"route": "done", "draft": "z"})
    return run


@pytest.mark.anyio
async def test_a_root_task_reads_a_loops_exit_value_by_input_and_by_node() -> None:
    run = await _exited(await _Run().start(_workflow(_ROOT_LOOP, _REFINE_BODY)))
    spec, message = _dispatch(run, "consume")
    assert spec["data"]["items"] == ["z", "z"]
    assert _worker_reads(message, "final.draft") == "z"
    assert _worker_reads(message, "refine.draft") == "z"


@pytest.mark.anyio
async def test_a_root_task_reads_a_loops_exit_value_after_a_restart() -> None:
    run = await _exited(await _Run().start(_workflow(_ROOT_LOOP, _REFINE_BODY)))
    restored = await run.restart()
    spec, _ = _dispatch(restored, "consume")
    assert spec["data"]["items"] == ["z", "z"]


def test_a_root_read_of_a_loop_with_several_values_names_its_port() -> None:
    body = _REFINE_BODY.replace(
        "inputs: [{name: state, role: carried}]",
        "inputs: [{name: state, role: carried}, {name: other, role: carried}]",
    )
    nodes = """
      - name: seed
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: refine
        dependsOn: [{node: seed, input: state}, {node: seed, input: other}]
        region:
          kind: loop
          body_ref: body
          loop_coordinate: round
          carried: [{name: state}, {name: other}]
      - name: consume
        dependsOn: [refine]
        spec: {taskType: echo, data: {type: list, items: ["${refine.draft}"]}}
"""
    with pytest.raises(CompileError) as raised:
        compile_text(_workflow(nodes, body))
    assert "ports.ambiguous-output" in str(raised.value)


_ROOT_BRANCH = """
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
        dependsOn: [{node: route, port: hit, input: picked}]
        spec: {taskType: echo, data: {type: list, items: ["${picked.label}"]}}
      - name: on_miss
        dependsOn: [{node: route, port: miss, input: picked}]
        spec: {taskType: echo, data: {type: list, items: ["${picked.label}"]}}
      - name: either
        dependsOn: [{node: route, port: hit}, {node: route, port: miss}]
        region: {kind: merge, combination: one_live}
      - name: joined
        dependsOn: [{node: either, input: chosen}]
        spec: {taskType: echo, data: {type: list, items: ["${chosen.label}"]}}
"""


@pytest.mark.anyio
async def test_a_root_task_reads_the_branch_arm_and_merge_it_takes() -> None:
    run = await _Run().start(_workflow(_ROOT_BRANCH))
    run.run("classify", {"label": "hit"})
    assert sorted(run.name(t) for t in run.ready) == ["joined", "on_hit"]
    spec, message = _dispatch(run, "on_hit")
    assert spec["data"]["items"] == ["hit"]
    assert _worker_reads(message, "picked.label") == "hit"
    spec, _ = _dispatch(run, "joined")
    assert spec["data"]["items"] == ["hit"]
    record = run.runtime.get_record(run.ids["on_miss"])
    assert record is not None and record.result_skip is not None


_ROOT_JOIN = f"""
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
      - name: tally
        dependsOn: [{{node: collect, input: members}}]
        spec: {_OUTCOME}
"""


@pytest.mark.anyio
async def test_a_root_task_reads_a_joins_members_with_each_outcome() -> None:
    run = await _Run().start(_workflow(_ROOT_JOIN))
    run.run("plan", {"items": ["x", "y"]})
    first, second = (t for t in run.ready if run.name(t) == "kid")
    run.run("kid", {"value": "ok"})
    run.ready.remove(second)
    run.runtime.mark_failed(second, "wkr-1", {}, _TS, error="boom")
    run.drive()

    spec, message = _dispatch(run, "tally")
    assert spec["data"]["items"] == ["declared_failure"]
    members = _worker_reads(message, "members")
    assert [(m["outcome"], m["value"]) for m in members] == [
        ("success", {"ok": True, "value": "ok"}),
        ("declared_failure", None),
    ]
    assert first not in run.ready


_ROOT_AGENT = _REFINE + """      - name: reader
        dependsOn: [{node: refine, port: state, input: final}]
        spec:
          taskType: agent
          task: read the final draft
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
            boundary: [invocation, yield]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


@pytest.mark.anyio
async def test_a_root_agent_accepts_a_loops_exit_value() -> None:
    run = await _Run().start(_workflow(_ROOT_AGENT, _REFINE_BODY))
    run.run("seed", {"draft": "a"})
    run.run("step", {"route": "again", "next": {"draft": "b"}})
    last = run.run("step", {"route": "done", "draft": "z"})
    assert [run.name(t) for t in run.ready] == ["reader"]
    (accepted,) = run.engine.accepted_inputs_for_task(run.ids["reader"])
    assert accepted.target_port == "final"
    (member,) = accepted.members
    assert member.value_ref is not None
    assert member.value_ref.legacy_task_id == last


_TASKS_ONLY = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: c
        dependsOn: [b]
        spec:
          taskType: echo
          data:
            type: list
            items: ["${a.x}", "${b.y.0}", "${b.task_id}", "${a.task_id}"]
"""


@pytest.mark.anyio
async def test_a_root_task_fed_by_tasks_reads_the_same_through_records_or_edges() -> (
    None
):
    run = await _Run().start(_workflow(_TASKS_ONLY))
    run.run("a", {"x": "from-a"})
    run.run("b", {"y": ["from-b"]})
    task_id = run.ids["c"]
    record = run.runtime.get_record(task_id)
    assert record is not None
    assert run.runtime.scoped_inputs(task_id) is None
    with run.runtime._lock:
        through_edges = {
            entry.name: ScopedInput(
                run.runtime._content_bindings._value_binding_locked(entry.value),
                entry.task_id,
            )
            for entry in run.engine.edge_inputs(task_id)
        }
    dispatcher = Dispatcher(
        runtime=run.runtime,
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("scoped-inputs"),
    )
    by_records, record_bindings = dispatcher._resolve_stage_references(
        task_id, record.task, dispatcher._build_stage_context(record)
    )
    by_edges, edge_bindings = dispatcher._resolve_stage_references(
        task_id, record.task, {}, through_edges
    )
    assert by_edges == by_records
    assert by_records.spec.model_dump()["data"]["items"] == [
        "from-a",
        "from-b",
        run.ids["b"],
        run.ids["a"],
    ]
    assert record_bindings is not None and edge_bindings is not None
    assert edge_bindings == record_bindings


_NAMED = """
      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [{node: a, input: src, project: [inner]}]
        spec: {taskType: echo, data: {type: list, items: ["${src.y}", "${a.inner.y}"]}}
"""


@pytest.mark.anyio
async def test_a_root_task_reads_a_named_input_of_a_task() -> None:
    run = await _Run().start(_workflow(_NAMED))
    run.run("a", {"inner": {"y": "v"}})
    task_id = run.ids["b"]
    record = run.runtime.get_record(task_id)
    assert record is not None
    dispatcher = Dispatcher(
        runtime=run.runtime,
        worker_registry=cast(WorkerRegistry, object()),
        logger=logging.getLogger("scoped-inputs"),
    )
    # As the dispatcher reads it: a root task fed only by unnamed task inputs reads
    # its records, and any other reads its edges.
    rendered, upstream = dispatcher._resolve_stage_references(
        task_id,
        record.task,
        dispatcher._build_stage_context(record),
        run.runtime.scoped_inputs(task_id),
    )
    assert rendered.spec.model_dump()["data"]["items"] == ["v", "v"]
    assert upstream is not None and upstream["src"].path == ("inner",)
