import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _live_runtime,
    _register,
)

_HEAD = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: t}
spec:
  graph:
    nodes:
"""


def _compile(nodes: str) -> None:
    text = _HEAD + nodes
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    compile_workflow("wfl-test", parsed, source)


def _reject(nodes: str) -> CompileError:
    with pytest.raises(CompileError) as exc:
        _compile(nodes)
    return exc.value


def test_recompute_over_live_read_rejected() -> None:
    err = _reject("""      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
          v2: {recovery: recompute}
""")
    assert any(d.code == "recovery.illegal-recompute" for d in err.diagnostics)


def test_bare_live_read_is_legal() -> None:
    # An unpinned live read is latitude, not an error (design 21 s5.5 cl.2).
    _compile("""      - name: caller
        spec:
          taskType: api
          api: {url: 'http://x', method: GET}
""")


def test_delegate_exceeds_invoke_rejected() -> None:
    err = _reject("""      - name: a
        spec:
          taskType: agent
          task: hi
          v2:
            authority: {invoke: [], delegate: [web_search]}
            tools: [{name: web_search}]
""")
    assert any(d.code == "authority.delegate-exceeds-invoke" for d in err.diagnostics)


def test_invoke_undeclared_tool_rejected() -> None:
    err = _reject("""      - name: a
        spec:
          taskType: agent
          task: hi
          v2: {authority: {invoke: [ghost]}}
""")
    assert any(d.code == "authority.undeclared-tool" for d in err.diagnostics)


def test_spawn_site_authority_rejected() -> None:
    err = _reject("""      - name: s
        region: {kind: spawn, child: c, authority: {invoke: [], delegate: [x]}}
""")
    assert any(d.code == "authority.delegate-exceeds-invoke" for d in err.diagnostics)


def test_spawn_child_unresolved_rejected() -> None:
    err = _reject("""      - name: s
        region: {kind: spawn, child: ghost, authority: {invoke: []}}
""")
    assert any(d.code == "region.spawn-child-unresolved" for d in err.diagnostics)


def test_spawn_child_non_dispatchable_rejected() -> None:
    err = _reject("""      - name: inner
        region: {kind: merge}
      - name: s
        region: {kind: spawn, child: inner, authority: {invoke: []}}
""")
    assert any(d.code == "region.spawn-child-not-dispatchable" for d in err.diagnostics)


def test_authority_on_non_agent_leaf_rejected() -> None:
    err = _reject("""      - name: e
        spec:
          taskType: echo
          data: {type: list, items: [x]}
          v2: {authority: {invoke: [t]}}
""")
    assert any(d.code == "v2.authority-on-leaf" for d in err.diagnostics)


def test_early_join_without_residual_rejected() -> None:
    err = _reject("""      - name: j
        region: {kind: join, completion: any}
""")
    assert any(d.code == "region.join-no-residual" for d in err.diagnostics)


def test_first_k_join_without_k_rejected() -> None:
    err = _reject("""      - name: j
        region: {kind: join, completion: first_k, residual: continue}
""")
    assert any(d.code == "region.join-bad-k" for d in err.diagnostics)


def test_first_k_join_negative_k_rejected() -> None:
    err = _reject("""      - name: j
        region: {kind: join, completion: first_k, residual: continue, k: -1}
""")
    assert any(d.code == "region.join-bad-k" for d in err.diagnostics)


def test_predicate_join_non_positive_threshold_rejected() -> None:
    err = _reject("""      - name: j
        region:
          kind: join
          completion: predicate
          residual: continue
          predicate: {min_qualifiers: 0}
""")
    assert any(d.code == "region.join-bad-predicate" for d in err.diagnostics)


def test_unknown_region_kind_rejected() -> None:
    err = _reject("""      - name: r
        region: {kind: frobnicate}
""")
    assert [d.code for d in err.diagnostics] == ["region.unknown-kind"]


@pytest.mark.parametrize(
    "region",
    ['{kind: branch, selection: "x", ports: [p]}', "{kind: loop, coordinate: t}"],
)
def test_a_branch_or_loop_with_unknown_fields_is_rejected(
    region: str,
) -> None:
    err = _reject(f"""      - name: r
        region: {region}
""")
    assert [d.code for d in err.diagnostics] == ["region.unknown-field"]


def test_a_feedback_key_is_invalid_input() -> None:
    with pytest.raises(ValueError, match="Invalid workflow payload"):
        _compile("""      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
        feedback: {to: a, port: p}
""")


def test_unstructured_cycle_rejected() -> None:
    # A dependsOn cycle is rejected by the parser before compilation.
    with pytest.raises(ValueError):
        _compile("""      - name: a
        dependsOn: [b]
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
""")


_SPAWNED_WORKER = """      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: worker
        spec:
          taskType: agent
          v2:
            inputs: [facet]
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
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
"""

_REGION_CONSUMER = """      - name: merge
        spec:
          taskType: agent
          v2:
            inputs: [{name: subs, from: %s, region: sub}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def test_a_region_input_from_a_spawned_agent_is_refused() -> None:
    err = _reject(_SPAWNED_WORKER + _REGION_CONSUMER % "worker")
    assert [d.code for d in err.diagnostics] == ["dataflow.spawned-region-output"]


def test_a_region_input_from_a_root_agent_compiles() -> None:
    _compile("""      - name: lead
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: sub, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: sub
        spec: {taskType: echo, data: {type: list, items: [k]}}
""" + _REGION_CONSUMER % "lead")


_SELF_READER = """      - name: lead
        spec:
          taskType: agent
          v2:
            inputs: [{name: mine, from: lead, region: sub}]
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: sub, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: sub
        spec: {taskType: echo, data: {type: list, items: [k]}}
"""

_UPSTREAM_READER = """      - name: lead
        spec:
          taskType: agent
          v2:
            inputs: [{name: theirs, from: helper, region: sub}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: helper
        dependsOn: [lead]
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: [model]}
            tools: [{name: model}]
            child: [{name: sub, authority: {invoke: [model], delegate: []}}]
          harness: {backend: scripted, version: v1, params: {script: []}}
      - name: sub
        spec: {taskType: echo, data: {type: list, items: [k]}}
"""


@pytest.mark.parametrize(
    "body", [_SELF_READER, _UPSTREAM_READER], ids=["own", "dependent"]
)
def test_an_agent_reading_a_region_that_waits_on_it_is_refused(body: str) -> None:
    err = _reject(body)
    assert [d.code for d in err.diagnostics] == ["topology.unstructured-cycle"]


_FAN = """      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: fan
        dependsOn: [planner]
        region: {kind: spawn, child: kid}
"""

SPAWN_DEPENDENTS = {
    "task": """      - name: after
        dependsOn: [fan]
        spec: {taskType: echo, data: {type: list, items: [z]}}
""",
    "region": """      - name: m
        dependsOn: [fan]
        region: {kind: merge}
""",
    "agent_input": """      - name: reader
        spec:
          taskType: agent
          v2:
            inputs: [{name: kids, from: fan}]
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
""",
}


@pytest.mark.parametrize("dependent", SPAWN_DEPENDENTS.values(), ids=SPAWN_DEPENDENTS)
def test_a_node_depending_on_a_spawn_is_refused(dependent: str) -> None:
    err = _reject(_FAN + dependent)
    assert "dataflow.spawn-dependent" in {d.code for d in err.diagnostics}


def test_a_node_depending_on_a_spawns_join_compiles() -> None:
    _compile(_FAN + """      - name: collect
        dependsOn: [fan]
        region: {kind: join, completion: all_settled}
      - name: after
        dependsOn: [collect]
        spec: {taskType: echo, data: {type: list, items: [z]}}
""")


@pytest.mark.anyio
async def test_a_node_depending_on_a_spawn_is_refused_at_submit() -> None:
    runtime = _live_runtime(FakeRegistry())
    with pytest.raises(CompileError) as exc:
        await _register(runtime, _HEAD + _FAN + SPAWN_DEPENDENTS["task"])
    assert "dataflow.spawn-dependent" in {d.code for d in exc.value.diagnostics}


_CALL = """      - name: planner
        spec: {taskType: echo, data: {type: list, items: [seed]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: c
        dependsOn: [planner]
        region: {kind: call, child: kid}
"""

REGION_FED_REGIONS = {
    "call_to_spawn": "{kind: spawn, child: kid}",
    "call_to_call": "{kind: call, child: kid}",
    "call_to_join": "{kind: join, completion: all_settled}",
}


@pytest.mark.parametrize("region", REGION_FED_REGIONS.values(), ids=REGION_FED_REGIONS)
def test_a_region_fed_by_a_call_is_refused(region: str) -> None:
    err = _reject(_CALL + f"""      - name: r
        dependsOn: [c]
        region: {region}
""")
    assert "dataflow.region-input" in {d.code for d in err.diagnostics}


def test_a_spawn_fed_by_a_merge_is_refused() -> None:
    err = _reject("""      - name: a
        spec: {taskType: echo, data: {type: list, items: [a]}}
      - name: kid
        spec: {taskType: echo, data: {type: list, items: [k]}}
      - name: m
        dependsOn: [a]
        region: {kind: merge}
      - name: fan
        dependsOn: [m]
        region: {kind: spawn, child: kid}
""")
    assert "dataflow.region-input" in {d.code for d in err.diagnostics}


def test_a_join_with_a_task_input_beside_its_spawn_compiles() -> None:
    _compile(_FAN + """      - name: x
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: collect
        dependsOn: [fan, x]
        region: {kind: join, completion: all_settled}
""")
