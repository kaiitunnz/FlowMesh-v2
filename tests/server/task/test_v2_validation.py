import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow

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
          configName: default
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
          configName: default
          task: hi
          v2: {authority: {invoke: [ghost]}}
""")
    assert any(d.code == "authority.undeclared-tool" for d in err.diagnostics)


def test_spawn_site_authority_rejected() -> None:
    err = _reject("""      - name: s
        region: {kind: spawn, child: c, authority: {invoke: [], delegate: [x]}}
""")
    assert any(d.code == "authority.delegate-exceeds-invoke" for d in err.diagnostics)


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


def test_unknown_region_kind_rejected() -> None:
    err = _reject("""      - name: r
        region: {kind: frobnicate}
""")
    assert any(d.code == "region.unknown-kind" for d in err.diagnostics)


def test_feedback_to_non_loop_rejected() -> None:
    err = _reject("""      - name: a
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: b
        dependsOn: [a]
        spec: {taskType: echo, data: {type: list, items: [y]}}
        feedback: {to: a, port: p}
""")
    assert any(d.code == "feedback.not-loop" for d in err.diagnostics)


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
