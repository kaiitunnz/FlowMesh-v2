"""A v2 agent declares its spawn child, authority, and harness from the frontend.

An ``agent`` node carrying ``agent.child`` and ``agent.authority`` compiles to an
AgentOperator with one declared child region and the declared invoke/delegate faces; the
engine built from it grants those faces so a declared boundary is admitted rather than
denied.
"""

from server.orchestration import OrchestrationEngine
from server.task.parser import parse_workflow
from server.task.v2 import FrontendWorkflowSource, compile_bundle
from server.task.v2.representations.operators import AgentOperator
from shared.harness.boundary import BoundaryEventKind

_AGENT_WORKFLOW = """
apiVersion: flowmesh/v2
kind: Workflow
metadata:
  name: agent-episode
spec:
  taskType: echo
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority:
              invoke: [model]
              delegate: [model]
            tools:
              - {name: model}
            child: reviewer
          harness:
            backend: scripted
            version: v1
            params:
              script:
                - {op: boundary, kind: invocation, call: c0, interface: model}
                - {op: boundary, kind: spawn, call: c1, region: reviewer}
                - {op: boundary, kind: spawn_seal, call: c2, region: reviewer}
                - {op: complete, value_from: c0}
      - name: reviewer
        spec:
          taskType: echo
          data:
            type: list
            items: [placeholder]
"""


def _bundle():
    parsed = parse_workflow(_AGENT_WORKFLOW, "native")
    source = FrontendWorkflowSource.capture(_AGENT_WORKFLOW, "native", name="wf")
    return compile_bundle("wfl-agent", parsed, source)


def _agent(bundle) -> AgentOperator:
    agent = next(
        op for op in bundle.template.operators if isinstance(op, AgentOperator)
    )
    return agent


def test_agent_declares_one_child_region_from_the_frontend() -> None:
    agent = _agent(_bundle())
    assert len(agent.child_region_refs) == 1
    ref = agent.child_region_refs[0]
    assert ref.name == "reviewer"
    assert agent.child_template_ref is None  # normalized into the region


def test_agent_authority_and_seal_boundary_are_declared() -> None:
    agent = _agent(_bundle())
    assert "model" in agent.authority.invoke and "model" in agent.authority.delegate
    assert BoundaryEventKind.SPAWN_SEAL in agent.boundary.events


def test_engine_grants_the_declared_agent_interface() -> None:
    bundle = _bundle()
    eng = OrchestrationEngine.build("wfl-agent", "owner", "org", bundle)
    # The declared interface is granted, so it survives at the root invoke face.
    assert "model" in eng._root_grant.invoke  # type: ignore[attr-defined]
