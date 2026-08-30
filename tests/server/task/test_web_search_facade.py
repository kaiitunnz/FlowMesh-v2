"""Compiling an agent pins its web_search facade and keeps child authority fail-closed.

An agent that declares the ``search/v1`` interface gets a pinned web_search facade the
gateway may inject; a spawnable agent also gets ``spawn_agent``. A child region
carries an explicit ceiling, and a child agent that declares an interface the region
omits is a compile error.
"""

import json
from typing import Any

import pytest

from server.task.parser import parse_workflow
from server.task.v2 import CompileError, FrontendWorkflowSource, compile_workflow
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.representations.operators import AgentOperator
from shared.harness import BoundaryEventKind

_BINDINGS = AgentBindingDefaults(default_backend="codex")

_SOLO = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: solo-search}
spec:
  graph:
    nodes:
      - name: researcher
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1, model], delegate: []}
            tools:
              - {name: web_search, interface: "search/v1"}
              - {name: model}
          harness: {backend: codex, version: v1, params: {}}
"""

_SPAWN_OK = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: spawn-search}
spec:
  graph:
    nodes:
      - name: lead
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1, model], delegate: [search/v1, model]}
            tools:
              - {name: web_search, interface: "search/v1"}
              - {name: model}
            child:
              - name: sub
                authority: {invoke: [search/v1, model], delegate: []}
          harness: {backend: codex, version: v1, params: {}}
      - name: sub
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1, model], delegate: []}
            tools:
              - {name: web_search, interface: "search/v1"}
              - {name: model}
          harness: {backend: codex, version: v1, params: {}}
"""

_SPAWN_OMITS = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: spawn-omits}
spec:
  graph:
    nodes:
      - name: lead
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1, model], delegate: [search/v1, model]}
            tools:
              - {name: web_search, interface: "search/v1"}
              - {name: model}
            child:
              - name: sub
                authority: {invoke: [model], delegate: []}
          harness: {backend: codex, version: v1, params: {}}
      - name: sub
        spec:
          taskType: agent
          v2:
            authority: {invoke: [search/v1, model], delegate: []}
            tools:
              - {name: web_search, interface: "search/v1"}
              - {name: model}
          harness: {backend: codex, version: v1, params: {}}
"""


def _compile(text: str) -> Any:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl-t", parsed, source, bindings=_BINDINGS)
    return template


def _agents(template: Any) -> list[AgentOperator]:
    return [o for o in template.operators if isinstance(o, AgentOperator)]


def _lead_and_sub(template: Any) -> tuple[AgentOperator, AgentOperator]:
    agents = _agents(template)
    lead = next(a for a in agents if a.child_region_refs)
    sub = next(a for a in agents if not a.child_region_refs)
    return lead, sub


def test_declared_search_interface_pins_a_web_search_facade() -> None:
    (agent,) = _agents(_compile(_SOLO))
    facades = {f.name: f for f in agent.facades}
    assert "web_search" in facades
    search = facades["web_search"]
    assert search.kind is BoundaryEventKind.INVOCATION
    assert search.interface == "search/v1"
    schema = json.loads(search.tool_schema)
    assert schema["type"] == "function" and schema["name"] == "web_search"
    # A solo agent with no child regions is offered no spawn facade.
    assert "spawn_agent" not in facades


def test_a_spawnable_agent_gets_the_spawn_facade() -> None:
    lead, _ = _lead_and_sub(_compile(_SPAWN_OK))
    names = {f.name for f in lead.facades}
    assert "spawn_agent" in names
    spawn = next(f for f in lead.facades if f.name == "spawn_agent")
    assert spawn.kind is BoundaryEventKind.SPAWN and spawn.interface is None


def test_child_gets_web_search_when_region_ceiling_declares_it() -> None:
    _, sub = _lead_and_sub(_compile(_SPAWN_OK))
    assert any(
        f.name == "web_search" and f.interface == "search/v1" for f in sub.facades
    )


def test_region_ceiling_omitting_a_child_interface_is_a_compile_error() -> None:
    with pytest.raises(CompileError) as excinfo:
        _compile(_SPAWN_OMITS)
    codes = {d.code for d in excinfo.value.diagnostics}
    assert "region.child-tool-omitted" in codes
