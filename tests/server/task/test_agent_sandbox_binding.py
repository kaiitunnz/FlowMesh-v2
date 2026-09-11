"""Compiling an agent pins the local sandbox its authority declares.

The authority ceiling is the gate: declaring ``sandbox.execute`` pins an envelope and a
locally-resolved ``run_command`` facade, and declaring an envelope without the authority
binds nothing. The binding names a bounded local runtime, never a host or a service.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from server.task.parser import parse_workflow
from server.task.v2 import (
    CompileError,
    FrontendWorkflowSource,
    compile_workflow,
)
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.representations.operators import AgentOperator
from shared.sandbox import SANDBOX_EXECUTE_INTERFACE
from shared.tasks.specs import AgentSandboxSpec
from shared.tools.facade import FacadeResolution

_BINDINGS = AgentBindingDefaults(default_backend="scripted", sandbox_enabled=True)
_DISABLED = AgentBindingDefaults(default_backend="scripted")

_DECLARED = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: coder}
spec:
  graph:
    nodes:
      - name: coder
        spec:
          taskType: agent
          v2:
            authority: {invoke: ["sandbox.execute"], delegate: []}
          harness: {backend: scripted, version: v1, params: {script: []}}
"""

_WITH_ENVELOPE = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: coder-bounded}
spec:
  graph:
    nodes:
      - name: coder
        spec:
          taskType: agent
          v2:
            authority: {invoke: ["sandbox.execute"], delegate: []}
          sandbox: {command_timeout_sec: 5, cpu_seconds: 3}
          harness: {backend: scripted, version: v1, params: {script: []}}
"""

_ENVELOPE_WITHOUT_AUTHORITY = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: coder-undeclared}
spec:
  graph:
    nodes:
      - name: coder
        spec:
          taskType: agent
          v2:
            authority: {invoke: [], delegate: []}
          sandbox: {command_timeout_sec: 5}
          harness: {backend: scripted, version: v1, params: {script: []}}
"""


def _agent(text: str, bindings: AgentBindingDefaults = _BINDINGS) -> AgentOperator:
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    template, _ = compile_workflow("wfl-t", parsed, source, bindings=bindings)
    agents: list[Any] = [o for o in template.operators if isinstance(o, AgentOperator)]
    return agents[0]


def test_declaring_the_interface_pins_a_default_envelope() -> None:
    agent = _agent(_DECLARED)

    assert agent.sandbox_binding is not None
    assert agent.sandbox_binding.profile.runtime == "posix_process"
    assert agent.sandbox_binding.profile.command_timeout_sec > 0


def test_a_declared_envelope_is_pinned_as_written() -> None:
    agent = _agent(_WITH_ENVELOPE)

    assert agent.sandbox_binding is not None
    assert agent.sandbox_binding.profile.command_timeout_sec == 5
    assert agent.sandbox_binding.profile.cpu_seconds == 3


def test_an_envelope_without_the_authority_binds_nothing() -> None:
    agent = _agent(_ENVELOPE_WITHOUT_AUTHORITY)

    assert agent.sandbox_binding is None
    assert all(f.interface != SANDBOX_EXECUTE_INTERFACE for f in agent.facades)


def test_a_sandbox_agent_gets_a_locally_resolved_run_command_facade() -> None:
    agent = _agent(_DECLARED)

    facade = next(f for f in agent.facades if f.interface == SANDBOX_EXECUTE_INTERFACE)
    assert facade.name == "run_command"
    # The worker runs it inside the held turn; it is never a member the fabric settles.
    assert facade.resolution is FacadeResolution.LOCAL_INLINE
    schema = json.loads(facade.tool_schema)
    assert schema["name"] == "run_command"
    assert "command" in schema["parameters"]["properties"]


def test_a_deployment_that_disables_the_sandbox_refuses_a_declaring_agent() -> None:
    """The operator gate is fleet-wide: a workflow cannot opt itself in."""
    with pytest.raises(CompileError) as raised:
        _agent(_DECLARED, _DISABLED)

    assert "agent.sandbox.disabled" in str(raised.value)


def test_an_unknown_runtime_is_refused_rather_than_silently_substituted() -> None:
    """Naming a runtime the deployment does not provide must not run the weaker one."""
    with pytest.raises(ValidationError):
        AgentSandboxSpec(runtime="docker")
