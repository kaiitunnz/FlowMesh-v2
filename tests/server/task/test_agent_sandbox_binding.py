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
from server.task.runtime import _sandbox_capability
from server.task.v2 import (
    CompileError,
    FrontendWorkflowSource,
    compile_workflow,
)
from server.task.v2.compiler.agent_binding import AgentBindingDefaults
from server.task.v2.representations.operators import (
    AgentOperator,
    EffectClass,
    EffectReplayContract,
)
from shared.private_state import PrivateStateAttachment
from shared.sandbox import (
    SANDBOX_EXECUTE_INTERFACE,
    SandboxEgressMode,
)
from shared.tasks.specs import AgentSandboxSpec
from shared.tools.facade import FacadeResolution

_ATTACHMENT = PrivateStateAttachment(
    attachment_id="psa-1",
    reference_id="aps-1",
    generation=1,
    worker_id="wkr-1",
    incarnation=2,
    write_epoch=3,
)

_BINDINGS = AgentBindingDefaults(default_backend="scripted", sandbox_enabled=True)
_DISABLED = AgentBindingDefaults(default_backend="scripted")
_EGRESS = AgentBindingDefaults(
    default_backend="scripted", sandbox_enabled=True, sandbox_egress_enabled=True
)

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


def _compile(text: str, bindings: AgentBindingDefaults = _BINDINGS):
    parsed = parse_workflow(text, "native")
    source = FrontendWorkflowSource.capture(text, "native", name="wf")
    return compile_workflow("wfl-t", parsed, source, bindings=bindings)


def _agent(text: str, bindings: AgentBindingDefaults = _BINDINGS) -> AgentOperator:
    template, _ = _compile(text, bindings)
    agents: list[Any] = [o for o in template.operators if isinstance(o, AgentOperator)]
    return agents[0]


def _egress_source(
    interfaces: str, mode: str | None = "author_owned_at_least_once"
) -> str:
    sandbox = "" if mode is None else f"\n          sandbox: {{network_egress: {mode}}}"
    return f"""
apiVersion: flowmesh/v2
kind: Workflow
metadata: {{name: coder-egress}}
spec:
  graph:
    nodes:
      - name: coder
        spec:
          taskType: agent
          v2:
            authority: {{invoke: [{interfaces}], delegate: []}}{sandbox}
          harness: {{backend: scripted, version: v1, params: {{script: []}}}}
"""


_BOTH_INTERFACES = '"sandbox.execute", "sandbox.egress"'
_EGRESS_DECLARED = _egress_source(_BOTH_INTERFACES)
_EGRESS_UNAUTHORIZED = _egress_source('"sandbox.execute"')
_EGRESS_DERIVED = _egress_source(_BOTH_INTERFACES, None)
_EGRESS_DELEGATE_ONLY = _egress_source(_BOTH_INTERFACES, "deny")


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


def test_the_egress_opt_in_pins_when_authority_and_deployment_agree() -> None:
    agent = _agent(_EGRESS_DECLARED, _EGRESS)

    assert agent.sandbox_binding is not None
    assert (
        agent.sandbox_binding.network_egress
        is SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE
    )


def test_the_default_binding_denies_egress() -> None:
    agent = _agent(_DECLARED)

    assert agent.sandbox_binding is not None
    assert agent.sandbox_binding.network_egress is SandboxEgressMode.DENY


def test_egress_without_the_distinct_authority_is_refused_not_downgraded() -> None:
    """sandbox.execute does not imply sandbox.egress, and the request is not silently
    run under the fence it asked to leave."""
    with pytest.raises(CompileError) as raised:
        _agent(_EGRESS_UNAUTHORIZED, _EGRESS)

    assert "agent.sandbox.egress.disabled" in str(raised.value)


def test_egress_without_the_deployment_gate_is_refused() -> None:
    with pytest.raises(CompileError) as raised:
        _agent(_EGRESS_DECLARED, _BINDINGS)

    assert "agent.sandbox.egress.disabled" in str(raised.value)


def test_an_egress_binding_declares_one_effect_boundary_for_the_whole_binding() -> None:
    """The waiver is source-mapped once, so no command earns a receipt of its own."""
    template, _ = _compile(_EGRESS_DECLARED, _EGRESS)
    agent = next(o for o in template.operators if isinstance(o, AgentOperator))

    boundaries = [
        b for b in template.effect_boundaries if b.source_ref == agent.operator_id
    ]
    assert len(boundaries) == 1
    assert boundaries[0].effect_class is EffectClass.EXTERNAL_EFFECT
    assert (
        boundaries[0].replay_contract is EffectReplayContract.AUTHOR_OWNED_AT_LEAST_ONCE
    )


def test_a_fenced_binding_declares_no_effect_boundary() -> None:
    template, _ = _compile(_DECLARED)
    agent = next(o for o in template.operators if isinstance(o, AgentOperator))

    assert not [
        b for b in template.effect_boundaries if b.source_ref == agent.operator_id
    ]


def test_the_tool_description_tells_the_model_which_fence_it_has() -> None:
    fenced = _agent(_DECLARED).facades
    opened = _agent(_EGRESS_DECLARED, _EGRESS).facades

    fenced_schema = json.loads(
        next(f for f in fenced if f.name == "run_command").tool_schema
    )
    open_schema = json.loads(
        next(f for f in opened if f.name == "run_command").tool_schema
    )
    assert "no network access" in fenced_schema["description"]
    assert "can reach the network" in open_schema["description"]


def test_declaring_the_egress_authority_opts_the_agent_in_without_a_binding_line() -> (
    None
):
    """The authority is the author-facing surface: an unset mode derives from it."""
    agent = _agent(_EGRESS_DERIVED, _EGRESS)

    assert agent.sandbox_binding is not None
    assert (
        agent.sandbox_binding.network_egress
        is SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE
    )


def test_a_derived_opt_in_declares_the_effect_boundary_the_written_one_declares() -> (
    None
):
    template, _ = _compile(_EGRESS_DERIVED, _EGRESS)
    agent = next(o for o in template.operators if isinstance(o, AgentOperator))

    boundaries = [
        b for b in template.effect_boundaries if b.source_ref == agent.operator_id
    ]
    assert len(boundaries) == 1
    assert boundaries[0].effect_class is EffectClass.EXTERNAL_EFFECT
    assert (
        boundaries[0].replay_contract is EffectReplayContract.AUTHOR_OWNED_AT_LEAST_ONCE
    )


def test_a_delegate_only_ceiling_opts_its_own_commands_back_out() -> None:
    """Holding sandbox.egress to hand to children is not running commands with it."""
    template, _ = _compile(_EGRESS_DELEGATE_ONLY, _EGRESS)
    agent = next(o for o in template.operators if isinstance(o, AgentOperator))

    assert agent.sandbox_binding is not None
    assert agent.sandbox_binding.network_egress is SandboxEgressMode.DENY
    assert not [
        b for b in template.effect_boundaries if b.source_ref == agent.operator_id
    ]


def test_a_derived_opt_in_is_still_decided_against_the_effective_grant() -> None:
    """The derive reads the declared ceiling, the mint reads what the parent delegated.

    A child whose ceiling names sandbox.egress derives an egress-on binding, and the
    mint must still fence it when the delegated face withheld the interface.
    """
    agent = _agent(_EGRESS_DERIVED, _EGRESS)
    assert agent.sandbox_binding is not None
    assert agent.sandbox_binding.network_egress.allows_egress

    capability = _sandbox_capability(agent, _ATTACHMENT, (SANDBOX_EXECUTE_INTERFACE,))

    assert capability is not None
    assert not capability.egress_allowed
    assert capability.network_egress is SandboxEgressMode.DENY
