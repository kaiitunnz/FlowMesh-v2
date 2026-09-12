"""Resolving an agent's sandbox authority at capability mint, against its grant.

The pinned binding says what the workflow asked for; the activation's own invoke face
says what its ancestors left it. The mint takes the intersection for both interfaces, so
a child never inherits code execution or egress its parent withheld.
"""

from server.task.runtime import _effective_facades, _sandbox_capability
from server.task.v2.compiler.facades import run_command_schema
from server.task.v2.representations.operators import (
    AgentOperator,
    AgentSandboxBinding,
    AuthorityCeiling,
    BindingKey,
    BindingProvenance,
)
from shared.harness.boundary import BoundaryEventKind
from shared.private_state import PrivateStateAttachment
from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    SandboxEgressMode,
    SandboxRuntimeProfile,
)
from shared.tasks.task_type import TaskType
from shared.tools.facade import FacadeDescriptor, FacadeResolution

_ATTACHMENT = PrivateStateAttachment(
    attachment_id="psa-1",
    reference_id="aps-1",
    generation=1,
    worker_id="wkr-1",
    incarnation=2,
    write_epoch=3,
)


def _agent(mode: SandboxEgressMode) -> AgentOperator:
    return AgentOperator(
        operator_id="coder",
        source_ref="coder",
        binding=BindingKey(task_type=TaskType.AGENT),
        authority=AuthorityCeiling(
            invoke=(SANDBOX_EXECUTE_INTERFACE, SANDBOX_EGRESS_INTERFACE)
        ),
        sandbox_binding=AgentSandboxBinding(
            profile=SandboxRuntimeProfile(),
            provenance=BindingProvenance.SOURCE,
            network_egress=mode,
        ),
    )


def test_an_egress_binding_mints_egress_when_the_grant_still_carries_it() -> None:
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
        _ATTACHMENT,
        (SANDBOX_EXECUTE_INTERFACE, SANDBOX_EGRESS_INTERFACE),
    )

    assert capability is not None
    assert capability.egress_allowed


def test_a_child_whose_grant_lost_the_interface_is_minted_fenced() -> None:
    """The declared ceiling still names egress; the attenuated grant does not."""
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
        _ATTACHMENT,
        (SANDBOX_EXECUTE_INTERFACE,),
    )

    assert capability is not None
    assert not capability.egress_allowed
    assert capability.network_egress is SandboxEgressMode.DENY


def test_a_fenced_binding_stays_fenced_under_a_grant_that_would_allow_egress() -> None:
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.DENY),
        _ATTACHMENT,
        (SANDBOX_EXECUTE_INTERFACE, SANDBOX_EGRESS_INTERFACE),
    )

    assert capability is not None
    assert not capability.egress_allowed


def test_an_agent_without_an_attachment_gets_no_capability() -> None:
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
            None,
            (SANDBOX_EGRESS_INTERFACE,),
        )
        is None
    )


def test_an_activation_without_effective_execute_gets_no_capability() -> None:
    """A declared ceiling is not authority: without the delegated interface the
    activation cannot run a command at all."""
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.DENY), _ATTACHMENT, ("web_search",)
        )
        is None
    )


def test_a_withheld_execute_denies_the_capability_even_with_egress_delegated() -> None:
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
            _ATTACHMENT,
            (SANDBOX_EGRESS_INTERFACE,),
        )
        is None
    )


def test_a_delegated_execute_face_still_mints() -> None:
    """The refusal is specific: the ordinary authorized child keeps working."""
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.DENY), _ATTACHMENT, (SANDBOX_EXECUTE_INTERFACE,)
    )

    assert capability is not None
    assert not capability.egress_allowed


def _facades(op: AgentOperator) -> tuple[str, ...]:
    return tuple(f.name for f in op.facades)


def _with_facades(mode: SandboxEgressMode) -> AgentOperator:
    op = _agent(mode)
    return op.model_copy(
        update={
            "facades": (
                FacadeDescriptor(
                    name="web_search",
                    kind=BoundaryEventKind.INVOCATION,
                    interface="search/v1",
                    tool_schema="{}",
                ),
                FacadeDescriptor(
                    name="run_command",
                    kind=BoundaryEventKind.STATE_ACCESS,
                    interface=SANDBOX_EXECUTE_INTERFACE,
                    tool_schema=run_command_schema(mode.allows_egress),
                    resolution=FacadeResolution.LOCAL_INLINE,
                ),
            )
        }
    )


def test_a_child_without_effective_execute_is_not_offered_run_command() -> None:
    op = _with_facades(SandboxEgressMode.DENY)

    offered = _effective_facades(op, ("search/v1",), None)

    assert [f.name for f in offered] == ["web_search"]


def test_a_child_without_effective_egress_is_offered_a_no_network_description() -> None:
    """The binding asks for egress; this activation's grant does not carry it."""
    op = _with_facades(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE)
    fenced = _sandbox_capability(op, _ATTACHMENT, (SANDBOX_EXECUTE_INTERFACE,))

    offered = _effective_facades(op, (SANDBOX_EXECUTE_INTERFACE,), fenced)

    schema = next(f for f in offered if f.name == "run_command").tool_schema
    assert "no network access" in schema
    assert "can reach the network" not in schema


def test_an_authorized_activation_keeps_its_facades_and_description() -> None:
    """A filter that refuses everything would fail here."""
    op = _with_facades(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE)
    face = (SANDBOX_EXECUTE_INTERFACE, SANDBOX_EGRESS_INTERFACE, "search/v1")
    capability = _sandbox_capability(op, _ATTACHMENT, face)

    offered = _effective_facades(op, face, capability)

    assert [f.name for f in offered] == ["web_search", "run_command"]
    schema = next(f for f in offered if f.name == "run_command").tool_schema
    assert "can reach the network" in schema


def test_a_mediated_facade_stays_offered_so_its_denial_is_recorded() -> None:
    """A mediated call the grant forbids settles as a durable authority denial; not
    offering it would erase that record."""
    op = _with_facades(SandboxEgressMode.DENY)
    capability = _sandbox_capability(op, _ATTACHMENT, (SANDBOX_EXECUTE_INTERFACE,))

    offered = _effective_facades(op, (SANDBOX_EXECUTE_INTERFACE,), capability)

    assert "web_search" in [f.name for f in offered]
