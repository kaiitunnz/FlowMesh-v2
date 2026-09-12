"""Resolving an agent's sandbox authority at capability mint, against its grant.

The pinned binding says what the workflow asked for; the activation's own invoke face
says what its ancestors left it. The mint takes the intersection for both interfaces, so
a child never inherits code execution or egress its parent withheld.
"""

from server.task.runtime import _sandbox_capability
from server.task.v2.representations.operators import (
    AgentOperator,
    AgentSandboxBinding,
    AuthorityCeiling,
    BindingKey,
    BindingProvenance,
)
from shared.private_state import PrivateStateAttachment
from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    SandboxEgressMode,
    SandboxRuntimeProfile,
)
from shared.tasks.task_type import TaskType

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
        4,
    )

    assert capability is not None
    assert capability.egress_allowed
    assert capability.authority_epoch == 4


def test_a_child_whose_grant_lost_the_interface_is_minted_fenced() -> None:
    """The declared ceiling still names egress; the attenuated grant does not."""
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
        _ATTACHMENT,
        (SANDBOX_EXECUTE_INTERFACE,),
        9,
    )

    assert capability is not None
    assert not capability.egress_allowed
    assert capability.network_egress is SandboxEgressMode.DENY


def test_a_fenced_binding_stays_fenced_under_a_grant_that_would_allow_egress() -> None:
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.DENY),
        _ATTACHMENT,
        (SANDBOX_EXECUTE_INTERFACE, SANDBOX_EGRESS_INTERFACE),
        1,
    )

    assert capability is not None
    assert not capability.egress_allowed


def test_an_agent_without_an_attachment_gets_no_capability() -> None:
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
            None,
            (SANDBOX_EGRESS_INTERFACE,),
            0,
        )
        is None
    )


def test_an_activation_without_effective_execute_gets_no_capability() -> None:
    """A declared ceiling is not authority: without the delegated interface the
    activation cannot run a command at all."""
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.DENY), _ATTACHMENT, ("web_search",), 2
        )
        is None
    )


def test_a_withheld_execute_denies_the_capability_even_with_egress_delegated() -> None:
    assert (
        _sandbox_capability(
            _agent(SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE),
            _ATTACHMENT,
            (SANDBOX_EGRESS_INTERFACE,),
            3,
        )
        is None
    )


def test_a_delegated_execute_face_still_mints() -> None:
    """The refusal is specific: the ordinary authorized child keeps working."""
    capability = _sandbox_capability(
        _agent(SandboxEgressMode.DENY), _ATTACHMENT, (SANDBOX_EXECUTE_INTERFACE,), 5
    )

    assert capability is not None
    assert not capability.egress_allowed
