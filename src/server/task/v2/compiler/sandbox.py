"""Pin each agent's local sandbox binding from its declared authority."""

from shared.sandbox import SANDBOX_EXECUTE_INTERFACE, SandboxRuntimeProfile

from ..representations.operators import (
    AgentOperator,
    AgentSandboxBinding,
    BindingProvenance,
)
from .project import LoweringAccumulator


def pin_agent_sandbox(acc: LoweringAccumulator, enabled: bool = False) -> None:
    """Reconcile each agent's sandbox binding with its declared authority, in place.

    Two gates, both of which must be open. ``enabled`` is the deployment's: the local
    fence is a single-trusted-tenant posture, so an operator opts the fleet in or no
    agent gets it. The authority ceiling is the workflow's: an agent that declares
    ``sandbox.execute`` without an envelope takes the default envelope, and an envelope
    declared without the authority binds nothing. A pinned binding therefore always
    means the agent may actually run commands, and an agent that asks for one where the
    deployment disables it is refused by validation rather than running on without it.
    """
    for index, op in enumerate(acc.operators):
        if not isinstance(op, AgentOperator):
            continue
        declared = enabled and SANDBOX_EXECUTE_INTERFACE in op.authority.invoke
        binding: AgentSandboxBinding | None = None
        if declared:
            binding = op.sandbox_binding or AgentSandboxBinding(
                profile=SandboxRuntimeProfile(), provenance=BindingProvenance.DEFAULT
            )
        if binding is not op.sandbox_binding:
            acc.operators[index] = op.model_copy(update={"sandbox_binding": binding})
