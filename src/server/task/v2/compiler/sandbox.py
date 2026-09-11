"""Pin each agent's local sandbox binding from its declared authority."""

from shared.sandbox import SANDBOX_EXECUTE_INTERFACE, SandboxRuntimeProfile

from ..representations.operators import (
    AgentOperator,
    AgentSandboxBinding,
    BindingProvenance,
)
from .project import LoweringAccumulator


def pin_agent_sandbox(acc: LoweringAccumulator) -> None:
    """Reconcile each agent's sandbox binding with its declared authority, in place.

    The authority ceiling is the gate: an agent that declares ``sandbox.execute``
    without an envelope takes the deployment-approved default, and an envelope declared
    without the authority binds nothing, so a pinned binding always means the agent may
    actually run commands.
    """
    for index, op in enumerate(acc.operators):
        if not isinstance(op, AgentOperator):
            continue
        declared = SANDBOX_EXECUTE_INTERFACE in op.authority.invoke
        binding: AgentSandboxBinding | None = None
        if declared:
            binding = op.sandbox_binding or AgentSandboxBinding(
                profile=SandboxRuntimeProfile(), provenance=BindingProvenance.DEFAULT
            )
        if binding is not op.sandbox_binding:
            acc.operators[index] = op.model_copy(update={"sandbox_binding": binding})
