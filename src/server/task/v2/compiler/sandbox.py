"""Pin each agent's local sandbox binding from its declared authority."""

from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    SandboxRuntimeProfile,
)

from ..representations.operators import (
    AgentOperator,
    AgentSandboxBinding,
    BindingProvenance,
    EffectBoundary,
    EffectClass,
    EffectReplayContract,
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

    A requested egress mode is pinned as the author wrote it, never downgraded here: an
    unauthorized or ungated request has to reach validation to be refused, because
    silently pinning it back to ``deny`` would run the workflow under a fence its author
    did not ask for.
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
        if binding is not None and binding.network_egress.allows_egress:
            # Declared once for the binding, never per command: the waiver is what the
            # source map records, and no individual command earns a receipt from it.
            acc.effect_boundaries.append(
                EffectBoundary(
                    effect_class=EffectClass.EXTERNAL_EFFECT,
                    replay_contract=EffectReplayContract.AUTHOR_OWNED_AT_LEAST_ONCE,
                    source_ref=op.operator_id,
                )
            )


def egress_requested(op: AgentOperator) -> bool:
    """Whether the agent's pinned binding asks for the network-egress opt-in."""
    return (
        op.sandbox_binding is not None
        and op.sandbox_binding.network_egress.allows_egress
    )


def egress_authorized(op: AgentOperator, enabled: bool) -> bool:
    """Whether the deployment and the agent's declared ceiling both permit egress.

    This is the static half of the decision. The effective half — whether the
    activation's own attenuated grant carries the interface — is resolved when the
    dispatch mints its capability, because a child's grant does not exist until then.
    """
    return enabled and SANDBOX_EGRESS_INTERFACE in op.authority.invoke
