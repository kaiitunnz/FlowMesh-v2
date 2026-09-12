"""Pin each agent's local sandbox binding from its declared authority."""

from shared.sandbox import (
    SANDBOX_EGRESS_INTERFACE,
    SANDBOX_EXECUTE_INTERFACE,
    SandboxEgressMode,
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

    The egress mode resolves here too, and only here: an author who wrote one gets it
    pinned as written, never downgraded, because an unauthorized or ungated request has
    to reach validation to be refused rather than run under a fence its author did not
    ask for; an author who wrote none takes the mode their declared authority implies.
    """
    for index, op in enumerate(acc.operators):
        if not isinstance(op, AgentOperator):
            continue
        binding: AgentSandboxBinding | None = None
        if enabled and SANDBOX_EXECUTE_INTERFACE in op.authority.invoke:
            binding = op.sandbox_binding or AgentSandboxBinding(
                profile=SandboxRuntimeProfile(), provenance=BindingProvenance.DEFAULT
            )
            binding = binding.model_copy(
                update={"network_egress": _resolve_egress(acc, op)}
            )
        if binding != op.sandbox_binding:
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


def _resolve_egress(acc: LoweringAccumulator, op: AgentOperator) -> SandboxEgressMode:
    """Resolve the binding's egress mode from the author's request and the authority.

    The derivation runs one way only. An unset mode reads the agent's declared ceiling,
    so declaring ``sandbox.egress`` opts the agent's own commands in and a ceiling that
    holds the interface purely to delegate it opts back out with an explicit ``deny``.
    The reverse never happens: a binding grants no authority, and what an activation may
    actually do is decided against its effective grant when its capability is minted.
    """
    requested = acc.sandbox_egress_requests.get(op.operator_id)
    if requested is not None:
        return requested
    if SANDBOX_EGRESS_INTERFACE in op.authority.invoke:
        return SandboxEgressMode.AUTHOR_OWNED_AT_LEAST_ONCE
    return SandboxEgressMode.DENY


def egress_requested(op: AgentOperator) -> bool:
    """Whether the agent's pinned binding carries the network-egress opt-in."""
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
