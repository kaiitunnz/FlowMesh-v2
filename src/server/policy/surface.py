"""Assembly of the deployment's advisory policy surface.

The surface is deployment-global: a deployment selects its policies through
configuration, and a workflow submission carries none. Each facet is consulted at its
own locus — the lowerer while a template compiles, the dispatcher while a ready episode
is placed, and the sealed-state inventory while an operator reads it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..config import PolicySurfaceConfig
from .lowering import LoweringPolicy
from .placement import InstanceStateLocality, PlacementPolicy
from .state_control import RecencyWarmth, StateControlPolicy

type _Factory[T] = Callable[[PolicySurfaceConfig], T]

_LOWERING: dict[str, _Factory[LoweringPolicy]] = {
    LoweringPolicy.name: lambda _: LoweringPolicy()
}
_PLACEMENT: dict[str, _Factory[PlacementPolicy]] = {
    PlacementPolicy.name: lambda _: PlacementPolicy(),
    InstanceStateLocality.name: lambda _: InstanceStateLocality(),
}
_STATE_CONTROL: dict[str, _Factory[StateControlPolicy]] = {
    StateControlPolicy.name: lambda _: StateControlPolicy(),
    RecencyWarmth.name: lambda config: RecencyWarmth(config.warm_generations),
}


@dataclass(frozen=True)
class PolicySurface:
    """The three advisory facets a deployment runs with."""

    lowering: LoweringPolicy
    placement: PlacementPolicy
    state_control: StateControlPolicy


def _build[T](
    registry: dict[str, _Factory[T]],
    name: str,
    knob: str,
    config: PolicySurfaceConfig,
) -> T:
    if (factory := registry.get(name)) is None:
        raise ValueError(
            f"{knob}='{name}' is not a policy; known values: "
            + ", ".join(sorted(registry))
        )
    return factory(config)


def build_policy_surface(config: PolicySurfaceConfig) -> PolicySurface | None:
    """The configured surface, or ``None`` where the deployment enables no policy."""
    if not config.enabled:
        return None
    return PolicySurface(
        lowering=_build(
            _LOWERING, config.lowering, "ORCHESTRATOR_LOWERING_POLICY", config
        ),
        placement=_build(
            _PLACEMENT, config.placement, "ORCHESTRATOR_PLACEMENT_POLICY", config
        ),
        state_control=_build(
            _STATE_CONTROL,
            config.state_control,
            "ORCHESTRATOR_STATE_CONTROL_POLICY",
            config,
        ),
    )
