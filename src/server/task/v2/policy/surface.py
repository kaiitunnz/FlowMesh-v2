"""Assembly of the deployment's advisory policy surface.

The surface is deployment-global: a deployment selects its policy through
configuration, and a workflow submission carries none. It is consulted at one locus,
the lowerer, while a template compiles.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ....config import PolicySurfaceConfig
from .demo import DemoPolicy, FusionVetoPolicy, WarmthPolicy
from .lowering import LoweringPolicy

type _Factory[T] = Callable[[PolicySurfaceConfig], T]

_LOWERING: dict[str, _Factory[LoweringPolicy]] = {
    LoweringPolicy.name: lambda _: LoweringPolicy(),
    FusionVetoPolicy.name: lambda _: FusionVetoPolicy(),
    WarmthPolicy.name: lambda _: WarmthPolicy(),
    DemoPolicy.name: lambda _: DemoPolicy(),
}


@dataclass(frozen=True)
class PolicySurface:
    """The advisory facets a deployment runs with."""

    lowering: LoweringPolicy


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
        )
    )
