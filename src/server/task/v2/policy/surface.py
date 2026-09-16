"""Assembly of the deployment's advisory policy surface.

A policy is deployment-global: a deployment selects one per hook through configuration,
and a workflow submission carries none. The surface is consulted at one locus, the
lowerer, while a template compiles.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ....config import PolicySurfaceConfig
from .builtin import RecomputeOnlyFusion, WarmRetention
from .lowering import (
    FusionPolicy,
    PolicySurface,
    ResidencyPolicy,
    ServiceFamilyPolicy,
)

type _Factory[T] = Callable[[PolicySurfaceConfig], T]


@dataclass(frozen=True)
class _Axis[T]:
    """One hook's registry and the knob a deployment selects it with."""

    knob: str
    registry: dict[str, _Factory[T]]


_FUSION = _Axis[FusionPolicy](
    knob="ORCHESTRATOR_FUSION_POLICY",
    registry={
        FusionPolicy.name: lambda _: FusionPolicy(),
        RecomputeOnlyFusion.name: lambda _: RecomputeOnlyFusion(),
    },
)

_RESIDENCY = _Axis[ResidencyPolicy](
    knob="ORCHESTRATOR_RESIDENCY_POLICY",
    registry={
        ResidencyPolicy.name: lambda _: ResidencyPolicy(),
        WarmRetention.name: lambda _: WarmRetention(),
    },
)

_SERVICE_FAMILY = _Axis[ServiceFamilyPolicy](
    knob="ORCHESTRATOR_SERVICE_FAMILY_POLICY",
    registry={ServiceFamilyPolicy.name: lambda _: ServiceFamilyPolicy()},
)


def _build[T](axis: _Axis[T], name: str, config: PolicySurfaceConfig) -> T:
    if (factory := axis.registry.get(name)) is None:
        raise ValueError(
            f"{axis.knob}='{name}' is not a policy; known values: "
            + ", ".join(sorted(axis.registry))
        )
    return factory(config)


def build_policy_surface(config: PolicySurfaceConfig) -> PolicySurface:
    """The policy selected at each hook, conservative where a deployment names none."""
    return PolicySurface(
        fusion=_build(_FUSION, config.fusion, config),
        residency=_build(_RESIDENCY, config.residency, config),
        service_family=_build(_SERVICE_FAMILY, config.service_family, config),
    )
