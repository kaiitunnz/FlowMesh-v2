"""Sealed generations of activation-private state.

A generation is the unit of recovery: the required components of a profile are sealed
together at one quiescence fence and restored together, so a harness home never resumes
beside a workspace from a different generation.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from .reference import BundleProfile


class StateComponentKind(StrEnum):
    """A registered, versioned kind of private-state component."""

    HARNESS_HOME_FS = "harness_home_fs"
    WORKSPACE_FS = "workspace_fs"


class Confidentiality(StrEnum):
    """Whether a component's bytes may carry secrets."""

    SECRET_BEARING = "secret_bearing"
    NON_SECRET = "non_secret"


class Exportability(StrEnum):
    """Where a sealed component may be materialized."""

    # Sealed in place under its holder; the seal certifies the holder's own copy.
    LOCAL_ONLY = "local_only"
    EXPORTABLE = "exportable"


class StateComponentSpec(BaseModel):
    """The registered contract of one component kind."""

    model_config = ConfigDict(frozen=True)

    kind: StateComponentKind
    schema_version: int
    confidentiality: Confidentiality
    exportability: Exportability


_SPECS: Mapping[StateComponentKind, StateComponentSpec] = MappingProxyType(
    {
        spec.kind: spec
        for spec in (
            # A harness home holds rollout transcripts and provider config, so it is
            # handled as secret-bearing regardless of what a given backend writes.
            StateComponentSpec(
                kind=StateComponentKind.HARNESS_HOME_FS,
                schema_version=1,
                confidentiality=Confidentiality.SECRET_BEARING,
                exportability=Exportability.LOCAL_ONLY,
            ),
            StateComponentSpec(
                kind=StateComponentKind.WORKSPACE_FS,
                schema_version=1,
                confidentiality=Confidentiality.SECRET_BEARING,
                exportability=Exportability.LOCAL_ONLY,
            ),
        )
    }
)

_PROFILE_COMPONENTS: Mapping[BundleProfile, frozenset[StateComponentKind]] = (
    MappingProxyType(
        {
            BundleProfile.AGENT_HARNESS: frozenset(
                {StateComponentKind.HARNESS_HOME_FS, StateComponentKind.WORKSPACE_FS}
            )
        }
    )
)


def component_spec(kind: StateComponentKind) -> StateComponentSpec:
    """The registered contract for a component kind."""
    return _SPECS[kind]


def required_components(profile: BundleProfile) -> frozenset[StateComponentKind]:
    """The components every sealed generation of a profile carries."""
    return _PROFILE_COMPONENTS[profile]


class SealedComponent(BaseModel):
    """One component of a sealed generation, identified by its content."""

    model_config = ConfigDict(frozen=True)

    kind: StateComponentKind
    schema_version: int
    content_digest: str  # canonical digest over the component's sealed tree
    size_bytes: int
    entry_count: int


class StateBundleManifest(BaseModel):
    """The immutable description of one sealed, coherent state generation.

    Every component seals at the manifest's ``quiescence_fence``, which is what makes
    the generation restorable as a unit. The manifest describes content and never
    locates it: a holder's path, mount, or cache entry is evidence about availability,
    not part of this identity.
    """

    model_config = ConfigDict(frozen=True)

    manifest_id: str
    reference_id: str
    generation: int
    profile: BundleProfile
    quiescence_fence: str
    components: tuple[SealedComponent, ...]

    @model_validator(mode="after")
    def _seal_is_coherent(self) -> Self:
        kinds = [component.kind for component in self.components]
        if len(set(kinds)) != len(kinds):
            raise ValueError("a sealed generation carries each component kind once")
        if missing := required_components(self.profile) - set(kinds):
            raise ValueError(
                "a sealed generation is incomplete without "
                + ", ".join(sorted(kind.value for kind in missing))
            )
        for component in self.components:
            expected = component_spec(component.kind).schema_version
            if component.schema_version != expected:
                raise ValueError(
                    f"component {component.kind.value} seals at schema version "
                    f"{expected}, not {component.schema_version}"
                )
        return self

    def component(self, kind: StateComponentKind) -> SealedComponent | None:
        """The sealed component of a kind, or None when the profile omits it."""
        return next((item for item in self.components if item.kind is kind), None)
