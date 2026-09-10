"""Activation-private state: opaque references, sealed generations, and attachments.

An activation owns mutable private state under an opaque
:class:`ActivationPrivateStateReference`. A :class:`PrivateStateBinding` names the
generation and recovery mode a continuation resumes on, a :class:`StateBundleManifest`
describes one sealed generation, and a :class:`PrivateStateAttachment` grants one
holder exclusive authority to materialize and write it.
"""

from .attachment import (
    PrivateStateAttachment,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
)
from .binding import OwnerFence, PrivateStateBinding, PrivateStateRecoveryMode
from .manifest import (
    Confidentiality,
    Exportability,
    SealedComponent,
    StateBundleManifest,
    StateComponentKind,
    StateComponentSpec,
    component_spec,
    required_components,
)
from .reference import (
    ActivationPrivateStateReference,
    BundleProfile,
    PrivateStateIsolation,
)
from .seal import seal_component, verify_component

__all__ = [
    "ActivationPrivateStateReference",
    "BundleProfile",
    "Confidentiality",
    "Exportability",
    "OwnerFence",
    "PrivateStateAttachment",
    "PrivateStateBinding",
    "PrivateStateIsolation",
    "PrivateStateRecoveryMode",
    "PrivateStateUnavailable",
    "PrivateStateUnavailableReason",
    "SealedComponent",
    "StateBundleManifest",
    "StateComponentKind",
    "StateComponentSpec",
    "component_spec",
    "required_components",
    "seal_component",
    "verify_component",
]
