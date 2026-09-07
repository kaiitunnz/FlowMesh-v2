"""The worker egress lane: the mediated-egress sidecar, its backends, and the fence.

The sidecar validates a one-use permit against the fence and dispatches it to the
per-interface backend that egresses through the local provider. ``model_turn`` builds on
this package; the dependency runs one way.
"""

from .backends import ModelEgress, SearchEgress
from .fence import ProviderBinding, fence_reason, materialize_tool_outcome
from .sidecar import (
    AudienceFn,
    EgressInterface,
    HeldEgressReject,
    MediatedEgressSidecar,
    OutcomeSink,
    SyncModelEgress,
)

__all__ = [
    "AudienceFn",
    "EgressInterface",
    "HeldEgressReject",
    "MediatedEgressSidecar",
    "ModelEgress",
    "OutcomeSink",
    "ProviderBinding",
    "SearchEgress",
    "SyncModelEgress",
    "fence_reason",
    "materialize_tool_outcome",
]
