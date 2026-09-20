"""Reference-backed invocation outcomes: bounded manifests over immutable content."""

from shared.content import content_digest

from .carrier import InlineControl, ManifestRef, OutcomeCarrier
from .content_store import FabricContentStore
from .finalizing_store import FinalizationIndexClient, FinalizingContentStore
from .manifest import OutcomeManifest

__all__ = [
    "FabricContentStore",
    "FinalizationIndexClient",
    "FinalizingContentStore",
    "InlineControl",
    "ManifestRef",
    "OutcomeCarrier",
    "OutcomeManifest",
    "content_digest",
]
