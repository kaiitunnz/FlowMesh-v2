"""Reference-backed invocation outcomes: bounded manifests over immutable content."""

from shared.content import content_digest

from .carrier import InlineControl, ManifestRef, OutcomeCarrier
from .content_store import FabricContentStore
from .manifest import OutcomeManifest

__all__ = [
    "FabricContentStore",
    "InlineControl",
    "ManifestRef",
    "OutcomeCarrier",
    "OutcomeManifest",
    "content_digest",
]
