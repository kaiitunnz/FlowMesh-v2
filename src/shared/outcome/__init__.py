"""Reference-backed invocation outcomes: bounded manifests over immutable content."""

from shared.content import content_digest

from .carrier import InlineControl, ManifestRef, OutcomeCarrier
from .content_store import FabricContentStore, OutcomeHydrationError
from .manifest import OutcomeManifest

__all__ = [
    "FabricContentStore",
    "InlineControl",
    "ManifestRef",
    "OutcomeCarrier",
    "OutcomeHydrationError",
    "OutcomeManifest",
    "content_digest",
]
