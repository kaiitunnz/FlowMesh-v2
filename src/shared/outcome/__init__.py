"""Reference-backed invocation outcomes: bounded manifests over immutable content."""

from .carrier import InlineControl, ManifestRef, OutcomeCarrier
from .content_store import (
    ContentStoreError,
    FabricContentStore,
    OutcomeHydrationError,
)
from .manifest import OutcomeManifest, content_digest

__all__ = [
    "ContentStoreError",
    "FabricContentStore",
    "InlineControl",
    "ManifestRef",
    "OutcomeCarrier",
    "OutcomeHydrationError",
    "OutcomeManifest",
    "content_digest",
]
