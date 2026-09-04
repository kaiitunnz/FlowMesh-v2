"""Reference-backed invocation outcomes: bounded manifests over immutable content."""

from .carrier import InlineControl, ManifestRef, OutcomeCarrier
from .content_store import (
    ContentStoreError,
    FabricContentStore,
    OutcomeHydrationError,
    OutcomeSpool,
)
from .manifest import OutcomeAccessBinding, OutcomeManifest, content_digest

__all__ = [
    "ContentStoreError",
    "FabricContentStore",
    "InlineControl",
    "ManifestRef",
    "OutcomeAccessBinding",
    "OutcomeCarrier",
    "OutcomeHydrationError",
    "OutcomeManifest",
    "OutcomeSpool",
    "content_digest",
]
