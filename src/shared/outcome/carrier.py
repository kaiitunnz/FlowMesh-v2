"""The settle-seam carrier for one durably delivered invocation outcome.

A worker producer settles a boundary with exactly one of two forms: a reference-backed
``ManifestRef`` for any engine/provider/service result, or a bounded, opaque
``InlineControl`` datum for a separately declared worker-produced control status. The
control plane routes the carrier by type without inspecting a payload: a ref lands as a
bounded manifest on the ledger and an inline datum as a bounded value.
"""

from pydantic import BaseModel, ConfigDict

from .manifest import OutcomeManifest


class InlineControl(BaseModel):
    """An opaque, bounded, worker-produced control datum injected inline."""

    model_config = ConfigDict(frozen=True)

    value: str


class ManifestRef(BaseModel):
    """A reference to materialized outcome content the resumed worker hydrates."""

    model_config = ConfigDict(frozen=True)

    manifest: OutcomeManifest


OutcomeCarrier = InlineControl | ManifestRef
