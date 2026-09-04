"""The settle-seam carrier for one durably delivered invocation outcome.

A worker producer settles a boundary with exactly one of two forms: a reference-backed
``ManifestRef`` for any engine/provider/service result, or a bounded, opaque
``InlineControl`` datum for a separately declared worker-produced control status. The
control plane routes the carrier without inspecting a payload — a ``ManifestRef`` lands
as a bounded manifest on the ledger and an ``InlineControl`` as a bounded value.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .manifest import OutcomeManifest


class InlineControl(BaseModel):
    """An opaque, bounded, worker-produced control datum injected inline."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["inline"] = "inline"
    value: str


class ManifestRef(BaseModel):
    """A reference to materialized outcome content the resumed worker hydrates."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["manifest"] = "manifest"
    manifest: OutcomeManifest


OutcomeCarrier = InlineControl | ManifestRef
