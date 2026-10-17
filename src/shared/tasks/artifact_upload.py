"""The wire contract of an artifact upload that follows a task's success.

A worker reports a task's success once its final artifacts are hashed into custody, and
uploads them to the shared store afterwards. The success says so with
``ARTIFACTS_PENDING``; the worker then reports how each task's upload went, and control
answers each report, and the success itself, with the upload's settled state.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from shared.content.refusal import StoreRefusal

# The success-payload flag marking the final artifacts it reports, its merged
# children's included, as staged rather than stored.
ARTIFACTS_PENDING = "artifacts_pending"

# The task event a worker reports an upload's progress with.
ARTIFACT_UPLOAD_EVENT = "TASK_ARTIFACT_UPLOAD"

# The control frame answering a worker about one task's upload.
UPLOAD_SETTLED_FRAME_KIND = "artifact_upload_settled"


class UploadReport(StrEnum):
    """What a worker reports about one task's upload."""

    # Every staged object is in the shared store.
    UPLOADED = "uploaded"
    # A staged copy no longer holds its bytes, so the upload cannot finish.
    UNUPLOADABLE = "unuploadable"
    # Staged under an earlier registration of this worker, which no longer uploads it.
    ABANDONED = "abandoned"


class ArtifactDurability(StrEnum):
    """Where an artifact's bytes are."""

    # Accepted at its task's success and still uploading to the shared store.
    PENDING = "pending"
    DURABLE = "durable"
    # Its producer left before storing it.
    LOST = "lost"


class UploadDisposition(StrEnum):
    """Control's answer about one task's upload."""

    # The success announcing it is accepted; the upload is still owed.
    PENDING = "pending"
    DURABLE = "durable"
    LOST = "lost"
    # No accepted upload matches the report, so nothing is owed for it.
    REJECTED = "rejected"
    # It cannot settle durable now; its refusal says what the worker does before it
    # reports the upload again.
    REFUSED = "refused"

    @property
    def settled(self) -> bool:
        return self not in _OWED


_OWED = frozenset(
    {
        UploadDisposition.PENDING,
        UploadDisposition.REFUSED,
    }
)


class UploadSettled(BaseModel):
    """Control's answer about one task's upload, for the dispatch that owes it."""

    task_id: str
    dispatch_id: str
    disposition: UploadDisposition
    refusal: StoreRefusal | None = None
    digests: list[str] = Field(default_factory=list)
    """The objects a refusal concerns."""
