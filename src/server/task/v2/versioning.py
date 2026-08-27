import hashlib

from pydantic import BaseModel, ConfigDict, Field


def content_digest(payload: str) -> str:
    """Return a stable content digest for a serialized representation."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VersionId(BaseModel):
    """Immutable identity of one representation revision.

    A revision is never mutated in place. A later revision is a compatible
    successor: it keeps the same ``lineage`` and carries a strictly higher
    ``revision`` with its own ``content_digest``.
    """

    model_config = ConfigDict(frozen=True)

    lineage: str = Field(description="Stable identity shared across revisions.")
    revision: int = Field(default=1, ge=1, description="Monotonic revision number.")
    content_digest: str = Field(description="Digest of the represented content.")

    def is_compatible_successor(self, candidate: "VersionId") -> bool:
        """Whether ``candidate`` is a legal successor of this version."""
        return candidate.lineage == self.lineage and candidate.revision > self.revision

    def next_revision(self, content_digest: str) -> "VersionId":
        """Return the next revision in this lineage for new content."""
        return VersionId(
            lineage=self.lineage,
            revision=self.revision + 1,
            content_digest=content_digest,
        )
