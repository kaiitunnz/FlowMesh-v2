from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    path: str


class ArtifactContext(BaseModel):
    """Which task published the artifacts a result's refs name; ``base_dir`` is where
    ``materialize`` extracted them locally."""

    model_config = ConfigDict(extra="ignore")

    task_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    revisions: dict[str, str] = Field(default_factory=dict, exclude_if=lambda v: not v)
    base_dir: str | None = Field(default=None, exclude_if=lambda v: v is None)


class ArtifactKind(StrEnum):
    """What a published artifact is."""

    FILE = "file"
    TREE = "tree"
    ARCHIVE = "archive"


class ArtifactDurability(StrEnum):
    """Where a published artifact's bytes are."""

    # Accepted at its task's success; its upload to the shared store is in progress.
    PENDING = "pending"
    DURABLE = "durable"
    # Its producer left before uploading it.
    LOST = "lost"


class DurabilityError(BaseModel):
    """Why an artifact is not durable."""

    code: str
    message: str


class PublishedArtifact(BaseModel):
    """One artifact a task published: a file, a tree of files, or an archive of a
    tree."""

    name: str
    revision: str
    kind: ArtifactKind
    size_bytes: int
    files: int
    sha256: str | None = None
    archive_of: str | None = None
    published_at: str | None = None
    durability: ArtifactDurability = ArtifactDurability.DURABLE
    durability_error: DurabilityError | None = None


class PublishedArtifactList(BaseModel):
    task_id: str
    artifacts: list[PublishedArtifact]
