from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    path: str


class ArtifactContext(BaseModel):
    """Which task published the artifacts a result's refs name; ``base_dir`` is where
    ``materialize`` extracted them locally."""

    model_config = ConfigDict(extra="ignore")

    task_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    base_dir: str | None = Field(default=None, exclude_if=lambda v: v is None)


class PublishedArtifact(BaseModel):
    """One artifact a task published: a file, a tree of files, or an archive of a
    tree."""

    name: str
    revision: str
    kind: str
    size_bytes: int
    files: int
    sha256: str | None = None
    archive_of: str | None = None
    published_at: str | None = None


class PublishedArtifactList(BaseModel):
    task_id: str
    artifacts: list[PublishedArtifact]
