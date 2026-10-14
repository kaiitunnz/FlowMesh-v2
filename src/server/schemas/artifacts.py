from pydantic import BaseModel, Field

from shared.tasks.artifact_binding import ArtifactKind


class PublishedArtifact(BaseModel):
    name: str = Field(description="Name the artifact is published and downloaded by.")
    revision: str = Field(description="final, or step-<n> for a training checkpoint.")
    kind: ArtifactKind = Field(
        description="file, tree (a directory of files) or archive."
    )
    size_bytes: int = Field(description="Size of the file, the tree or the archive.")
    files: int = Field(description="Files the artifact holds.")
    sha256: str | None = Field(
        default=None, description="SHA-256 of a file's or an archive's whole bytes."
    )
    archive_of: str | None = Field(
        default=None, description="Name of the tree an archive packs."
    )
    published_at: str | None = Field(
        default=None, description="When the artifact was published."
    )


class PublishedArtifactList(BaseModel):
    task_id: str = Field(description="Task that published the artifacts.")
    artifacts: list[PublishedArtifact] = Field(
        description="Artifacts, by revision then name."
    )
