from pydantic import BaseModel, Field

from shared.tasks.artifact_binding import ArtifactKind
from shared.tasks.artifact_upload import ArtifactDurability


class DurabilityError(BaseModel):
    code: str = Field(description="Why the artifact is not durable: artifact_lost.")
    message: str = Field(description="What was lost, and why.")


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
    durability: ArtifactDurability = Field(
        default=ArtifactDurability.DURABLE,
        description="pending while its upload to the shared store is still owed, "
        "durable once stored, lost when its producer left before storing it.",
    )
    durability_error: DurabilityError | None = Field(
        default=None, description="Why a lost artifact was lost."
    )


class PublishedArtifactList(BaseModel):
    task_id: str = Field(description="Task that published the artifacts.")
    artifacts: list[PublishedArtifact] = Field(
        description="Artifacts, by revision then name."
    )
