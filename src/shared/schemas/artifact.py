from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    path: str = Field(description="Path relative to the task's artifacts/ dir.")


CHECKPOINTS_ARTIFACT = "checkpoints"


class ArtifactContext(BaseModel):
    """Which task's catalog the artifact refs in a result are published in."""

    model_config = ConfigDict(extra="ignore")

    task_id: str | None = Field(
        default=None, description="The task whose artifacts the refs name."
    )
    checkpoints: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Revision of the last checkpoint the task published.",
    )

    def revision_of(self, path: str) -> str | None:
        """Return the revision a ref to ``path`` names, or None for the final set."""
        if path.split("/", 1)[0] == CHECKPOINTS_ARTIFACT:
            return self.checkpoints
        return None


def artifact_download_path(task_id: str, path: str, revision: str | None = None) -> str:
    """Return the server route a task's published artifact downloads from."""
    route = f"/api/v1/results/{task_id}/files/{path}"
    return route if revision is None else f"{route}?revision={revision}"
