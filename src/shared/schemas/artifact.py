from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    path: str = Field(description="Path relative to the task's artifacts/ dir.")


class ArtifactContext(BaseModel):
    """Which task's catalog the artifact refs in a result are published in."""

    model_config = ConfigDict(extra="ignore")

    task_id: str | None = Field(
        default=None, description="The task whose artifacts the refs name."
    )


def artifact_download_path(task_id: str, path: str) -> str:
    """The server route a task's published artifact downloads from."""
    return f"/api/v1/results/{task_id}/files/{path}"
