from pathlib import PurePosixPath

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    path: str = Field(description="Path relative to the task's artifacts/ dir.")


class ArtifactContext(BaseModel):
    base_dir: str = Field(description="Producing task's output directory.")
    base_url: str | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="HTTP origin (scheme://host[:port]) for upload.",
    )

    def url_for(self, path: str) -> str:
        """Where a consumer reads the producer's artifact at ``path``: its download URL
        on the server, or with no server origin the path on the producer's node."""
        if self.base_url:
            task_id = PurePosixPath(self.base_dir).name
            return f"{self.base_url.rstrip('/')}/api/v1/results/{task_id}/files/{path}"
        return (PurePosixPath(self.base_dir) / "artifacts" / path).as_posix()
