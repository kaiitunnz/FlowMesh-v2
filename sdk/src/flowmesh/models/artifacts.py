from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    path: str


class ArtifactContext(BaseModel):
    """Which task published the artifacts a result's refs name; ``base_dir`` is where
    ``materialize`` extracted them locally."""

    model_config = ConfigDict(extra="ignore")

    task_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    base_dir: str | None = Field(default=None, exclude_if=lambda v: v is None)
