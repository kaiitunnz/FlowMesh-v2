from pydantic import BaseModel, ConfigDict, Field

from ...utils.time import now_iso
from .versioning import content_digest


class FrontendWorkflowSource(BaseModel):
    """Immutable compiler input retained for provenance and diagnostics.

    Records what the author submitted. It is not a durable runtime object and is
    never conflated with the template/plan version or the workflow's task-ID list.
    """

    model_config = ConfigDict(frozen=True)

    raw_payload: str = Field(description="Verbatim submitted workflow text.")
    format: str = Field(default="native", description="Submission format.")
    name: str | None = Field(default=None, description="Author-declared name.")
    digest: str = Field(description="Content digest of the submitted payload.")
    submitted_at: str = Field(
        default_factory=now_iso, description="Submission timestamp."
    )

    @classmethod
    def capture(
        cls,
        raw_payload: str,
        format: str = "native",
        name: str | None = None,
    ) -> "FrontendWorkflowSource":
        """Capture submitted source text with its content digest."""
        return cls(
            raw_payload=raw_payload,
            format=format,
            name=name,
            digest=content_digest(raw_payload),
        )
