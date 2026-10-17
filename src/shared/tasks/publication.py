"""The wire contract of a worker's publication: a report naming content it stored, and
control's answer to it.

A worker reports a task's success, a checkpoint it published, or the request an input
preparation materialized by naming the objects it stored. Control validates every named
object before it binds any, and answers each report once its outcome is durable: a
committed or rejected report ends the worker's duty to resend it, and a refused one says
what the worker does before it reports again. A report is identified by the publication
it makes, so a resend of it is the same publication.
"""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from shared.content import ContentReference, StoreRefusal

from .artifact_binding import ArtifactBinding

# The control frame answering a worker about one publication.
PUBLICATION_FRAME_KIND = "publication_answer"

# The task-update field carrying a checkpoint's publication.
CHECKPOINT_FIELD = "artifact_published"
# The success field carrying an input preparation's materialization.
PREPARED_FIELD = "input_materialization"
PREPARED_KEY = "prepared"


class PublicationKind(StrEnum):
    """What a publication binds."""

    RESULT = "result"
    """A task's success: its result, its final artifacts and its merged children's."""
    CHECKPOINT = "checkpoint"
    """One checkpoint a running task published."""
    PREPARED = "prepared"
    """The request an input preparation materialized."""


class Publication(BaseModel):
    """One publication, as the worker reporting it and control both identify it."""

    model_config = ConfigDict(frozen=True)

    registration: str
    """The worker registration reporting it."""
    dispatch_id: str
    kind: PublicationKind
    key: str
    """What it publishes within its dispatch: the task, a checkpoint's
    ``<revision>/<name>``, or the prepared request."""
    candidate: str
    """The digest of the object it names first, which tells two of its versions
    apart."""

    @property
    def correlation(self) -> tuple[str, str, str, str, str]:
        return (
            self.registration,
            self.dispatch_id,
            self.kind.value,
            self.key,
            self.candidate,
        )

    @classmethod
    def of(
        cls,
        registration: str,
        dispatch_id: str | None,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> Self | None:
        """Return the publication a task event makes, or None when it names no stored
        object."""
        if dispatch_id is None:
            return None
        match event_type:
            case "TASK_SUCCEEDED":
                prepared = _reference(payload.get(PREPARED_FIELD), "reference")
                if prepared is not None:
                    kind, key, candidate = (
                        PublicationKind.PREPARED,
                        PREPARED_KEY,
                        prepared,
                    )
                elif reference := _reference(payload.get("result_reference")):
                    kind, key, candidate = PublicationKind.RESULT, task_id, reference
                else:
                    return None
            case "TASK_UPDATE":
                if (binding := _checkpoint(payload.get(CHECKPOINT_FIELD))) is None:
                    return None
                kind = PublicationKind.CHECKPOINT
                key = f"{binding.revision}/{binding.name}"
                candidate = binding.reference.content_digest
            case _:
                return None
        return cls(
            registration=registration,
            dispatch_id=dispatch_id,
            kind=kind,
            key=key,
            candidate=candidate,
        )


class PublicationDisposition(StrEnum):
    """Control's answer about one publication."""

    COMMITTED = "committed"
    """Its outcome is durable; nothing more is owed for it."""
    REJECTED = "rejected"
    """It binds nothing and never will, as when its dispatch no longer holds its task;
    nothing more is owed for it."""
    REFUSED = "refused"
    """It binds nothing now; its refusal says what the worker does before it reports
    again."""

    @property
    def final(self) -> bool:
        return self is not PublicationDisposition.REFUSED


class PublicationAnswer(BaseModel):
    """Control's answer about one publication."""

    publication: Publication
    disposition: PublicationDisposition
    refusal: StoreRefusal | None = None
    reason: str | None = None
    digests: list[str] = []
    """The objects a refusal concerns."""


def _reference(raw: Any, field: str | None = None) -> str | None:
    if isinstance(raw, dict) and field is not None:
        raw = raw.get(field)
    if raw is None:
        return None
    try:
        return ContentReference.model_validate(raw).content_digest
    except ValidationError:
        return None


def _checkpoint(raw: Any) -> ArtifactBinding | None:
    if not isinstance(raw, list) or len(raw) != 1:
        return None
    try:
        return ArtifactBinding.model_validate(raw[0])
    except ValidationError:
        return None


__all__ = [
    "CHECKPOINT_FIELD",
    "PREPARED_FIELD",
    "PREPARED_KEY",
    "PUBLICATION_FRAME_KIND",
    "Publication",
    "PublicationAnswer",
    "PublicationDisposition",
    "PublicationKind",
]
