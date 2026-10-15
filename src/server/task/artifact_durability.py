"""Whether a task's accepted artifacts are in the shared store yet.

A success may announce its final artifacts while they are still only in its worker's
custody. The task is DONE at once; its artifacts' upload is a separate obligation,
owed by the dispatch that reported the success, which holds its workflow open until
it settles. It settles durable when the worker reports every object stored, or when
control finds them all in the store; it settles lost for the artifacts control finds
missing once their owner is gone. A lost artifact's siblings keep their own answer.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from shared.tasks.artifact_binding import ArtifactBinding
from shared.tasks.artifact_upload import ArtifactDurability, UploadReport

ARTIFACT_LOST = "artifact_lost"
ARTIFACT_PENDING = "artifact_pending"


class UploadStateUnavailable(RuntimeError):
    """A task's upload state could not be read from the control store."""


class ArtifactLoss(BaseModel):
    """Why an accepted artifact is lost."""

    model_config = ConfigDict(frozen=True)

    code: str = ARTIFACT_LOST
    message: str


class ArtifactUpload(BaseModel):
    """One task's owed upload of the final artifacts its success accepted."""

    model_config = ConfigDict(frozen=True)

    state: ArtifactDurability = ArtifactDurability.PENDING
    # The dispatch whose success announced the artifacts, the task it ran, under whose
    # store access the upload writes, and the worker registration that holds them in
    # custody; only they complete the upload.
    dispatch_id: str
    dispatch_task_id: str
    owner_id: str
    # The accepted artifacts found missing, by revision and name, once settled lost.
    lost: dict[str, ArtifactLoss] = Field(default_factory=dict)
    # When it settled, as the control plane's wall clock.
    settled_ts: float | None = None

    @property
    def pending(self) -> bool:
        return self.state is ArtifactDurability.PENDING

    def settle(self, lost: dict[str, ArtifactLoss], ts: float) -> Self:
        """Return the upload settled, lost for the artifacts in ``lost`` and durable
        for the rest."""
        return self.model_copy(
            update={
                "state": (
                    ArtifactDurability.LOST if lost else ArtifactDurability.DURABLE
                ),
                "lost": lost,
                "settled_ts": ts,
            }
        )

    def durability(
        self, binding: ArtifactBinding
    ) -> tuple[ArtifactDurability, ArtifactLoss | None]:
        """Return where one of the task's ``final`` artifacts stands; an archive
        stands with the tree it views."""
        if self.state is not ArtifactDurability.LOST:
            return self.state, None
        loss = self.lost.get(binding_key(binding)) or (
            self.lost.get(binding_key(binding, binding.of))
            if binding.of is not None
            else None
        )
        return (
            (ArtifactDurability.LOST, loss)
            if loss is not None
            else (ArtifactDurability.DURABLE, None)
        )

    def failure(self, task_id: str) -> str:
        """Return the workflow failure this upload's loss is reported as."""
        names = ", ".join(sorted(key.replace("\0", "/") for key in self.lost))
        return f"{ARTIFACT_LOST}: task {task_id} lost {names}"


class EarlyUploadReceipt(BaseModel):
    """An upload report that arrived while the success it follows was in flight."""

    model_config = ConfigDict(frozen=True)

    dispatch_id: str
    report: UploadReport
    error: str | None = None


def lost_read(task_id: str, binding: ArtifactBinding) -> str:
    """Return why a reader of one lost artifact cannot run."""
    return f"{ARTIFACT_LOST}: task {task_id} lost {binding.revision}/{binding.name}"


def binding_key(binding: ArtifactBinding, name: str | None = None) -> str:
    """Return the key a binding's loss is recorded under: its revision and name."""
    return f"{binding.revision}\0{name or binding.name}"
