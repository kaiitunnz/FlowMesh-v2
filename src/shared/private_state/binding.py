"""What a continuation must supply before it may resume against private state."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from .manifest import StateBundleManifest
from .reference import ActivationPrivateStateReference


class PrivateStateRecoveryMode(StrEnum):
    """What a continuation declares should happen when its generation is unavailable.

    A mode is listed once the fabric honors it, so a binding never declares recovery
    behavior that recovery does not apply.
    """

    # The bound generation is irreplaceable: recovery fails closed rather than
    # substituting a fresh or partial one.
    OWNER_LOCAL = "owner_local"


class OwnerFence(BaseModel):
    """The single holder incarnation that can supply an owner-local generation.

    A worker registration mints a new identity per incarnation, so a restarted owner
    matches neither half of this fence.
    """

    model_config = ConfigDict(frozen=True)

    worker_id: str
    incarnation: int


class PrivateStateBinding(BaseModel):
    """The exact reference, generation, and recovery mode a continuation resumes on.

    Generation 0 binds a lineage with nothing to restore: any eligible holder may
    create its first generation. A later generation names the sealed manifest a holder
    must supply in full, and its ``owner`` is the feasible holder set — the scheduler's
    hard placement constraint while the seal is local to that holder.
    """

    model_config = ConfigDict(frozen=True)

    reference: ActivationPrivateStateReference
    generation: int = 0
    recovery: PrivateStateRecoveryMode = PrivateStateRecoveryMode.OWNER_LOCAL
    manifest: StateBundleManifest | None = None
    owner: OwnerFence | None = None

    @model_validator(mode="after")
    def _generation_is_bound(self) -> Self:
        if self.generation < 0:
            raise ValueError("a state generation is never negative")
        if self.generation == 0:
            if self.manifest is not None or self.owner is not None:
                raise ValueError("an unseeded lineage binds no generation to restore")
            return self
        if self.manifest is None or self.owner is None:
            raise ValueError(
                f"generation {self.generation} of {self.reference.reference_id} binds "
                "a sealed manifest and the holder that sealed it"
            )
        if self.manifest.generation != self.generation:
            raise ValueError(
                "a binding never mixes generations: manifest "
                f"{self.manifest.generation} against binding {self.generation}"
            )
        if self.manifest.reference_id != self.reference.reference_id:
            raise ValueError("a binding never restores another lineage's manifest")
        return self
