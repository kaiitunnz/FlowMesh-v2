"""The engine's authority over activation-private state lineages.

The orchestration ledger owns a lineage's identity, its bound generation, and which
holder may write it. That authority is over state access alone; capacity admission is
the separate concern a service claim carries.
"""

from shared.private_state import (
    ActivationPrivateStateReference,
    BundleProfile,
    OwnerFence,
    PrivateStateAttachment,
    PrivateStateBinding,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
    StateBundleManifest,
)
from shared.utils.ids import new_private_state_reference_id, new_state_attachment_id

from .state import PrivateStateLineage


class PrivateStateLedger:
    """Tracks each private-state lineage's binding and its one writable attachment."""

    def __init__(self, lineages: list[PrivateStateLineage] | None = None) -> None:
        self._lineages: dict[str, PrivateStateLineage] = {
            lineage.binding.reference.activation_id: lineage
            for lineage in (lineages or [])
        }

    def lineages(self) -> list[PrivateStateLineage]:
        return list(self._lineages.values())

    def holders(self) -> frozenset[str]:
        """The workers that hold a sealed generation of any lineage."""
        return frozenset(
            lineage.binding.owner.worker_id
            for lineage in self._lineages.values()
            if lineage.binding.owner is not None
        )

    def generation(self, activation_id: str) -> int | None:
        """The generation a lineage is bound to, or None for an unknown lineage."""
        lineage = self._lineages.get(activation_id)
        return lineage.binding.generation if lineage else None

    def owner(self, activation_id: str) -> OwnerFence | None:
        lineage = self._lineages.get(activation_id)
        return lineage.binding.owner if lineage else None

    def ensure(
        self,
        activation_id: str,
        instance_id: str,
        *,
        tenant: str | None = None,
        profile: BundleProfile = BundleProfile.AGENT_HARNESS,
    ) -> PrivateStateBinding:
        """The lineage's binding, minting an unseeded one on first use."""
        if (lineage := self._lineages.get(activation_id)) is not None:
            return lineage.binding
        binding = PrivateStateBinding(
            reference=ActivationPrivateStateReference(
                reference_id=new_private_state_reference_id(),
                instance_id=instance_id,
                activation_id=activation_id,
                tenant=tenant,
                profile=profile,
            )
        )
        self._lineages[activation_id] = PrivateStateLineage(binding=binding)
        return binding

    def attach(
        self, activation_id: str, worker_id: str, incarnation: int
    ) -> PrivateStateAttachment:
        """Grant one holder exclusive write authority, superseding any prior epoch.

        A holder that does not satisfy a bound generation's owner fence is refused here
        rather than handed a home it cannot supply.
        """
        lineage = self._lineages[activation_id]
        binding = lineage.binding
        if (owner := binding.owner) is not None:
            if owner.worker_id != worker_id:
                raise PrivateStateUnavailable(
                    PrivateStateUnavailableReason.OWNER_LOST,
                    f"generation {binding.generation} is held by {owner.worker_id}",
                    reference_id=binding.reference.reference_id,
                )
            if owner.incarnation != incarnation:
                raise PrivateStateUnavailable(
                    PrivateStateUnavailableReason.INCARNATION_MISMATCH,
                    f"{worker_id} restarted since generation {binding.generation}",
                    reference_id=binding.reference.reference_id,
                )
        lineage.write_epoch += 1
        lineage.attachment = PrivateStateAttachment(
            attachment_id=new_state_attachment_id(),
            reference_id=binding.reference.reference_id,
            generation=binding.generation,
            worker_id=worker_id,
            incarnation=incarnation,
            write_epoch=lineage.write_epoch,
        )
        return lineage.attachment

    def seal(
        self, activation_id: str, manifest: StateBundleManifest, write_epoch: int
    ) -> None:
        """Advance the lineage to the generation its live attachment sealed.

        A superseded epoch's seal is refused: its holder lost the write, so accepting it
        would bind a generation another holder has already moved past.
        """
        lineage = self._lineages[activation_id]
        attachment = lineage.attachment
        reference = lineage.binding.reference
        if attachment is None or attachment.write_epoch != write_epoch:
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.STALE_EPOCH,
                f"write epoch {write_epoch} no longer holds {reference.reference_id}",
                reference_id=reference.reference_id,
            )
        if manifest.reference_id != reference.reference_id:
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.COMPONENT_MISMATCH,
                "a seal never adopts another lineage's generation",
                reference_id=reference.reference_id,
            )
        lineage.binding = PrivateStateBinding(
            reference=reference,
            generation=manifest.generation,
            recovery=lineage.binding.recovery,
            manifest=manifest,
            owner=OwnerFence(
                worker_id=attachment.worker_id, incarnation=attachment.incarnation
            ),
        )

    def release(self, activation_id: str) -> None:
        """Drop the live attachment, so no holder retains write authority."""
        if (lineage := self._lineages.get(activation_id)) is not None:
            lineage.attachment = None
