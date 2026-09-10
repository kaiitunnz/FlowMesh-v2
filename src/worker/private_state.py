"""Materializing an activation's private state under its attachment.

The holder keeps each lineage in its own opaque root, restores only the generation the
binding names, and seals the required components together at the step's quiescence
fence. A generation that cannot be supplied in full fails closed rather than resuming
against a fresh or partial home.

A step that ends without sealing — a failure part way through a turn — leaves the tree
ahead of the generation the binding still names, so the next attempt fails closed at
verification rather than resuming from a point no fence covers.

A lineage root outlives the activation that owned it: the holder reaps nothing on its
own, because a completed step is not always the episode's terminal one and the root is
shared with the workers co-located on its node. Its contents are private to the holder
at 0700 and unreachable once the holder's incarnation ends, since no later incarnation
satisfies an owner fence.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from shared.private_state import (
    BundleProfile,
    PrivateStateAttachment,
    PrivateStateBinding,
    PrivateStateSealReport,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
    StateBundleManifest,
    StateComponentKind,
    required_components,
    seal_component,
    verify_component,
)
from shared.utils.ids import new_state_bundle_manifest_id

_PRIVATE_MODE = 0o700
_EPOCH_FILE = ".attachment"
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class MaterializedState:
    """The component roots one holder may use for a bound generation."""

    reference_id: str
    generation: int
    profile: BundleProfile
    components: dict[StateComponentKind, Path]

    @property
    def harness_home(self) -> Path:
        return self.components[StateComponentKind.HARNESS_HOME_FS]

    @property
    def workspace(self) -> Path:
        return self.components[StateComponentKind.WORKSPACE_FS]


class PrivateStateHolder:
    """A worker's private-state root and the attachments it honors."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _lineage_root(self, binding: PrivateStateBinding) -> Path:
        reference_id = binding.reference.reference_id
        if not _OPAQUE_ID.match(reference_id):
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.CONTAINMENT_VIOLATION,
                "a state reference is an opaque identifier",
                reference_id=reference_id,
            )
        _private_dir(self._root, parents=True)
        return _private_dir(self._root / reference_id)

    def open(
        self,
        binding: PrivateStateBinding,
        attachment: PrivateStateAttachment,
        *,
        legacy_home: Path | None = None,
    ) -> MaterializedState:
        """Materialize the bound generation for the attachment's holder.

        Restoring verifies every required component against the sealed generation, so a
        component that was edited, lost, or never sealed refuses the resume. An unseeded
        lineage starts empty, adopting a reachable legacy harness home into its first
        generation.
        """
        reference_id = binding.reference.reference_id
        if (
            attachment.reference_id != reference_id
            or attachment.generation != binding.generation
        ):
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.STALE_EPOCH,
                "the attachment does not authorize the bound generation",
                reference_id=reference_id,
            )
        lineage = self._lineage_root(binding)
        _claim_epoch(lineage, attachment)
        kinds = sorted(required_components(binding.reference.profile))
        if binding.manifest is None:
            components = {kind: _private_dir(lineage / kind.value) for kind in kinds}
            self._adopt_legacy_home(
                components[StateComponentKind.HARNESS_HOME_FS], legacy_home
            )
        else:
            # A bound generation is verified as it stands: creating a component first
            # would repair away a removed one instead of refusing the resume.
            components = {kind: lineage / kind.value for kind in kinds}
            self._restore(binding.manifest, components, reference_id)
            for path in components.values():
                path.chmod(_PRIVATE_MODE)
        return MaterializedState(
            reference_id, binding.generation, binding.reference.profile, components
        )

    def seal(
        self, state: MaterializedState, attachment: PrivateStateAttachment
    ) -> PrivateStateSealReport:
        """Seal every component of the lineage as the next coherent generation."""
        lineage = self._root / state.reference_id
        _verify_epoch(lineage, attachment)
        generation = state.generation + 1
        components = tuple(
            seal_component(kind, path, reference_id=state.reference_id)
            for kind, path in sorted(state.components.items())
        )
        manifest = StateBundleManifest(
            manifest_id=new_state_bundle_manifest_id(),
            reference_id=state.reference_id,
            generation=generation,
            profile=state.profile,
            quiescence_fence=f"{attachment.attachment_id}:{generation}",
            components=components,
        )
        return PrivateStateSealReport(
            manifest=manifest, write_epoch=attachment.write_epoch
        )

    @staticmethod
    def _restore(
        manifest: StateBundleManifest,
        components: dict[StateComponentKind, Path],
        reference_id: str,
    ) -> None:
        for kind, path in components.items():
            sealed = manifest.component(kind)
            if sealed is None:
                raise PrivateStateUnavailable(
                    PrivateStateUnavailableReason.COMPONENT_MISSING,
                    f"the sealed generation omits {kind.value}",
                    reference_id=reference_id,
                )
            verify_component(sealed, path, reference_id=reference_id)

    @staticmethod
    def _adopt_legacy_home(home: Path, legacy_home: Path | None) -> None:
        if legacy_home is None or not legacy_home.is_dir():
            return
        if any(home.iterdir()):
            return
        shutil.copytree(legacy_home, home, dirs_exist_ok=True, symlinks=False)


def _private_dir(path: Path, *, parents: bool = False) -> Path:
    """Create or adopt a directory readable only by the holder's own user."""
    path.mkdir(parents=parents, exist_ok=True)
    path.chmod(_PRIVATE_MODE)
    return path


def _held_epoch(lineage: Path) -> int | None:
    marker = lineage / _EPOCH_FILE
    if not marker.is_file():
        return None
    raw = marker.read_text().strip()
    return int(raw) if raw.isdigit() else None


def _claim_epoch(lineage: Path, attachment: PrivateStateAttachment) -> None:
    """Take the lineage's write epoch, refusing one a later grant superseded."""
    held = _held_epoch(lineage)
    if held is not None and held > attachment.write_epoch:
        raise PrivateStateUnavailable(
            PrivateStateUnavailableReason.STALE_EPOCH,
            f"write epoch {attachment.write_epoch} is superseded by {held}",
            reference_id=attachment.reference_id,
        )
    marker = lineage / _EPOCH_FILE
    marker.write_text(str(attachment.write_epoch))
    marker.chmod(0o600)


def _verify_epoch(lineage: Path, attachment: PrivateStateAttachment) -> None:
    """Confirm the holder still owns the write before its seal counts."""
    if _held_epoch(lineage) != attachment.write_epoch:
        raise PrivateStateUnavailable(
            PrivateStateUnavailableReason.STALE_EPOCH,
            f"write epoch {attachment.write_epoch} does not hold the lineage",
            reference_id=attachment.reference_id,
        )
