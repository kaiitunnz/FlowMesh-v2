"""Materializing an activation's private state under its attachment.

The holder keeps each lineage in its own opaque root, restores only the generation the
binding names, and seals the required components together at the step's quiescence
fence. A generation that cannot be supplied in full fails closed rather than resuming
against a fresh or partial home.
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
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(_PRIVATE_MODE)
        lineage = self._root / reference_id
        lineage.mkdir(exist_ok=True)
        lineage.chmod(_PRIVATE_MODE)
        return lineage

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
        self._claim_epoch(lineage, attachment)
        components = {}
        for kind in sorted(required_components(binding.reference.profile)):
            path = lineage / kind.value
            path.mkdir(exist_ok=True)
            path.chmod(_PRIVATE_MODE)
            components[kind] = path
        if binding.manifest is None:
            self._adopt_legacy_home(
                components[StateComponentKind.HARNESS_HOME_FS], legacy_home
            )
        else:
            self._restore(binding.manifest, components, reference_id)
        return MaterializedState(
            reference_id, binding.generation, binding.reference.profile, components
        )

    def seal(
        self, state: MaterializedState, attachment: PrivateStateAttachment
    ) -> PrivateStateSealReport:
        """Seal every component of the lineage as the next coherent generation."""
        lineage = self._root / state.reference_id
        self._verify_epoch(lineage, attachment)
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

    @staticmethod
    def _claim_epoch(lineage: Path, attachment: PrivateStateAttachment) -> None:
        marker = lineage / _EPOCH_FILE
        held = PrivateStateHolder._held_epoch(marker)
        if held is not None and held > attachment.write_epoch:
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.STALE_EPOCH,
                f"write epoch {attachment.write_epoch} is superseded by {held}",
                reference_id=attachment.reference_id,
            )
        marker.write_text(str(attachment.write_epoch))
        marker.chmod(0o600)

    @staticmethod
    def _verify_epoch(lineage: Path, attachment: PrivateStateAttachment) -> None:
        held = PrivateStateHolder._held_epoch(lineage / _EPOCH_FILE)
        if held != attachment.write_epoch:
            raise PrivateStateUnavailable(
                PrivateStateUnavailableReason.STALE_EPOCH,
                f"write epoch {attachment.write_epoch} no longer holds the lineage",
                reference_id=attachment.reference_id,
            )

    @staticmethod
    def _held_epoch(marker: Path) -> int | None:
        if not marker.is_file():
            return None
        raw = marker.read_text().strip()
        return int(raw) if raw.isdigit() else None
