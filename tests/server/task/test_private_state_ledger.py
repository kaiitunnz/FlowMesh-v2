"""Lineage binding, exclusive write fencing, and seal acceptance in the ledger."""

from pathlib import Path

import pytest

from server.orchestration.private_state import PrivateStateLedger
from server.orchestration.state import PrivateStateLineage
from shared.private_state import (
    BundleProfile,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
    StateBundleManifest,
    StateComponentKind,
    seal_component,
)


def _manifest(
    tmp_path: Path, reference_id: str, generation: int
) -> StateBundleManifest:
    components = []
    for kind, name in (
        (StateComponentKind.HARNESS_HOME_FS, "home"),
        (StateComponentKind.WORKSPACE_FS, "work"),
    ):
        root = tmp_path / f"{name}{generation}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "file.txt").write_text(f"{name}-{generation}")
        components.append(seal_component(kind, root, reference_id=reference_id))
    return StateBundleManifest(
        manifest_id=f"sbm-{generation}",
        reference_id=reference_id,
        generation=generation,
        profile=BundleProfile.AGENT_HARNESS,
        quiescence_fence=f"fence-{generation}",
        components=tuple(components),
    )


def _seeded(tmp_path: Path) -> tuple[PrivateStateLedger, str]:
    ledger = PrivateStateLedger()
    binding = ledger.ensure("act-1", "wfl-1")
    attachment = ledger.attach("act-1", "wkr-1", 3)
    ledger.seal(
        "act-1",
        _manifest(tmp_path, binding.reference.reference_id, 1),
        attachment.write_epoch,
    )
    return ledger, binding.reference.reference_id


def test_a_fresh_lineage_binds_no_generation_and_pins_no_holder() -> None:
    ledger = PrivateStateLedger()
    binding = ledger.ensure("act-1", "wfl-1")
    assert binding.generation == 0
    assert ledger.owner("act-1") is None


def test_the_same_activation_keeps_one_reference() -> None:
    ledger = PrivateStateLedger()
    first = ledger.ensure("act-1", "wfl-1")
    assert ledger.ensure("act-1", "wfl-1").reference == first.reference


def test_co_located_activations_get_separate_lineages() -> None:
    ledger = PrivateStateLedger()
    one = ledger.ensure("act-1", "wfl-1").reference.reference_id
    two = ledger.ensure("act-2", "wfl-1").reference.reference_id
    assert one != two


def test_a_seal_binds_the_generation_to_the_holder_that_produced_it(
    tmp_path: Path,
) -> None:
    ledger, _ = _seeded(tmp_path)
    binding = ledger.lineages()[0].binding
    assert binding is not None and binding.generation == 1
    assert binding.owner is not None
    assert (binding.owner.worker_id, binding.owner.incarnation) == ("wkr-1", 3)


def test_each_grant_supersedes_the_previous_write_epoch() -> None:
    ledger = PrivateStateLedger()
    ledger.ensure("act-1", "wfl-1")
    first = ledger.attach("act-1", "wkr-1", 3)
    second = ledger.attach("act-1", "wkr-1", 3)
    assert second.write_epoch > first.write_epoch


def test_a_superseded_holder_cannot_seal(tmp_path: Path) -> None:
    ledger = PrivateStateLedger()
    binding = ledger.ensure("act-1", "wfl-1")
    stale = ledger.attach("act-1", "wkr-1", 3)
    ledger.attach("act-1", "wkr-1", 3)
    with pytest.raises(PrivateStateUnavailable) as raised:
        ledger.seal(
            "act-1",
            _manifest(tmp_path, binding.reference.reference_id, 1),
            stale.write_epoch,
        )
    assert raised.value.reason is PrivateStateUnavailableReason.STALE_EPOCH


def test_a_seal_never_adopts_another_lineages_generation(tmp_path: Path) -> None:
    ledger = PrivateStateLedger()
    ledger.ensure("act-1", "wfl-1")
    attachment = ledger.attach("act-1", "wkr-1", 3)
    with pytest.raises(PrivateStateUnavailable) as raised:
        ledger.seal(
            "act-1", _manifest(tmp_path, "aps-foreign", 1), attachment.write_epoch
        )
    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISMATCH


def test_another_worker_cannot_attach_to_a_bound_generation(tmp_path: Path) -> None:
    ledger, _ = _seeded(tmp_path)
    with pytest.raises(PrivateStateUnavailable) as raised:
        ledger.attach("act-1", "wkr-2", 3)
    assert raised.value.reason is PrivateStateUnavailableReason.OWNER_LOST


def test_a_restarted_owner_cannot_attach_to_a_bound_generation(tmp_path: Path) -> None:
    ledger, _ = _seeded(tmp_path)
    with pytest.raises(PrivateStateUnavailable) as raised:
        ledger.attach("act-1", "wkr-1", 4)
    assert raised.value.reason is PrivateStateUnavailableReason.INCARNATION_MISMATCH


def test_release_drops_write_authority_but_keeps_the_binding(tmp_path: Path) -> None:
    ledger, _ = _seeded(tmp_path)
    ledger.release("act-1")
    lineage = ledger.lineages()[0]
    assert lineage.attachment is None
    assert lineage.binding.generation == 1


def test_a_lineage_survives_a_snapshot_round_trip(tmp_path: Path) -> None:
    ledger, reference_id = _seeded(tmp_path)
    restored = PrivateStateLedger(
        [
            PrivateStateLineage.model_validate_json(lineage.model_dump_json())
            for lineage in ledger.lineages()
        ]
    )
    binding = restored.lineages()[0].binding
    assert binding is not None
    assert binding.reference.reference_id == reference_id
    assert binding.generation == 1
    assert restored.owner("act-1") == ledger.owner("act-1")


def test_a_cancelled_activation_is_granted_no_new_write_authority() -> None:
    """A dispatch still in flight at cancellation cannot re-take the released write."""
    ledger = PrivateStateLedger()
    ledger.ensure("act-1", "wfl-1")
    ledger.attach("act-1", "wkr-1", 3)
    ledger.release("act-1")
    assert ledger.lineages()[0].attachment is None
