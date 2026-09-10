"""Materializing, fencing, and sealing an activation's private state on a holder."""

import shutil
import stat
from pathlib import Path

import pytest

from shared.private_state import (
    ActivationPrivateStateReference,
    OwnerFence,
    PrivateStateAttachment,
    PrivateStateBinding,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
    StateComponentKind,
)
from shared.utils.ids import new_private_state_reference_id
from worker.private_state import MaterializedState, PrivateStateHolder


def _binding(reference_id: str | None = None) -> PrivateStateBinding:
    return PrivateStateBinding(
        reference=ActivationPrivateStateReference(
            reference_id=reference_id or new_private_state_reference_id(),
            instance_id="wfl-1",
            activation_id="act-1",
        )
    )


def _attachment(
    binding: PrivateStateBinding, *, write_epoch: int = 1, generation: int | None = None
) -> PrivateStateAttachment:
    return PrivateStateAttachment(
        attachment_id=f"psa-{write_epoch}",
        reference_id=binding.reference.reference_id,
        generation=binding.generation if generation is None else generation,
        worker_id="wkr-1",
        incarnation=1,
        write_epoch=write_epoch,
    )


def _advance(
    holder: PrivateStateHolder, binding: PrivateStateBinding, epoch: int
) -> tuple[PrivateStateBinding, MaterializedState]:
    """Run one step: materialize, write, seal, and bind the sealed generation."""
    attachment = _attachment(binding, write_epoch=epoch)
    state = holder.open(binding, attachment)
    (state.harness_home / "rollout.jsonl").write_text(f"turn-{epoch}")
    report = holder.seal(state, attachment)
    return (
        PrivateStateBinding(
            reference=binding.reference,
            generation=report.manifest.generation,
            manifest=report.manifest,
            owner=OwnerFence(worker_id="wkr-1", incarnation=1),
        ),
        state,
    )


def test_an_unseeded_lineage_materializes_every_profile_component(
    tmp_path: Path,
) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()

    state = holder.open(binding, _attachment(binding))

    assert set(state.components) == {
        StateComponentKind.HARNESS_HOME_FS,
        StateComponentKind.WORKSPACE_FS,
    }
    assert state.harness_home.is_dir() and state.workspace.is_dir()


def test_a_lineage_root_is_private_to_its_holder(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()

    state = holder.open(binding, _attachment(binding))

    for path in (tmp_path, state.harness_home.parent, state.harness_home):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_co_located_activations_materialize_under_separate_roots(
    tmp_path: Path,
) -> None:
    holder = PrivateStateHolder(tmp_path)
    first_binding, second_binding = _binding(), _binding()
    one = holder.open(first_binding, _attachment(first_binding))
    other = holder.open(second_binding, _attachment(second_binding))

    assert one.harness_home != other.harness_home
    assert not one.harness_home.is_relative_to(other.harness_home.parent)


def test_a_sealed_generation_restores_on_the_same_holder(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    bound, _ = _advance(holder, _binding(), 1)

    resumed = holder.open(bound, _attachment(bound, write_epoch=2))

    assert (resumed.harness_home / "rollout.jsonl").read_text() == "turn-1"
    assert resumed.generation == 1


def test_each_step_seals_the_next_generation(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    first, _ = _advance(holder, _binding(), 1)
    second, _ = _advance(holder, first, 2)

    assert (first.generation, second.generation) == (1, 2)
    assert second.manifest is not None
    assert len(second.manifest.components) == 2


def test_a_resume_refuses_an_edited_component(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    bound, state = _advance(holder, _binding(), 1)
    (state.harness_home / "rollout.jsonl").write_text("tampered")

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(bound, _attachment(bound, write_epoch=2))

    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISMATCH


def test_a_resume_refuses_a_generation_missing_a_required_component(
    tmp_path: Path,
) -> None:
    holder = PrivateStateHolder(tmp_path)
    bound, _ = _advance(holder, _binding(), 1)
    assert bound.manifest is not None
    partial = bound.manifest.model_copy(
        update={"components": bound.manifest.components[:1]}
    )
    mixed = PrivateStateBinding.model_construct(
        reference=bound.reference,
        generation=bound.generation,
        recovery=bound.recovery,
        manifest=partial,
        owner=bound.owner,
    )

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(mixed, _attachment(mixed, write_epoch=2))

    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISSING


def test_a_superseded_epoch_cannot_materialize(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()
    holder.open(binding, _attachment(binding, write_epoch=4))

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(binding, _attachment(binding, write_epoch=3))

    assert raised.value.reason is PrivateStateUnavailableReason.STALE_EPOCH


def test_a_superseded_epoch_cannot_seal(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()
    stale_attachment = _attachment(binding, write_epoch=1)
    state = holder.open(binding, stale_attachment)
    holder.open(binding, _attachment(binding, write_epoch=2))

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.seal(state, stale_attachment)

    assert raised.value.reason is PrivateStateUnavailableReason.STALE_EPOCH


def test_an_attachment_for_another_generation_is_refused(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(binding, _attachment(binding, generation=3))

    assert raised.value.reason is PrivateStateUnavailableReason.STALE_EPOCH


def test_an_attachment_for_another_lineage_is_refused(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding()

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(binding, _attachment(_binding()))

    assert raised.value.reason is PrivateStateUnavailableReason.STALE_EPOCH


def test_a_legacy_harness_home_drains_into_the_first_generation(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "rollout.jsonl").write_text("pre-existing")
    holder = PrivateStateHolder(tmp_path / "private")
    binding = _binding()

    state = holder.open(binding, _attachment(binding), legacy_home=legacy)

    assert (state.harness_home / "rollout.jsonl").read_text() == "pre-existing"


def test_a_legacy_home_never_overwrites_a_restored_generation(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "rollout.jsonl").write_text("pre-existing")
    holder = PrivateStateHolder(tmp_path / "private")
    bound, _ = _advance(holder, _binding(), 1)

    resumed = holder.open(bound, _attachment(bound, write_epoch=2), legacy_home=legacy)

    assert (resumed.harness_home / "rollout.jsonl").read_text() == "turn-1"


def test_a_non_opaque_reference_never_reaches_the_filesystem(tmp_path: Path) -> None:
    holder = PrivateStateHolder(tmp_path)
    binding = _binding("../escape")

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(binding, _attachment(binding))

    assert raised.value.reason is PrivateStateUnavailableReason.CONTAINMENT_VIOLATION


def test_a_resume_refuses_a_component_removed_from_under_the_holder(
    tmp_path: Path,
) -> None:
    """A missing component is refused, not repaired into an empty one."""
    holder = PrivateStateHolder(tmp_path)
    bound, state = _advance(holder, _binding(), 1)
    shutil.rmtree(state.workspace)

    with pytest.raises(PrivateStateUnavailable) as raised:
        holder.open(bound, _attachment(bound, write_epoch=2))

    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISSING
