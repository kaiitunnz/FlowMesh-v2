"""Private-state reference, sealed-generation, binding, and seal-verification rules."""

from pathlib import Path

import pytest

from shared.private_state import (
    ActivationPrivateStateReference,
    BundleProfile,
    OwnerFence,
    PrivateStateBinding,
    PrivateStateUnavailable,
    PrivateStateUnavailableReason,
    SealedComponent,
    StateBundleManifest,
    StateComponentKind,
    seal_component,
    verify_component,
)

_HOME = StateComponentKind.HARNESS_HOME_FS
_WORKSPACE = StateComponentKind.WORKSPACE_FS


def _reference(reference_id: str = "aps-one") -> ActivationPrivateStateReference:
    return ActivationPrivateStateReference(
        reference_id=reference_id, instance_id="wfl-1", activation_id="act-1"
    )


def _tree(root: Path, name: str, body: str) -> Path:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "file.txt").write_text(body)
    return target


def _components(
    root: Path, reference_id: str = "aps-one"
) -> tuple[SealedComponent, ...]:
    return (
        seal_component(
            _HOME, _tree(root, "home", "rollout"), reference_id=reference_id
        ),
        seal_component(
            _WORKSPACE, _tree(root, "work", "notes"), reference_id=reference_id
        ),
    )


def _manifest(
    root: Path, *, generation: int = 1, reference_id: str = "aps-one"
) -> StateBundleManifest:
    return StateBundleManifest(
        manifest_id="sbm-1",
        reference_id=reference_id,
        generation=generation,
        profile=BundleProfile.AGENT_HARNESS,
        quiescence_fence="fence-1",
        components=_components(root, reference_id),
    )


def test_sealed_generation_requires_every_profile_component(tmp_path: Path) -> None:
    home = seal_component(
        _HOME, _tree(tmp_path, "home", "rollout"), reference_id="aps-one"
    )
    with pytest.raises(ValueError, match="workspace_fs"):
        StateBundleManifest(
            manifest_id="sbm-1",
            reference_id="aps-one",
            generation=1,
            profile=BundleProfile.AGENT_HARNESS,
            quiescence_fence="fence-1",
            components=(home,),
        )


def test_sealed_generation_rejects_a_repeated_component_kind(tmp_path: Path) -> None:
    home, workspace = _components(tmp_path)
    with pytest.raises(ValueError, match="each component kind once"):
        StateBundleManifest(
            manifest_id="sbm-1",
            reference_id="aps-one",
            generation=1,
            profile=BundleProfile.AGENT_HARNESS,
            quiescence_fence="fence-1",
            components=(home, workspace, home),
        )


def test_sealed_generation_rejects_an_unregistered_schema_version(
    tmp_path: Path,
) -> None:
    home, workspace = _components(tmp_path)
    with pytest.raises(ValueError, match="schema version"):
        StateBundleManifest(
            manifest_id="sbm-1",
            reference_id="aps-one",
            generation=1,
            profile=BundleProfile.AGENT_HARNESS,
            quiescence_fence="fence-1",
            components=(home.model_copy(update={"schema_version": 99}), workspace),
        )


def test_an_unseeded_binding_carries_no_generation_to_restore() -> None:
    binding = PrivateStateBinding(reference=_reference())
    assert binding.generation == 0
    assert binding.manifest is None and binding.owner is None


def test_an_unseeded_binding_rejects_a_bound_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unseeded lineage"):
        PrivateStateBinding(
            reference=_reference(),
            generation=0,
            owner=OwnerFence(worker_id="wkr-1", incarnation=1),
        )


def test_a_bound_generation_requires_a_manifest_and_its_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed manifest and the holder"):
        PrivateStateBinding(reference=_reference(), generation=1)
    with pytest.raises(ValueError, match="sealed manifest and the holder"):
        PrivateStateBinding(
            reference=_reference(), generation=1, manifest=_manifest(tmp_path)
        )


def test_a_binding_never_mixes_generations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never mixes generations"):
        PrivateStateBinding(
            reference=_reference(),
            generation=2,
            manifest=_manifest(tmp_path, generation=1),
            owner=OwnerFence(worker_id="wkr-1", incarnation=1),
        )


def test_a_binding_never_restores_another_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="another lineage"):
        PrivateStateBinding(
            reference=_reference("aps-one"),
            generation=1,
            manifest=_manifest(tmp_path, reference_id="aps-two"),
            owner=OwnerFence(worker_id="wkr-1", incarnation=1),
        )


def test_a_seal_is_reproducible_across_holders(tmp_path: Path) -> None:
    first = _tree(tmp_path / "a", "home", "rollout")
    second = _tree(tmp_path / "b", "home", "rollout")
    assert (
        seal_component(_HOME, first, reference_id="aps-one").content_digest
        == seal_component(_HOME, second, reference_id="aps-one").content_digest
    )


def test_verification_rejects_an_edited_component(tmp_path: Path) -> None:
    home = _tree(tmp_path, "home", "rollout")
    sealed = seal_component(_HOME, home, reference_id="aps-one")
    (home / "file.txt").write_text("rewritten")
    with pytest.raises(PrivateStateUnavailable) as raised:
        verify_component(sealed, home, reference_id="aps-one")
    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISMATCH


def test_verification_rejects_an_added_file(tmp_path: Path) -> None:
    home = _tree(tmp_path, "home", "rollout")
    sealed = seal_component(_HOME, home, reference_id="aps-one")
    (home / "extra.txt").write_text("smuggled")
    with pytest.raises(PrivateStateUnavailable) as raised:
        verify_component(sealed, home, reference_id="aps-one")
    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISMATCH


def test_verification_rejects_an_unmaterialized_component(tmp_path: Path) -> None:
    sealed = seal_component(
        _HOME, _tree(tmp_path, "home", "rollout"), reference_id="aps-one"
    )
    with pytest.raises(PrivateStateUnavailable) as raised:
        verify_component(sealed, tmp_path / "absent", reference_id="aps-one")
    assert raised.value.reason is PrivateStateUnavailableReason.COMPONENT_MISSING


def test_sealing_refuses_a_link_out_of_the_private_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not ours")
    home = _tree(tmp_path, "home", "rollout")
    (home / "escape").symlink_to(outside / "secret.txt")
    with pytest.raises(PrivateStateUnavailable) as raised:
        seal_component(_HOME, home, reference_id="aps-one")
    assert raised.value.reason is PrivateStateUnavailableReason.CONTAINMENT_VIOLATION
