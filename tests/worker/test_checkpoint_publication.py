"""A trainer's saved checkpoints publish as revisions while it runs, from snapshots its
rotation cannot remove."""

import logging
from pathlib import Path
from typing import Any

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from shared.content import ContentStoreError, SharedFilesystemObjectStore, TreeManifest
from shared.tasks.artifact_binding import ArtifactBinding
from worker.content.artifacts import ArtifactPublisher, spool_checkpoint, spooled_steps
from worker.content.checkpoints import CheckpointPublishFailed, CheckpointUploader
from worker.executors.utils.checkpoint_spool import CheckpointSpool


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    (checkpoint / "sub").mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(f"model-{step}".encode())
    (checkpoint / "sub" / "optimizer.pt").write_bytes(b"optimizer")
    return checkpoint


def test_a_snapshot_survives_the_trainer_rotating_its_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", 10)
    spool = tmp_path / "spool"

    snapshot = spool_checkpoint(checkpoint, spool, 10)
    for path in sorted(checkpoint.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()

    assert snapshot is not None
    assert (snapshot / "checkpoint-10" / "model.bin").read_bytes() == b"model-10"
    assert spooled_steps(spool) == [(10, snapshot)]
    assert spool_checkpoint(checkpoint, spool, 10) is None


def test_a_half_written_snapshot_is_not_spooled(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    (spool / ".step-3.partial" / "checkpoint-3").mkdir(parents=True)

    assert spooled_steps(spool) == []


def _uploader(
    tmp_path: Path, store: Any, reports: list[ArtifactBinding]
) -> CheckpointUploader:
    return CheckpointUploader(
        ArtifactPublisher(store, "org"),
        tmp_path / "spool",
        reports.append,
        logging.getLogger("test"),
        poll_sec=0.01,
    )


def test_each_spooled_checkpoint_publishes_once_as_its_step(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "cas")
    reports: list[ArtifactBinding] = []
    uploader = _uploader(tmp_path, store, reports)
    uploader.start()
    for step in (20, 10):
        spool_checkpoint(
            _checkpoint(tmp_path / "checkpoints", step), tmp_path / "spool", step
        )

    published = uploader.drain()

    assert [(b.name, b.revision) for b in published] == [
        ("checkpoints", "step-10"),
        ("checkpoints", "step-20"),
    ]
    assert sorted(b.revision for b in reports) == ["step-10", "step-20"]
    manifest = TreeManifest.decode(store.hydrate(published[0].reference))
    assert [m.path for m in manifest.members] == [
        "checkpoint-10/model.bin",
        "checkpoint-10/sub/optimizer.pt",
    ]
    assert not (tmp_path / "spool").exists()


def test_a_checkpoint_the_store_refuses_fails_the_drain(tmp_path: Path) -> None:
    class _Refusing(SharedFilesystemObjectStore):
        def write_file(self, scope: str, path: Path, **kwargs: Any) -> Any:
            raise ContentStoreError("store unreachable")

    uploader = _uploader(tmp_path, _Refusing(tmp_path / "cas"), [])
    spool_checkpoint(_checkpoint(tmp_path / "checkpoints", 1), tmp_path / "spool", 1)
    uploader.start()

    with pytest.raises(CheckpointPublishFailed):
        uploader.drain()


def test_an_aborted_upload_leaves_no_spool(tmp_path: Path) -> None:
    uploader = _uploader(tmp_path, SharedFilesystemObjectStore(tmp_path / "cas"), [])
    uploader.start()
    (tmp_path / "spool" / ".step-1.partial").mkdir(parents=True)

    uploader.abort()

    assert not (tmp_path / "spool").exists()


@pytest.mark.parametrize("zero", [True, False])
def test_only_the_first_rank_snapshots_a_save(tmp_path: Path, zero: bool) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, 7)
    state = TrainerState(global_step=7)
    state.is_world_process_zero = zero

    CheckpointSpool(tmp_path / "spool").on_save(
        TrainingArguments(output_dir=root.as_posix(), report_to=[]),
        state,
        TrainerControl(),
    )

    assert bool(spooled_steps(tmp_path / "spool")) is zero
