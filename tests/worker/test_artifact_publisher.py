"""A task's artifacts publish into the store as files, trees and archive views before
any binding names them, and only what the task left in place publishes."""

import gzip
import io
import os
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from shared.content import (
    SharedFilesystemObjectStore,
    TreeManifest,
    TreeMember,
    content_digest,
)
from shared.content.archive import MemberReader, TreeArchive
from shared.schemas.artifact import ArtifactRef
from shared.schemas.result.catalog import SFTResult
from shared.tasks.artifact_binding import ArtifactKind
from worker.content.artifacts import (
    ArtifactChanged,
    ArtifactPublisher,
    ReservedArtifactName,
    declared_archives,
)


def _store(tmp_path: Path) -> SharedFilesystemObjectStore:
    return SharedFilesystemObjectStore(tmp_path / "cas")


def _artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    (root / "final_model" / "nested").mkdir(parents=True)
    (root / "final_model" / "config.json").write_bytes(b"{}")
    (root / "final_model" / "nested" / "weights.bin").write_bytes(os.urandom(70_000))
    (root / "metrics.json").write_bytes(b'{"loss": 1}')
    (root / "checkpoints" / "checkpoint-10").mkdir(parents=True)
    (root / "checkpoints" / "checkpoint-10" / "optimizer.pt").write_bytes(b"opt")
    (root / "checkpoints" / "final_model").mkdir()
    (root / "checkpoints" / "final_model" / "model.bin").write_bytes(b"dpo")
    (root / ".fm-tmp-x").write_bytes(b"in flight")
    (root / "link").symlink_to(tmp_path)
    return root


def _read(store: SharedFilesystemObjectStore, scope: str) -> MemberReader:
    def read(member: TreeMember, start: int, length: int) -> Iterator[bytes]:
        return store.open_range(member.reference(scope), start, length)

    return read


def test_the_final_set_publishes_each_top_level_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _artifacts(tmp_path)

    bindings = {
        b.name: b
        for b in ArtifactPublisher(store, "org").publish_final(
            root, {"final_model.tar.gz": "final_model"}, step_revisions=True
        )
    }

    assert set(bindings) == {
        "final_model",
        "final_model.tar.gz",
        "metrics.json",
        "checkpoints/final_model",
    }
    metrics = bindings["metrics.json"]
    assert metrics.kind is ArtifactKind.FILE
    assert store.hydrate(metrics.reference) == b'{"loss": 1}'
    tree = bindings["final_model"]
    manifest = TreeManifest.decode(store.hydrate(tree.reference))
    assert [m.path for m in manifest.members] == ["config.json", "nested/weights.bin"]
    assert tree.files == 2 and tree.size_bytes == 70_002
    assert tree.largest is not None and tree.largest.size_bytes == 70_000
    for member in manifest.members:
        assert (
            store.hydrate(member.reference("org"))
            == (root / "final_model" / member.path).read_bytes()
        )


def test_a_task_without_step_revisions_publishes_its_checkpoints_whole(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    bindings = {
        b.name: b
        for b in ArtifactPublisher(store, "org").publish_final(_artifacts(tmp_path), {})
    }

    assert set(bindings) == {"final_model", "metrics.json", "checkpoints"}
    manifest = TreeManifest.decode(store.hydrate(bindings["checkpoints"].reference))
    assert [m.path for m in manifest.members] == [
        "checkpoint-10/optimizer.pt",
        "final_model/model.bin",
    ]


def test_an_artifact_named_as_the_result_is_refused(tmp_path: Path) -> None:
    root = _artifacts(tmp_path)
    (root / "results.json").write_bytes(b"{}")

    with pytest.raises(ReservedArtifactName):
        ArtifactPublisher(_store(tmp_path), "org").publish_final(root, {})


def test_an_archive_view_hashes_the_bytes_a_reader_rebuilds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _artifacts(tmp_path)
    bindings = {
        b.name: b
        for b in ArtifactPublisher(store, "org").publish_final(
            root, {"final_model.tar.gz": "final_model"}
        )
    }
    archive = bindings["final_model.tar.gz"]
    assert archive.kind is ArtifactKind.ARCHIVE and archive.of == "final_model"
    assert archive.archive is not None
    manifest = TreeManifest.decode(store.hydrate(archive.reference))

    rebuilt = b"".join(
        TreeArchive(manifest, archive.archive.top).chunks(
            _read(store, "org"), crc32=archive.archive.crc32
        )
    )

    assert content_digest(rebuilt) == archive.archive.sha256
    assert len(rebuilt) == archive.archive.size_bytes == archive.size_bytes
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(rebuilt))) as tar:
        assert sorted(tar.getnames()) == [
            "final_model/config.json",
            "final_model/nested/weights.bin",
        ]


def test_an_archive_of_no_published_tree_is_not_named(tmp_path: Path) -> None:
    bindings = ArtifactPublisher(_store(tmp_path), "org").publish_final(
        _artifacts(tmp_path), {"absent.tar.gz": "absent"}
    )

    assert "absent.tar.gz" not in {b.name for b in bindings}


def test_a_file_changed_after_it_published_fails_the_archive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _artifacts(tmp_path)
    publisher = ArtifactPublisher(store, "org")
    tree, manifest = publisher.publish_tree(
        "final_model", root / "final_model", "final"
    )
    (root / "final_model" / "config.json").write_bytes(b"[]")

    with pytest.raises(ArtifactChanged):
        publisher.publish_archive(
            "final_model.tar.gz", tree, manifest, root / "final_model"
        )


def test_a_result_names_its_archive_views() -> None:
    result = SFTResult(
        final_model=ArtifactRef(path="final_model"),
        final_model_archive=ArtifactRef(path="final_model.tar.gz"),
    )

    assert declared_archives(result) == {"final_model.tar.gz": "final_model"}
