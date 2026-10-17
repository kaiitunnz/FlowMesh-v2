"""Upload custody: staged artifact bytes outlive every remover of the cache's copy until
their upload settles, across every process sharing the content directory."""

import errno
import os
import subprocess  # nosec B404 - runs this interpreter on a fixed script
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from shared.content import ContentReference, WriteCancelled, reference_for
from tests.support.custody import stage
from tests.support.filesystems import other_filesystem
from worker.content import WorkerContentCache
from worker.content import custody as custody_module
from worker.content.custody import ArtifactCustody, StagedTask, UploadJournal


def _source(tmp_path: Path, data: bytes, name: str = "weights.bin") -> Path:
    path = tmp_path / "out" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _age(cache: WorkerContentCache, reference: ContentReference, ago: float) -> None:
    path = cache._objects.object_path(
        reference.authorization_scope, reference.content_digest
    )
    at = time.time() - ago
    os.utime(path, (at, at))


def _journal(
    dispatch_id: str, tasks: dict[str, StagedTask] | None = None
) -> UploadJournal:
    return UploadJournal(
        dispatch_id=dispatch_id,
        scope="org",
        alias="w1",
        worker_id="wkr-1",
        incarnation=1,
        task_id="tsk-1",
        tasks=tasks or {},
    )


def _cached(
    root: Path, retain_sec: float = 0.0
) -> tuple[WorkerContentCache, ArtifactCustody]:
    cache = WorkerContentCache(root, retain_sec=retain_sec)
    return cache, ArtifactCustody(root, keep=cache.keep_staged)


def test_staging_links_the_file_it_hashes(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    source = _source(tmp_path, b"weights")

    reference = stage(custody, "org", "dsp-1", source)

    held = custody.source(_journal("dsp-1"), reference)
    assert reference == reference_for("org", b"weights")
    assert os.path.samestat(held.stat(), source.stat())


def test_a_file_at_its_link_limit_is_held_by_a_copy(tmp_path, monkeypatch) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    source = _source(tmp_path, b"weights")
    link = os.link

    def limited(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if Path(dst).name == reference_for("org", b"weights").content_digest and (
            os.stat(src).st_ino == source.stat().st_ino
        ):
            raise OSError(errno.EMLINK, "Too many links")
        link(src, dst, *args, **kwargs)

    monkeypatch.setattr(custody_module.os, "link", limited)

    reference = stage(custody, "org", "dsp-1", source)

    held = custody.source(_journal("dsp-1"), reference)
    assert reference == reference_for("org", b"weights")
    assert held.read_bytes() == b"weights"
    assert not os.path.samestat(held.stat(), source.stat())
    assert [p.name for p in held.parent.iterdir()] == [held.name]


def test_staging_links_the_opened_file_whatever_its_path_names_by_then(
    tmp_path,
) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    source = _source(tmp_path, b"weights")
    kept = tmp_path / "kept"
    os.link(source, kept)
    with source.open("rb") as opened:
        _source(tmp_path, b"impostor", "other.bin").replace(source)
        reference = custody.stage("org", "dsp-1", opened)

    held = custody.source(_journal("dsp-1"), reference)
    assert reference == reference_for("org", b"weights")
    assert os.path.samestat(held.stat(), kept.stat())


def test_a_file_that_cannot_be_linked_is_copied_from_its_descriptor(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(custody_module, "link_opened", lambda *args: False)
    custody = ArtifactCustody(tmp_path / "content")
    source = _source(tmp_path, b"weights")
    with source.open("rb") as opened:
        opened.read(3)
        _source(tmp_path, b"impostor", "other.bin").replace(source)
        reference = custody.stage("org", "dsp-1", opened)

    held = custody.source(_journal("dsp-1"), reference)
    assert reference == reference_for("org", b"weights")
    assert held.read_bytes() == b"weights"
    assert held.stat().st_nlink == 1


def test_a_file_on_another_filesystem_is_copied_into_custody(tmp_path) -> None:
    if (other := other_filesystem(tmp_path)) is None:
        pytest.skip("no second writable filesystem")
    custody = ArtifactCustody(tmp_path / "content")
    with tempfile.NamedTemporaryFile(dir=other) as source:
        source.write(b"weights")
        source.flush()
        source.seek(0)
        reference = custody.stage("org", "dsp-1", cast(BinaryIO, source))

        held = custody.source(_journal("dsp-1"), reference)
        assert held.read_bytes() == b"weights"
        assert not os.path.samestat(held.stat(), os.stat(source.name))


def test_a_link_failure_the_filesystem_does_not_excuse_fails_staging(
    tmp_path, monkeypatch
) -> None:
    def full(*args: Any) -> bool:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(custody_module, "link_opened", full)
    custody = ArtifactCustody(tmp_path / "content")

    with pytest.raises(OSError) as raised:
        stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))

    assert raised.value.errno == errno.ENOSPC
    assert list(custody.publication_dir("org", "dsp-1").iterdir()) == []


def test_a_cancelled_staging_holds_nothing(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    with (
        _source(tmp_path, b"weights").open("rb") as opened,
        pytest.raises(WriteCancelled),
    ):
        custody.stage("org", "dsp-1", opened, cancelled=lambda: True)

    assert list(custody.publication_dir("org", "dsp-1").iterdir()) == []


def test_custody_holds_the_bytes_it_hashed_not_a_copy_already_cached(
    tmp_path,
) -> None:
    cache, custody = _cached(tmp_path / "content")
    wanted = reference_for("org", b"weights")
    impostor = cache._objects.object_path("org", wanted.content_digest)
    impostor.parent.mkdir(parents=True)
    impostor.write_bytes(b"XXXXXXX")

    reference = stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))

    assert custody.source(_journal("dsp-1"), reference).read_bytes() == b"weights"


def test_a_staged_copy_survives_eviction_of_the_cache_name(tmp_path) -> None:
    cache, custody = _cached(tmp_path / "content", retain_sec=1.0)
    reference = stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))
    assert cache.holds(reference)
    _age(cache, reference, ago=3600)

    assert cache.evict() == 0
    assert cache.holds(reference)

    custody.release("org", "dsp-1")
    assert cache.evict() == 1
    assert not cache.holds(reference)


def test_another_process_on_the_node_does_not_evict_a_staged_copy(tmp_path) -> None:
    root = tmp_path / "content"
    _, custody = _cached(root)
    reference = stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))
    _age(WorkerContentCache(root), reference, ago=3600)
    script = (
        "import sys; from pathlib import Path; "
        "from worker.content import WorkerContentCache; "
        "print(WorkerContentCache(Path(sys.argv[1]), retain_sec=1.0).evict())"
    )

    evicted = subprocess.run(  # nosec B603 - this interpreter, a fixed script
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert evicted.stdout.strip() == "0"
    assert custody.source(_journal("dsp-1"), reference).read_bytes() == b"weights"


def test_custody_keeps_the_bytes_when_the_cache_name_is_removed(tmp_path) -> None:
    cache, custody = _cached(tmp_path / "content")
    reference = stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))

    cache.drop(reference)

    assert custody.source(_journal("dsp-1"), reference).read_bytes() == b"weights"


def test_without_a_cache_nothing_is_left_once_custody_ends(tmp_path) -> None:
    root = tmp_path / "content"
    custody = ArtifactCustody(root)
    stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))

    custody.release("org", "dsp-1")

    assert [p for p in root.rglob("*") if p.is_file()] == []


def test_overlapping_publications_hold_one_object_independently(tmp_path) -> None:
    cache, custody = _cached(tmp_path / "content", retain_sec=1.0)
    first = stage(custody, "org", "dsp-1", _source(tmp_path, b"same"))
    second = stage(custody, "org", "dsp-2", _source(tmp_path, b"same", "copy.bin"))
    assert first == second

    custody.release("org", "dsp-1")
    _age(cache, first, ago=3600)
    cache.evict()
    assert custody.source(_journal("dsp-2"), second).read_bytes() == b"same"


def test_journals_round_trip_and_release_removes_them(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    reference = stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))
    journal = _journal(
        "dsp-1", {"tsk-1": StagedTask(task_id="tsk-1", members=[reference])}
    )
    journal.task_id = "tsk-1"

    custody.record(journal)
    assert list(custody.journals()) == [journal]

    custody.release("org", "dsp-1")
    assert list(custody.journals()) == []


def test_custody_without_a_journal_is_not_reported(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    stage(custody, "org", "dsp-1", _source(tmp_path, b"weights"))

    assert list(custody.journals()) == []
    assert custody.publication_dir("org", "dsp-1").is_dir()
