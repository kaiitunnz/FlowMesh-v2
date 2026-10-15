"""Upload custody: staged artifact bytes outlive every remover of the cache's copy until
their upload settles, across every process sharing the content directory."""

import os
import subprocess  # nosec B404 - runs this interpreter on a fixed script
import sys
import time
from pathlib import Path

from shared.content import ContentReference
from worker.content import WorkerContentCache
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


def test_staging_copies_and_hashes_the_bytes_it_holds(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    source = _source(tmp_path, b"weights")

    reference = custody.stage("org", "dsp-1", source)
    source.write_bytes(b"changed after staging")

    journal = _journal("dsp-1")
    assert custody.source(journal, reference).read_bytes() == b"weights"


def test_a_staged_copy_survives_eviction_of_the_cache_name(tmp_path) -> None:
    cache = WorkerContentCache(tmp_path / "content", retain_sec=1.0)
    custody = ArtifactCustody(tmp_path / "content", write_file=cache.write_file)
    reference = custody.stage("org", "dsp-1", _source(tmp_path, b"weights"))
    _age(cache, reference, ago=3600)

    assert cache.evict() == 0
    assert cache.holds(reference)

    custody.release("org", "dsp-1")
    assert cache.evict() == 1
    assert not cache.holds(reference)


def test_another_process_on_the_node_does_not_evict_a_staged_copy(tmp_path) -> None:
    root = tmp_path / "content"
    custody = ArtifactCustody(root, write_file=WorkerContentCache(root).write_file)
    reference = custody.stage("org", "dsp-1", _source(tmp_path, b"weights"))
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
    cache = WorkerContentCache(tmp_path / "content")
    custody = ArtifactCustody(tmp_path / "content", write_file=cache.write_file)
    reference = custody.stage("org", "dsp-1", _source(tmp_path, b"weights"))
    journal = _journal("dsp-1")

    cache.drop(reference)
    assert custody.source(journal, reference).read_bytes() == b"weights"

    custody.restore_copy(journal, reference)
    assert cache.holds(reference)


def test_overlapping_publications_hold_one_object_independently(tmp_path) -> None:
    cache = WorkerContentCache(tmp_path / "content", retain_sec=1.0)
    custody = ArtifactCustody(tmp_path / "content", write_file=cache.write_file)
    first = custody.stage("org", "dsp-1", _source(tmp_path, b"same"))
    second = custody.stage("org", "dsp-2", _source(tmp_path, b"same", "copy.bin"))
    assert first == second

    custody.release("org", "dsp-1")
    _age(cache, first, ago=3600)
    assert cache.evict() == 0
    assert custody.source(_journal("dsp-2"), second).read_bytes() == b"same"


def test_journals_round_trip_and_release_removes_them(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    reference = custody.stage("org", "dsp-1", _source(tmp_path, b"weights"))
    journal = _journal(
        "dsp-1", {"tsk-1": StagedTask(task_id="tsk-1", members=[reference])}
    )
    journal.success = {"artifacts_pending": True}

    custody.record(journal)
    assert list(custody.journals()) == [journal]

    custody.release("org", "dsp-1")
    assert list(custody.journals()) == []


def test_custody_being_set_up_is_left_to_its_writer(tmp_path) -> None:
    custody = ArtifactCustody(tmp_path / "content")
    custody.stage("org", "dsp-1", _source(tmp_path, b"weights"))

    assert list(custody.journals()) == []
    assert custody.publication_dir("org", "dsp-1").is_dir()
