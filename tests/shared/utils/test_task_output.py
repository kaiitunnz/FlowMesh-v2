"""Tests for the shared manifest helpers."""

import errno
import os
import stat
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.utils.nofollow import PathRefused
from shared.utils.task_output import (
    ARTIFACTS_DIR,
    LOGS_DIR,
    SCRATCH_DIR,
    detach_shared_files,
    prepare_output_dir,
    scratch_dir,
)


def _race(fn, *, threads: int = 8) -> list[BaseException]:
    """Run ``fn`` on ``threads`` threads released simultaneously."""
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            fn()
        except BaseException as exc:
            with lock:
                errors.append(exc)

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return errors


class TestPrepareOutputDir:
    def test_directories_are_world_writable(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_is_idempotent(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        prepare_output_dir(out)
        prepare_output_dir(out)
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_existing_directory_is_remoded(self, tmp_path: Path) -> None:
        """A directory another writer created at its own mode ends world-writable,
        so peer worker UIDs can still write into it."""
        out = tmp_path / "task-out"
        out.mkdir(mode=0o0755)
        prepare_output_dir(out)
        assert stat.S_IMODE(out.stat().st_mode) == 0o0777

    def test_concurrent_calls_do_not_raise(self, tmp_path: Path) -> None:
        """Concurrent writers materializing one task directory all succeed."""
        out = tmp_path / "tsk-concurrent"
        assert _race(lambda: prepare_output_dir(out)) == []
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_a_directory_another_writer_just_created_is_kept(
        self, tmp_path: Path
    ) -> None:
        """A directory that appears after a writer checked for it is reused."""
        out = tmp_path / "task-out"
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            d.mkdir(parents=True)
        with patch.object(Path, "exists", return_value=False):
            prepare_output_dir(out)
        for d in (out, out / LOGS_DIR, out / ARTIFACTS_DIR):
            assert stat.S_IMODE(d.stat().st_mode) == 0o0777

    def test_rejects_non_directory_at_path(self, tmp_path: Path) -> None:
        out = tmp_path / "task-out"
        out.write_text("not a directory")
        with pytest.raises(PathRefused):
            prepare_output_dir(out)


class TestScratchDir:
    def test_creates_world_writable_dir(self, tmp_path: Path) -> None:
        path = scratch_dir(tmp_path)
        assert path == tmp_path / SCRATCH_DIR
        assert stat.S_IMODE(path.stat().st_mode) == 0o0777

    def test_concurrent_calls_do_not_raise(self, tmp_path: Path) -> None:
        assert _race(lambda: scratch_dir(tmp_path)) == []

    def test_a_directory_another_writer_just_created_is_kept(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / SCRATCH_DIR).mkdir()
        with patch.object(Path, "exists", return_value=False):
            path = scratch_dir(tmp_path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o0777
        assert stat.S_IMODE((tmp_path / SCRATCH_DIR).stat().st_mode) == 0o0777


def _tree(tmp_path: Path) -> Path:
    top = tmp_path / ARTIFACTS_DIR
    (top / "checkpoints" / "checkpoint-5").mkdir(parents=True)
    return top


def test_a_file_shared_outside_the_tree_gets_a_private_copy(tmp_path: Path) -> None:
    top = _tree(tmp_path)
    weights = top / "checkpoints" / "checkpoint-5" / "weights"
    weights.write_bytes(b"accepted")
    held = tmp_path / "custody"
    os.link(weights, held)

    assert detach_shared_files(top) == 1
    weights.write_bytes(b"rewritten by the next attempt")

    assert held.read_bytes() == b"accepted"
    assert weights.stat().st_nlink == 1


def test_a_detached_copy_keeps_the_mode_and_times(tmp_path: Path) -> None:
    top = _tree(tmp_path)
    model = top / "model"
    model.write_bytes(b"accepted")
    model.chmod(0o640)
    os.utime(model, ns=(1_000_000_000, 2_000_000_000))
    os.link(model, tmp_path / "cache")

    detach_shared_files(top)

    detached = model.stat()
    assert stat.S_IMODE(detached.st_mode) == 0o640
    assert detached.st_mtime_ns == 2_000_000_000
    assert model.read_bytes() == b"accepted"


def test_names_in_the_tree_linking_one_shared_file_keep_linking_one_copy(
    tmp_path: Path,
) -> None:
    top = _tree(tmp_path)
    (top / "a").write_bytes(b"shared")
    os.link(top / "a", top / "checkpoints" / "b")
    os.link(top / "a", tmp_path / "custody")

    assert detach_shared_files(top) == 1

    assert os.path.samestat((top / "a").stat(), (top / "checkpoints" / "b").stat())
    assert not os.path.samestat((top / "a").stat(), (tmp_path / "custody").stat())


def test_files_linked_only_inside_the_tree_are_left_as_they_are(
    tmp_path: Path,
) -> None:
    top = _tree(tmp_path)
    (top / "a").write_bytes(b"own")
    (top / "lone").write_bytes(b"lone")
    os.link(top / "a", top / "b")
    before = {name: (top / name).stat().st_ino for name in ("a", "b", "lone")}

    assert detach_shared_files(top) == 0
    assert {name: (top / name).stat().st_ino for name in before} == before


def test_a_link_in_the_tree_is_not_followed(tmp_path: Path) -> None:
    top = _tree(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    os.link(outside, tmp_path / "another")
    (top / "pointer").symlink_to(outside)

    assert detach_shared_files(top) == 0
    assert outside.stat().st_nlink == 2


def test_a_missing_tree_has_nothing_to_detach(tmp_path: Path) -> None:
    assert detach_shared_files(tmp_path / ARTIFACTS_DIR) == 0


def test_a_copy_that_cannot_be_written_leaves_the_shared_file(
    tmp_path: Path,
) -> None:
    top = _tree(tmp_path)
    model = top / "model"
    model.write_bytes(b"accepted")
    os.link(model, tmp_path / "custody")

    with (
        patch(
            "shared.utils.task_output.os.fchmod",
            side_effect=OSError(errno.ENOSPC, "No space"),
        ),
        pytest.raises(OSError),
    ):
        detach_shared_files(top)

    assert os.path.samestat(model.stat(), (tmp_path / "custody").stat())
    assert sorted(p.name for p in top.iterdir()) == ["checkpoints", "model"]


class _Cancelled(Exception):
    pass


def test_a_cancel_between_chunks_ends_the_pass_leaving_every_name_whole(
    tmp_path: Path,
) -> None:
    top = _tree(tmp_path)
    model = top / "model"
    model.write_bytes(b"a" * (3 << 20))
    os.link(model, tmp_path / "custody")
    chunks = 0

    def cancel_on_the_second_chunk() -> None:
        nonlocal chunks
        chunks += 1
        if chunks == 3:
            raise _Cancelled

    with pytest.raises(_Cancelled):
        detach_shared_files(top, cancel_on_the_second_chunk)

    assert os.path.samestat(model.stat(), (tmp_path / "custody").stat())
    assert sorted(p.name for p in top.iterdir()) == ["checkpoints", "model"]


def test_a_cancel_before_a_file_copies_nothing(tmp_path: Path) -> None:
    top = _tree(tmp_path)
    (top / "model").write_bytes(b"accepted")
    os.link(top / "model", tmp_path / "custody")

    def cancelled() -> None:
        raise _Cancelled

    with pytest.raises(_Cancelled):
        detach_shared_files(top, cancelled)

    assert (top / "model").stat().st_nlink == 2
