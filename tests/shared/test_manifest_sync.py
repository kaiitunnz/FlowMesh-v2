"""Manifest syncs and the atomic writes they scan."""

import hashlib
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from shared.utils import atomic, manifest
from shared.utils.manifest import MANIFEST_NAME, sync_manifest


def test_a_sync_never_overwrites_a_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = manifest.atomic_write_text
    first_scanned = threading.Event()
    later_wrote = threading.Event()

    def _held_first_write(target: Path, content: str, **kwargs: Any) -> None:
        if not first_scanned.is_set():
            # The first sync has scanned; hold its write until the later sync writes,
            # or briefly when the later sync cannot run alongside it.
            first_scanned.set()
            later_wrote.wait(0.5)
        else:
            later_wrote.set()
        write(target, content, **kwargs)

    monkeypatch.setattr(manifest, "atomic_write_text", _held_first_write)
    (tmp_path / "a.txt").write_text("a")
    first = threading.Thread(target=sync_manifest, args=(tmp_path, "t", []))
    first.start()
    assert first_scanned.wait(5)
    (tmp_path / "b.txt").write_text("b")
    later = threading.Thread(target=sync_manifest, args=(tmp_path, "t", []))
    later.start()
    first.join(5)
    later.join(5)

    paths = {
        e["path"] for e in json.loads((tmp_path / MANIFEST_NAME).read_text())["entries"]
    }
    assert {"a.txt", "b.txt"} <= paths


def test_a_directory_lock_lives_only_while_a_sync_holds_it(tmp_path: Path) -> None:
    for index in range(50):
        sync_manifest(tmp_path / f"task-{index}", f"t{index}", [])

    assert len(manifest._MANIFEST_LOCKS) == 0


def test_a_scan_skips_in_flight_writes_and_files_removed_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "kept.bin").write_bytes(b"x" * 3)
    in_flight = Path(tempfile.mkstemp(prefix=".fm-tmp-", dir=artifacts)[1])
    in_flight.write_bytes(b"partial")
    assert atomic.is_atomic_temp(in_flight.name)
    (tmp_path / in_flight.name).write_bytes(b"partial")
    walk = manifest.walk

    # A file the scan lists, then finds gone: renamed or removed between the two.
    def _with_removed(top_fd: int) -> Any:
        for rel_dir, dirs, others, dirfd in walk(top_fd):
            yield rel_dir, dirs, [*others, "gone.bin"], dirfd

    monkeypatch.setattr(manifest, "walk", _with_removed)

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    assert in_flight.name not in entries
    assert entries["artifacts"]["file_count"] == 1
    assert entries["artifacts"]["size"] == 3


def test_a_long_filename_writes_atomically(tmp_path: Path) -> None:
    target = tmp_path / ("a" * 250)

    atomic.atomic_write_stream(target, io.BytesIO(b"data"))

    assert target.read_bytes() == b"data"


def test_an_existing_entry_that_is_not_a_file_is_present(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    entries = {
        e["path"]: e for e in sync_manifest(tmp_path, "t", ["gone.bin"])["entries"]
    }

    pipe = entries["pipe"]
    assert (pipe["status"], pipe["size"], pipe["file_count"]) == ("present", 0, 0)
    assert entries["gone.bin"]["status"] == "missing"


def test_a_declared_name_holding_a_nul_reads_as_missing(tmp_path: Path) -> None:
    entries = {
        e["path"]: e for e in sync_manifest(tmp_path, "t", ["bad\0name"])["entries"]
    }

    assert entries["bad\0name"]["status"] == "missing"


_BUDGET = 256 << 20


class _CountingHashlib:
    """``hashlib`` for the manifest module, counting the bytes it digests."""

    def __init__(self) -> None:
        self.digested = 0

    def sha256(self) -> Any:
        counter = self
        real = hashlib.sha256()

        class _Counting:
            def update(self, data: bytes) -> None:
                counter.digested += len(data)
                real.update(data)

            def hexdigest(self) -> str:
                return real.hexdigest()

        return _Counting()


def _sparse(path: Path, size: int) -> None:
    with path.open("wb") as fh:
        fh.truncate(size)


def test_a_file_over_the_hash_budget_is_listed_by_size_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = _CountingHashlib()
    monkeypatch.setattr(manifest, "hashlib", counting)
    _sparse(tmp_path / "huge.bin", 8 << 30)

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    assert entries["huge.bin"]["size"] == 8 << 30
    assert "sha256" not in entries["huge.bin"]
    assert counting.digested == 0


def test_a_sync_digests_at_most_its_budget_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counting = _CountingHashlib()
    monkeypatch.setattr(manifest, "hashlib", counting)
    for index in range(10):
        _sparse(tmp_path / f"f{index}.bin", 60 << 20)
    _sparse(tmp_path / "f9.bin", 1 << 20)

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    digested = [f"f{i}.bin" for i in range(10) if "sha256" in entries[f"f{i}.bin"]]
    assert digested == ["f0.bin", "f1.bin", "f2.bin", "f3.bin", "f9.bin"]
    assert counting.digested <= _BUDGET


def test_a_small_manifest_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest, "now_iso", lambda: "2026-01-01T00:00:00+00:00")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "a.txt").write_bytes(b"alpha")
    (tmp_path / "results.json").write_bytes(b"{}")
    (tmp_path / "extra.bin").write_bytes(b"x" * 9000)

    sync_manifest(tmp_path, "t", ["artifacts/a.txt"])

    def entry(path: str, kind: str, required: bool, data: bytes) -> dict[str, Any]:
        return {
            "name": path,
            "path": path,
            "type": kind,
            "required": required,
            "status": "present",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def directory(path: str, kind: str, size: int, count: int) -> dict[str, Any]:
        return {
            "name": path,
            "path": path,
            "type": kind,
            "required": True,
            "status": "present",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "size": size,
            "file_count": count,
        }

    expected = {
        "task_id": "t",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "entries": [
            directory("artifacts", "artifact", 5, 1),
            entry("artifacts/a.txt", "artifact", True, b"alpha"),
            directory("logs", "logs", 0, 0),
            entry("results.json", "result", True, b"{}"),
            entry("extra.bin", "artifact", False, b"x" * 9000),
        ],
    }
    assert (tmp_path / MANIFEST_NAME).read_text() == json.dumps(
        expected, ensure_ascii=False, indent=2
    )
