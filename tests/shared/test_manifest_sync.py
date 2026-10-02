"""Manifest syncs and the atomic writes they scan."""

import io
import json
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
    is_file = Path.is_file

    # A file the scan sees, then finds gone: renamed or removed between the two.
    def _removed_after_listing(self: Path) -> bool:
        return self.name == "gone.bin" or is_file(self)

    monkeypatch.setattr(Path, "is_file", _removed_after_listing)
    monkeypatch.setattr(Path, "rglob", _with_extra(artifacts / "gone.bin"))

    entries = {e["path"]: e for e in sync_manifest(tmp_path, "t", [])["entries"]}

    assert in_flight.name not in entries
    assert entries["artifacts"]["file_count"] == 1
    assert entries["artifacts"]["size"] == 3


def _with_extra(extra: Path) -> Any:
    rglob = Path.rglob

    def _rglob(self: Path, pattern: str) -> Any:
        yield from rglob(self, pattern)
        if self == extra.parent:
            yield extra

    return _rglob


def test_a_long_filename_writes_atomically(tmp_path: Path) -> None:
    target = tmp_path / ("a" * 250)

    atomic.atomic_write_stream(target, io.BytesIO(b"data"))

    assert target.read_bytes() == b"data"
