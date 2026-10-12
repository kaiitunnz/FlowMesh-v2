"""Committing a finished file to its name only where nothing is there yet."""

import os
from pathlib import Path

import pytest

from shared.utils import atomic
from shared.utils.atomic import link_no_replace, temp_path


def _finished(directory: Path, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(directory)
    tmp.write_bytes(data)
    return tmp


def test_a_link_lands_where_nothing_is(tmp_path) -> None:
    target = tmp_path / "objects" / "a" / "obj"

    assert link_no_replace(_finished(tmp_path, b"first"), target)
    assert target.read_bytes() == b"first"
    assert [p.name for p in tmp_path.iterdir()] == ["objects"]


def test_a_link_never_replaces_what_is_there(tmp_path) -> None:
    target = tmp_path / "obj"
    target.write_bytes(b"first")

    assert not link_no_replace(_finished(tmp_path, b"second"), target)
    assert target.read_bytes() == b"first"
    assert [p.name for p in tmp_path.iterdir()] == ["obj"]


def test_a_durable_writer_that_finds_another_s_link_waits_for_it_to_reach_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "obj"
    target.write_bytes(b"first")
    synced: list[str] = []
    fsync = os.fsync

    def recording(fd: int) -> None:
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        fsync(fd)

    monkeypatch.setattr(atomic.os, "fsync", recording)

    assert not link_no_replace(_finished(tmp_path, b"first"), target, durable=True)
    assert target.as_posix() in synced
    assert tmp_path.as_posix() in synced
