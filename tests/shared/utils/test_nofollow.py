"""Directory and file access below a directory that never follows a link."""

import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from shared.utils import nofollow
from shared.utils.atomic import atomic_write_text
from shared.utils.nofollow import (
    PathRefused,
    open_append,
    open_below,
    open_dir,
    open_dir_at,
    open_regular,
    regular_files,
    walk,
)


@pytest.fixture
def target(tmp_path: Path) -> Path:
    path = tmp_path / "target"
    path.mkdir(mode=0o700)
    (path / "file").write_bytes(b"secret")
    return path


def test_a_link_at_or_below_the_base_is_refused(tmp_path: Path, target: Path) -> None:
    (tmp_path / "linked").symlink_to(target)
    base = tmp_path / "base"
    base.mkdir()
    (base / "logs").symlink_to(target)

    for path, parts in [(tmp_path / "linked", ()), (base, ("logs",))]:
        with (
            pytest.raises(PathRefused),
            open_dir(path, *parts, create=True, mode=0o777),
        ):
            pass
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_the_base_parent_is_trusted(tmp_path: Path, target: Path) -> None:
    (tmp_path / "root").symlink_to(target)
    with open_dir(tmp_path / "root" / "made", "sub", create=True, mode=0o777) as fd:
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o777
    assert (target / "made" / "sub").is_dir()


def test_regular_reads_skip_links_and_special_files(
    tmp_path: Path, target: Path
) -> None:
    (tmp_path / "link").symlink_to(target / "file")
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "plain").write_bytes(b"ok")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert open_regular(fd, "link") is None
        assert open_regular(fd, "pipe") is None
        assert open_regular(fd, "gone") is None
        opened = open_regular(fd, "plain")
        assert opened is not None
        with opened:
            assert opened.read() == b"ok"
        with pytest.raises(PathRefused), open_dir_at(fd, "plain"):
            pass
        with pytest.raises(PathRefused):
            open_append(fd, "link")
    finally:
        os.close(fd)
    assert open_below(tmp_path, Path("link")) is None
    assert (target / "file").read_bytes() == b"secret"


def test_a_walk_yields_regular_files_only(tmp_path: Path, target: Path) -> None:
    (tmp_path / "top" / "sub").mkdir(parents=True)
    (tmp_path / "top" / "sub" / "a.bin").write_bytes(b"a")
    (tmp_path / "top" / "linked").symlink_to(target)
    (tmp_path / "top" / "file").symlink_to(target / "file")

    names = []
    for name, opened in regular_files(tmp_path / "top"):
        assert not isinstance(opened, OSError)
        opened.close()
        names.append(name)

    assert names == ["sub/a.bin"]


def test_an_atomic_write_replaces_a_link_rather_than_writing_through_it(
    tmp_path: Path, target: Path
) -> None:
    (tmp_path / "out.json").symlink_to(target / "file")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        atomic_write_text(Path("out.json"), "{}", dir_fd=fd)
    finally:
        os.close(fd)

    assert not (tmp_path / "out.json").is_symlink()
    assert (tmp_path / "out.json").read_text() == "{}"
    assert (target / "file").read_bytes() == b"secret"
    assert stat.S_IMODE((tmp_path / "out.json").stat().st_mode) == 0o666


def test_a_walk_never_opens_a_linked_directory(
    tmp_path: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "top" / "sub").mkdir(parents=True)
    (tmp_path / "top" / "linked").symlink_to(target)
    opened: list[str] = []
    open_child = nofollow._open_child

    def _recording(fd: int, name: str, *args: Any, **kwargs: Any) -> int:
        opened.append(name)
        return open_child(fd, name, *args, **kwargs)

    monkeypatch.setattr(nofollow, "_open_child", _recording)

    with open_dir(tmp_path / "top") as top_fd:
        seen = [(rel.as_posix(), dirs, others) for rel, dirs, others, _ in walk(top_fd)]

    assert seen == [(".", ["sub"], ["linked"]), ("sub", [], [])]
    assert opened == ["top", "sub"]


def test_a_special_file_is_never_opened_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.mkfifo(tmp_path / "pipe")
    opened: list[str] = []
    real_open = os.open

    def _recording(path: Any, *args: Any, **kwargs: Any) -> int:
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.setattr(nofollow.os, "open", _recording)
        assert open_regular(fd, "pipe") is None
    finally:
        monkeypatch.undo()
        os.close(fd)
    assert opened == []


@pytest.mark.parametrize("segment", ["", ".", "..", "a/b", "a\0b"])
def test_a_segment_that_leaves_its_directory_is_refused(
    tmp_path: Path, segment: str
) -> None:
    with pytest.raises(PathRefused), open_dir(tmp_path, segment):
        pass
    assert open_below(tmp_path, PurePosixPath("x", segment or "y", "z")) is None


def test_a_walk_reaches_a_tree_deeper_than_the_recursion_limit(tmp_path: Path) -> None:
    depth = sys.getrecursionlimit() + 500
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    os.mkdir("top", dir_fd=fd)
    for level in range(depth + 1):
        child = os.open(
            "top" if level == 0 else "d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd
        )
        os.close(fd)
        fd = child
        if level < depth:
            os.mkdir("d", dir_fd=fd)
    leaf = os.open("leaf.bin", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd)
    os.write(leaf, b"deep")
    os.close(leaf)
    os.close(fd)

    found = []
    for name, opened in regular_files(tmp_path / "top"):
        assert not isinstance(opened, OSError)
        with opened:
            found.append((name.count("/"), opened.read()))

    assert found == [(depth, b"deep")]
