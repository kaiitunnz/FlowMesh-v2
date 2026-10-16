"""Directory and file access below a directory that never follows a link."""

import contextlib
import errno
import os
import stat
import tracemalloc
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

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
from tests.support.filesystems import other_filesystem


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
        seen = [(rel, dirs, others) for rel, dirs, others, _ in walk(top_fd)]

    assert seen == [("", ["sub"], ["linked"]), ("sub", [], [])]
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


def _chain(top: Path, depth: int, files: dict[int, bytes]) -> None:
    """A chain of ``depth`` directories below ``top``, with ``files[level]`` in the
    directory ``level`` deep."""
    fd = os.open(top.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.mkdir(top.name, dir_fd=fd)
    for level in range(depth + 1):
        child = os.open(
            top.name if level == 0 else "d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd
        )
        os.close(fd)
        fd = child
        if level in files:
            leaf = os.open("leaf.bin", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd)
            os.write(leaf, files[level])
            os.close(leaf)
        if level < depth:
            os.mkdir("d", dir_fd=fd)
    os.close(fd)


def test_a_walk_stops_below_its_depth_bound_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    depth = nofollow._MAX_WALK_DEPTH
    _chain(tmp_path / "top", depth + 5, {})

    with open_dir(tmp_path / "top") as top_fd:
        levels = sum(1 for _ in walk(top_fd))

    assert levels == depth + 1
    assert caplog.text.count("deeper than") == 1


def test_the_files_of_a_tree_too_deep_end_refused(tmp_path: Path) -> None:
    depth = nofollow._MAX_WALK_DEPTH
    _chain(tmp_path / "top", depth + 5, {depth: b"at", depth + 1: b"past"})

    found: list[tuple[int, bytes]] = []
    errors: list[OSError] = []
    for name, opened in regular_files(tmp_path / "top"):
        if isinstance(opened, OSError):
            errors.append(opened)
            continue
        with opened:
            found.append((name.count("/"), opened.read()))

    assert found == [(depth, b"at")]
    assert len(errors) == 1 and isinstance(errors[0], nofollow.TreeTooDeep)


def test_a_walk_of_a_very_deep_tree_holds_bounded_memory(tmp_path: Path) -> None:
    _chain(tmp_path / "top", 5000, {})
    tracemalloc.start()
    try:
        with open_dir(tmp_path / "top") as top_fd:
            levels = sum(1 for _ in walk(top_fd))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert levels == nofollow._MAX_WALK_DEPTH + 1
    assert peak < 4 << 20


def test_linking_an_opened_file_names_its_inode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    kept = tmp_path / "kept"
    os.link(source, kept)
    with open_regular_path(source) as opened, open_dir(tmp_path) as dir_fd:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"replacement")
        os.replace(replacement, source)

        assert nofollow.link_opened(opened.fileno(), dir_fd, "linked")

    linked = tmp_path / "linked"
    assert os.path.samestat(linked.stat(), kept.stat())
    assert linked.read_bytes() == b"bytes"


def test_an_opened_file_on_another_filesystem_is_not_linked(tmp_path: Path) -> None:
    if (other := other_filesystem(tmp_path)) is None:
        pytest.skip("no second writable filesystem")
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    with open_regular_path(source) as opened, open_dir(other) as dir_fd:
        name = f"flowmesh-link-test-{os.getpid()}"
        try:
            assert not nofollow.link_opened(opened.fileno(), dir_fd, name)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=dir_fd)


@pytest.mark.parametrize(
    "code", [errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK]
)
def test_a_link_the_filesystem_cannot_make_reports_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    monkeypatch.setattr(nofollow.os, "link", _raising(code))
    with open_dir(tmp_path) as dir_fd:
        assert not nofollow.link_opened(0, dir_fd, "linked")


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO, errno.EEXIST])
def test_any_other_link_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    monkeypatch.setattr(nofollow.os, "link", _raising(code))
    with open_dir(tmp_path) as dir_fd, pytest.raises(OSError) as raised:
        nofollow.link_opened(0, dir_fd, "linked")
    assert raised.value.errno == code


def _raising(code: int) -> Any:
    def link(*args: Any, **kwargs: Any) -> None:
        raise OSError(code, os.strerror(code))

    return link


@contextlib.contextmanager
def open_regular_path(path: Path) -> Iterator[BinaryIO]:
    with open_dir(path.parent) as dir_fd:
        opened = open_regular(dir_fd, path.name)
    assert opened is not None
    with opened:
        yield opened
