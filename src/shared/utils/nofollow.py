"""Filesystem access below a directory that never follows a link.

A task's results directory is shared with code the task runs, which can replace any
entry under it with a link, a FIFO, or another special file. These helpers hold each
directory open and reach every name relative to it without following a link, so a
link, or a directory swapped for one, is never chmodded, written through, or read
through, and a special file is never opened as a file.
"""

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
_APPEND_FLAGS = (
    os.O_WRONLY
    | os.O_APPEND
    | os.O_CREAT
    | os.O_NOFOLLOW
    | os.O_NONBLOCK
    | os.O_NOCTTY
    | os.O_CLOEXEC
)
_REFUSED_DIR_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})
_REFUSED_FILE_ERRNOS = frozenset({errno.ELOOP, errno.ENXIO, errno.EISDIR})
_SKIPPED_READ_ERRNOS = frozenset({errno.ELOOP, errno.ENXIO, errno.ENOENT})


class PathRefused(OSError):
    """A path names a link, a directory where a file belongs or a non-directory where
    a directory belongs, a special file, or a segment that leaves its directory."""


def is_plain_segment(name: str) -> bool:
    """Whether ``name`` names an entry of its directory: not empty, ``.`` or ``..``,
    and holding no separator or NUL."""
    return name not in {"", ".", ".."} and "/" not in name and "\0" not in name


@contextmanager
def open_dir(
    base_dir: Path, *parts: str, create: bool = False, mode: int | None = None
) -> Iterator[int]:
    """Open ``base_dir/parts...`` and yield its directory descriptor.

    ``base_dir``'s parent is trusted and opened as given, so a caller passes a path
    whose parent no untrusted party controls, such as ``<results>/<task>``.
    ``base_dir`` and every part below it are opened without following a link. With
    ``create`` each missing directory is made, and ``mode``, when given, is set on
    every directory opened from ``base_dir`` down, where the caller may.

    Raises ``PathRefused`` when any of them is a link, not a directory, or not a
    plain segment.
    """
    if create:
        base_dir.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(base_dir.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in (base_dir.name, *parts):
            child = _open_child(fd, name, create, mode)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_dir_at(dir_fd: int, *parts: str) -> Iterator[int]:
    """Open ``parts...`` below the directory ``dir_fd`` without following a link and
    yield its descriptor; raises ``PathRefused`` as ``open_dir`` does."""
    fd = os.dup(dir_fd)
    try:
        for name in parts:
            child = _open_child(fd, name, create=False, mode=None)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _open_child(fd: int, name: str, create: bool, mode: int | None) -> int:
    if not is_plain_segment(name):
        raise PathRefused(errno.EINVAL, "not a plain path segment", name)
    if create:
        try:
            os.mkdir(name, dir_fd=fd)
        except FileExistsError:
            pass
    try:
        child = os.open(name, _DIR_FLAGS, dir_fd=fd)
    except OSError as exc:
        if exc.errno in _REFUSED_DIR_ERRNOS:
            raise PathRefused(exc.errno, "not a directory", name) from exc
        raise
    if mode is not None:
        try:
            os.fchmod(child, mode)
        except PermissionError:
            pass
    return child


def walk(
    top_fd: int,
) -> Iterator[tuple[PurePosixPath, list[str], list[str], int]]:
    """Walk the tree under the directory ``top_fd`` top-down, never opening a link.

    Yields ``(relative directory, subdirectories, other entries, descriptor)`` per
    directory; a caller may prune the subdirectories in place. A subdirectory is
    opened without following a link, and one removed or replaced by anything but a
    directory during the walk is skipped. Links and special files are other entries.
    """
    yield from _walk(top_fd, PurePosixPath())


def _walk(
    fd: int, rel_dir: PurePosixPath
) -> Iterator[tuple[PurePosixPath, list[str], list[str], int]]:
    dirs: list[str] = []
    others: list[str] = []
    with os.scandir(fd) as entries:
        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except FileNotFoundError:
                continue
            (dirs if is_dir else others).append(entry.name)
    dirs.sort()
    others.sort()
    yield rel_dir, dirs, others, fd
    for name in dirs:
        try:
            child = _open_child(fd, name, create=False, mode=None)
        except (FileNotFoundError, PathRefused):
            continue
        try:
            yield from _walk(child, rel_dir / name)
        finally:
            os.close(child)


def open_regular(dir_fd: int, name: str) -> BinaryIO | None:
    """Open the regular file ``name`` in ``dir_fd`` for reading, or return None when
    it is missing, a link, or not a regular file. A special file is never opened."""
    try:
        before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _SKIPPED_READ_ERRNOS:
            return None
        raise
    if not _same_regular_file(fd, before):
        os.close(fd)
        return None
    os.set_blocking(fd, True)
    return os.fdopen(fd, "rb")


def open_below(base_dir: Path, rel_path: PurePosixPath | Path) -> BinaryIO | None:
    """Open the regular file ``base_dir/rel_path`` for reading without following a
    link at or below ``base_dir``, or return None when it is missing, a link, not a
    regular file, or below one that is not a directory."""
    if not is_plain_segment(rel_path.name):
        return None
    try:
        with open_dir(base_dir, *rel_path.parent.parts) as fd:
            return open_regular(fd, rel_path.name)
    except (FileNotFoundError, PathRefused):
        return None


def open_append(dir_fd: int, name: str) -> BinaryIO:
    """Open the regular file ``name`` in ``dir_fd`` for appending, creating it when
    missing; raises ``PathRefused`` when it is a link or anything but a regular
    file, which is never opened blocking."""
    try:
        before: os.stat_result | None = os.stat(
            name, dir_fd=dir_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        before = None
    if before is not None and not stat.S_ISREG(before.st_mode):
        raise PathRefused(errno.EINVAL, "not a regular file", name)
    try:
        fd = os.open(name, _APPEND_FLAGS, 0o666, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _REFUSED_FILE_ERRNOS:
            raise PathRefused(exc.errno, "not a regular file", name) from exc
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise PathRefused(errno.EINVAL, "not a regular file", name)
    os.set_blocking(fd, True)
    return os.fdopen(fd, "ab")


def _same_regular_file(fd: int, before: os.stat_result) -> bool:
    after = os.fstat(fd)
    return stat.S_ISREG(after.st_mode) and (after.st_dev, after.st_ino) == (
        before.st_dev,
        before.st_ino,
    )


def regular_files(top: Path) -> Iterator[tuple[str, BinaryIO | OSError]]:
    """Yield each regular file under ``top`` by its path relative to ``top``, opened
    for reading, or with the error that kept it from opening.

    ``top``'s parent is trusted, as in ``open_dir``. Nothing is yielded when ``top``
    is missing or a link; links, special files, and names removed during the walk
    are skipped.
    """
    try:
        with open_dir(top) as top_fd:
            for rel_dir, _, others, dir_fd in walk(top_fd):
                for name in others:
                    rel_name = (rel_dir / name).as_posix()
                    try:
                        opened = open_regular(dir_fd, name)
                    except OSError as exc:
                        yield rel_name, exc
                        continue
                    if opened is not None:
                        yield rel_name, opened
    except (FileNotFoundError, PathRefused):
        return
