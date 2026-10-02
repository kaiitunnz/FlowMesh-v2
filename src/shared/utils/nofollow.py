"""Filesystem access below a directory that never follows a link.

A task's results directory is shared with code the task runs, which can replace any
entry under it with a link. These helpers hold each directory open and reach every
name relative to it without following a link, so a link, or a directory swapped for
one, is never chmodded, written through, or read through.
"""

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_REFUSED_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})
_SKIPPED_OPEN_ERRNOS = frozenset({errno.ELOOP, errno.ENXIO, errno.ENOENT})


class LinkRefused(OSError):
    """A path names a link, or a non-directory where a directory belongs."""


@contextmanager
def open_dir(
    base_dir: Path, *parts: str, create: bool = False, mode: int | None = None
) -> Iterator[int]:
    """Open ``base_dir/parts...`` and yield its directory descriptor.

    ``base_dir``'s parent is opened as given; ``base_dir`` and every part below it
    is opened without following a link. With ``create`` each missing directory is
    made, and ``mode``, when given, is set on every directory opened from
    ``base_dir`` down, where the caller may.

    Raises ``LinkRefused`` when any of them is a link or not a directory.
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
    yield its descriptor; raises ``LinkRefused`` as ``open_dir`` does."""
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
    if create:
        try:
            os.mkdir(name, dir_fd=fd)
        except FileExistsError:
            pass
    try:
        child = os.open(name, _DIR_FLAGS, dir_fd=fd)
    except OSError as exc:
        if exc.errno in _REFUSED_ERRNOS:
            raise LinkRefused(exc.errno, "not a directory", name) from exc
        raise
    if mode is not None:
        try:
            os.fchmod(child, mode)
        except PermissionError:
            pass
    return child


def open_regular(dir_fd: int, name: str) -> BinaryIO | None:
    """Open the regular file ``name`` in ``dir_fd`` for reading, or return None when
    it is missing, a link, or not a regular file."""
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _SKIPPED_OPEN_ERRNOS:
            return None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return None
    return os.fdopen(fd, "rb")


def open_below(base_dir: Path, rel_path: Path) -> BinaryIO | None:
    """Open the regular file ``base_dir/rel_path`` for reading without following a
    link at or below ``base_dir``, or return None when it is missing, a link, not a
    regular file, or below one that is not a directory."""
    try:
        with open_dir(base_dir, *rel_path.parent.parts) as fd:
            return open_regular(fd, rel_path.name)
    except (FileNotFoundError, LinkRefused):
        return None


def open_append(dir_fd: int, name: str) -> BinaryIO:
    """Open ``name`` in ``dir_fd`` for appending, creating it when missing; raises
    ``LinkRefused`` when it is a link."""
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o666,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise LinkRefused(exc.errno, "a link", name) from exc
        raise
    return os.fdopen(fd, "ab")


def regular_files(
    top: Path | str, dir_fd: int | None = None
) -> Iterator[tuple[str, BinaryIO | OSError]]:
    """Yield each regular file under ``top`` by its path relative to ``top``, opened
    for reading, or with the error that kept it from opening.

    ``top`` is relative to ``dir_fd`` when given. A link, a socket, and a name
    removed during the walk are skipped.
    """
    for dirpath, dirs, files, walk_fd in os.fwalk(
        top, follow_symlinks=False, dir_fd=dir_fd
    ):
        dirs.sort()
        rel_dir = Path(dirpath).relative_to(top)
        for name in sorted(files):
            rel_name = (rel_dir / name).as_posix()
            try:
                opened = open_regular(walk_fd, name)
            except OSError as exc:
                yield rel_name, exc
                continue
            if opened is not None:
                yield rel_name, opened
