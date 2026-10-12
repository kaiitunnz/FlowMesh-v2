"""Atomic file-write primitives for files written by multiple parties."""

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

_SHARED_FILE_MODE = 0o0666
_COPY_CHUNK_BYTES = 1 << 20
_TEMP_PREFIX = ".fm-tmp-"


def is_atomic_temp(name: str) -> bool:
    """Return whether ``name`` is an in-flight atomic write's temp file."""
    return name.startswith(_TEMP_PREFIX)


def atomic_write_text(
    target: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    dir_fd: int | None = None,
) -> None:
    """Replace ``target`` with ``content`` atomically via a temp file + os.replace.

    The writer only needs write permission on the parent directory, not on
    any pre-existing file (which may be owned by a different UID under a
    shared results volume). The new file is chmodded to 0o0666 so a peer
    UID can replace it on the next call. With ``dir_fd``, ``target`` is a name in
    that directory.
    """
    _atomic_replace(target, lambda fh: fh.write(content.encode(encoding)), dir_fd)


def _temp_name() -> str:
    return f"{_TEMP_PREFIX}{os.urandom(6).hex()}"


def temp_path(directory: Path) -> Path:
    """A fresh in-flight temp file name in ``directory``."""
    return directory / _temp_name()


def write_replacing(dest: Path, write: Callable[[BinaryIO], Any]) -> None:
    """Replace ``dest`` with what ``write`` puts in a temp file beside it.

    ``dest`` only ever appears whole, with its bytes on disk; a write that fails leaves
    nothing behind. The rename itself is not flushed, so ``dest`` may be absent after a
    crash.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(dest.parent)
    try:
        with tmp.open("xb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def link_no_replace(tmp: Path, target: Path, *, durable: bool = False) -> bool:
    """Move a finished temp file to ``target`` only where nothing is, and return
    whether it landed.

    The link is the commit: it returns False rather than replacing a file already
    there, including one a concurrent writer landed a moment earlier. The temp name is
    gone either way. With ``durable`` the file at ``target`` and its directory entry
    are on disk before this returns, whichever writer's link it is.
    """
    try:
        if durable:
            _fsync_file(tmp)
        created = _make_parents(target.parent)
        try:
            os.link(tmp, target)
            landed = True
        except FileExistsError:
            landed = False
            if durable:
                # Another writer's link may not have reached the disk yet, and this
                # writer is about to report the object as stored.
                _fsync_file(target)
        if durable:
            # A directory this commit created is only reachable once its own entry in
            # its parent is on disk too.
            for directory in (target.parent, *(d.parent for d in created)):
                _fsync_dir(directory)
        return landed
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def _make_parents(directory: Path) -> list[Path]:
    """Create ``directory`` and any missing ancestors, returning those it created."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for path in reversed(missing):
        path.mkdir(exist_ok=True)
    return missing


def _fsync_dir(directory: Path) -> None:
    """Flush a directory's entries to disk."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_stream(
    target: Path, source: BinaryIO, *, dir_fd: int | None = None
) -> None:
    """Replace ``target`` with the rest of ``source`` atomically, copying it in
    bounded chunks. Without ``dir_fd`` the parent directory is created when missing;
    with it, ``target`` is a name in that directory."""
    if dir_fd is None:
        target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(
        target, lambda fh: shutil.copyfileobj(source, fh, _COPY_CHUNK_BYTES), dir_fd
    )


def _atomic_replace(
    target: Path, write: Callable[[BinaryIO], Any], dir_fd: int | None
) -> None:
    if dir_fd is not None:
        _replace_at(dir_fd, target.name, write)
        return
    parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        _replace_at(parent, target.name, write)
    finally:
        os.close(parent)


def _replace_at(dir_fd: int, name: str, write: Callable[[BinaryIO], Any]) -> None:
    """Write a temp file in ``dir_fd`` and rename it over ``name``; a link at
    ``name`` is replaced, never written through."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    while True:
        tmp_name = _temp_name()
        try:
            fd = os.open(tmp_name, flags, 0o600, dir_fd=dir_fd)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            os.fchmod(fh.fileno(), _SHARED_FILE_MODE)
        os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        raise
