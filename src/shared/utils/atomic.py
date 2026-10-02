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


def atomic_write_bytes(target: Path, data: bytes, *, if_absent: bool = False) -> None:
    """Replace ``target`` with ``data`` atomically via tempfile + os.replace.

    Creates the parent directory when missing. With ``if_absent`` an existing file
    is left untouched, so a concurrent or re-driven write of immutable content is a
    no-op rather than a rewrite.
    """
    if if_absent and target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(target, lambda fh: fh.write(data), None)


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
        tmp_name = f"{_TEMP_PREFIX}{os.urandom(6).hex()}"
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
