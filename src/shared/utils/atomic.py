"""Atomic file-write primitives for files written by multiple parties."""

import os
import tempfile
from pathlib import Path

_SHARED_FILE_MODE = 0o0666


def atomic_write_text(target: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Replace ``target`` with ``content`` atomically via tempfile + os.replace.

    The writer only needs write permission on the parent directory, not on
    any pre-existing file (which may be owned by a different UID under a
    shared results volume). The new file is chmodded to 0o0666 so a peer
    UID can replace it on the next call.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        tmp_path.chmod(_SHARED_FILE_MODE)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(target: Path, data: bytes, *, if_absent: bool = False) -> None:
    """Replace ``target`` with ``data`` atomically via tempfile + os.replace.

    Creates the parent directory when missing. With ``if_absent`` an existing file
    is left untouched, so a concurrent or re-driven write of immutable content is a
    no-op rather than a rewrite.
    """
    if if_absent and target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        tmp_path.chmod(_SHARED_FILE_MODE)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
