"""The layout of a task's output directory."""

import os
import shutil
import stat
from pathlib import Path
from typing import BinaryIO

from .atomic import temp_name
from .nofollow import open_dir, open_dir_at, open_regular, walk

RESULTS_NAME = "results.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"
SCRATCH_DIR = "scratch"

_SHARED_DIR_MODE = 0o0777
_COPY_CHUNK_BYTES = 1 << 20


def prepare_output_dir(base_dir: Path) -> None:
    """Ensure the base directory and standard sub-directories exist, writable by peer
    UIDs; raises ``PathRefused`` when any of them is a link."""
    for parts in ((), (LOGS_DIR,), (ARTIFACTS_DIR,)):
        with open_dir(base_dir, *parts, create=True, mode=_SHARED_DIR_MODE):
            pass


def scratch_dir(base_dir: Path) -> Path:
    """Return `out_dir/scratch/`, creating it if needed."""
    with open_dir(base_dir, SCRATCH_DIR, create=True, mode=_SHARED_DIR_MODE):
        pass
    return base_dir / SCRATCH_DIR


def clear_directory(directory: Path) -> None:
    """Remove everything under ``directory``, leaving it empty."""
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def detach_shared_files(top: Path) -> int:
    """Give each file under ``top`` that shares its inode with a name outside ``top``
    a private copy of its own, and return how many inodes were copied.

    Names under ``top`` that link one inode keep linking one copy together. Each name
    is replaced whole, so the inode it shared keeps its bytes for every other name.
    Nothing is followed through a link; ``top``'s parent is trusted, as in
    ``open_dir``, and a missing ``top`` has nothing to detach.
    """
    try:
        with open_dir(top) as top_fd:
            groups: dict[tuple[int, int], list[tuple[str, str]]] = {}
            links: dict[tuple[int, int], int] = {}
            for rel_dir, _, others, dir_fd in walk(top_fd, strict=True):
                for name in others:
                    try:
                        found = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(found.st_mode) or found.st_nlink < 2:
                        continue
                    inode = (found.st_dev, found.st_ino)
                    groups.setdefault(inode, []).append((rel_dir, name))
                    links[inode] = found.st_nlink
            copied = 0
            for inode, names in groups.items():
                if len(names) < links[inode]:
                    _detach_group(top_fd, inode, names)
                    copied += 1
            return copied
    except FileNotFoundError:
        return 0


def _detach_group(
    top_fd: int, inode: tuple[int, int], names: list[tuple[str, str]]
) -> None:
    """Copy one shared inode privately and point every one of ``names`` at the copy."""
    (first_dir, first), *rest = names
    with open_dir_at(top_fd, *_parts(first_dir)) as dir_fd:
        source = open_regular(dir_fd, first)
        if source is None:
            return
        with source:
            held = os.fstat(source.fileno())
            if (held.st_dev, held.st_ino) != inode:
                return
            tmp = _copy_beside(dir_fd, source, held)
        try:
            os.replace(tmp, first, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            os.unlink(tmp, dir_fd=dir_fd)
            raise
        for rel_dir, name in rest:
            with open_dir_at(top_fd, *_parts(rel_dir)) as other_fd:
                alias = temp_name()
                os.link(
                    first,
                    alias,
                    src_dir_fd=dir_fd,
                    dst_dir_fd=other_fd,
                    follow_symlinks=False,
                )
                try:
                    os.replace(alias, name, src_dir_fd=other_fd, dst_dir_fd=other_fd)
                except BaseException:
                    os.unlink(alias, dir_fd=other_fd)
                    raise


def _copy_beside(dir_fd: int, source: BinaryIO, held: os.stat_result) -> str:
    """Copy ``source`` into a new temp file in ``dir_fd`` with its mode, times and,
    where this user may set it, its owner; returns the temp file's name."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    name = temp_name()
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as copy:
            shutil.copyfileobj(source, copy, _COPY_CHUNK_BYTES)
            copy.flush()
            os.fchmod(copy.fileno(), stat.S_IMODE(held.st_mode))
            try:
                os.fchown(copy.fileno(), held.st_uid, held.st_gid)
            except PermissionError:
                pass
            os.utime(copy.fileno(), ns=(held.st_atime_ns, held.st_mtime_ns))
    except BaseException:
        os.unlink(name, dir_fd=dir_fd)
        raise
    return name


def _parts(rel_dir: str) -> list[str]:
    return rel_dir.split("/") if rel_dir else []
