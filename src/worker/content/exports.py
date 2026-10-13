"""Copying a completing agent's declared exports out of its workspace.

The copy reads the workspace through descriptors that never follow a link, after every
writer the step started is proved stopped, and lands under the task's artifacts, where
the task publishes it like any other artifact. Anything an agent could plant to reach
past its workspace or to alias other bytes ends the copy refused: a link, a special
file, a file with a second hard link, an in-flight write, or a path the fabric never
exports. A declared path that does not exist ends it as missing.
"""

import os
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path

from shared.tasks.components.exports import (
    EXPORT_MISSING,
    EXPORT_UNSAFE,
    AgentExport,
    ExportFailure,
    excluded_export_path,
)
from shared.utils.atomic import is_atomic_temp
from shared.utils.nofollow import (
    PathRefused,
    join_relative,
    open_dir,
    open_dir_at,
    open_regular,
    walk,
)

_COPY_BYTES = 1 << 20


class ExportFailed(RuntimeError):
    """A declared export could not be copied out."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.failure = ExportFailure(kind=kind, message=message)


def _unsafe(message: str) -> ExportFailed:
    return ExportFailed(EXPORT_UNSAFE, message)


def spool_exports(
    workspace: Path, exports: Sequence[AgentExport], artifacts_dir: Path
) -> None:
    """Copy each export from ``workspace`` to ``artifacts_dir`` under its name.

    The copies land whole or not at all: they are staged beside the artifacts and
    replace any earlier copy under the same names only once every export copied.
    """
    staging = artifacts_dir.parent / ".export-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        try:
            with open_dir(workspace) as workspace_fd:
                for export in exports:
                    _copy_export(workspace_fd, export, staging)
        except (FileNotFoundError, PathRefused) as exc:
            raise ExportFailed(EXPORT_MISSING, f"the workspace is unreadable: {exc}")
        for export in exports:
            name = export.published_name
            dest = artifacts_dir / name
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink(missing_ok=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).rename(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _copy_export(workspace_fd: int, export: AgentExport, staging: Path) -> None:
    path = export.path
    if (excluded := excluded_export_path(path)) is not None:
        raise _unsafe(f"export {path!r} lies under {excluded!r}")
    *parents, last = path.split("/")
    dest = staging / export.published_name
    try:
        with open_dir_at(workspace_fd, *parents) as parent_fd:
            try:
                st = os.stat(last, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise ExportFailed(EXPORT_MISSING, f"export {path!r} does not exist")
            if stat.S_ISREG(st.st_mode):
                _copy_file(parent_fd, last, dest, path)
            elif stat.S_ISDIR(st.st_mode):
                with open_dir_at(parent_fd, last) as top_fd:
                    _copy_tree(top_fd, dest, path)
            else:
                raise _unsafe(f"export {path!r} is not a regular file or directory")
    except FileNotFoundError:
        raise ExportFailed(EXPORT_MISSING, f"export {path!r} does not exist")
    except PathRefused as exc:
        raise _unsafe(f"export {path!r} is reached through a link: {exc}")


def _copy_tree(top_fd: int, dest: Path, path: str) -> None:
    dest.mkdir(parents=True)
    for rel_dir, dirs, others, dir_fd in walk(top_fd):
        for name in dirs:
            _check_entry(path, join_relative(rel_dir, name))
            (dest / rel_dir / name).mkdir()
        for name in others:
            rel = join_relative(rel_dir, name)
            _check_entry(path, rel)
            _copy_file(dir_fd, name, dest / rel, f"{path}/{rel}")


def _check_entry(path: str, rel: str) -> None:
    full = f"{path}/{rel}"
    if is_atomic_temp(rel.rsplit("/", 1)[-1]):
        raise _unsafe(f"export entry {full!r} is an in-flight write")
    if (excluded := excluded_export_path(full)) is not None:
        raise _unsafe(f"export entry {full!r} lies under {excluded!r}")


def _copy_file(dir_fd: int, name: str, dest: Path, path: str) -> None:
    opened = open_regular(dir_fd, name)
    if opened is None:
        raise _unsafe(f"export entry {path!r} is not a regular file")
    with opened as source:
        st = os.fstat(source.fileno())
        if st.st_nlink > 1:
            raise _unsafe(f"export entry {path!r} has another hard link")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as target:
            copied = 0
            while chunk := source.read(_COPY_BYTES):
                target.write(chunk)
                copied += len(chunk)
        if copied != st.st_size:
            raise _unsafe(f"export entry {path!r} changed while it was copied")


__all__ = ["ExportFailed", "spool_exports"]
