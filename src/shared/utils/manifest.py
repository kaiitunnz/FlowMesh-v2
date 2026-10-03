"""Manifest helpers used by the worker pipeline when persisting outputs."""

import hashlib
import json
import os
import stat
import threading
import weakref
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

from .atomic import atomic_write_text, is_atomic_temp
from .nofollow import (
    PathRefused,
    is_plain_segment,
    open_dir,
    open_dir_at,
    open_regular,
    walk,
)
from .time import now_iso

MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"
SCRATCH_DIR = "scratch"

_SHARED_DIR_MODE = 0o0777
# A manifest is a rescan of its directory; serializing a directory's rescans makes
# the last manifest written list every file present before its scan. A directory's
# lock lives only while a sync holds it.
_MANIFEST_LOCKS: weakref.WeakValueDictionary[Path, threading.Lock] = (
    weakref.WeakValueDictionary()
)
_MANIFEST_LOCKS_GUARD = threading.Lock()


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


def sync_manifest(
    base_dir: Path, task_id: str, expected: Iterable[str]
) -> dict[str, Any]:
    """
    Build a manifest by reconciling expected versus actual files.
    """
    with _manifest_lock(base_dir):
        return _sync_manifest(base_dir, task_id, expected)


def _manifest_lock(base_dir: Path) -> threading.Lock:
    key = Path(os.path.abspath(base_dir))
    with _MANIFEST_LOCKS_GUARD:
        if (lock := _MANIFEST_LOCKS.get(key)) is None:
            lock = _MANIFEST_LOCKS[key] = threading.Lock()
        return lock


def _sync_manifest(
    base_dir: Path, task_id: str, expected: Iterable[str]
) -> dict[str, Any]:
    prepare_output_dir(base_dir)
    expected_set = {_normalize_artifact_name(item) for item in expected or [] if item}
    expected_set.update({RESULTS_NAME, LOGS_DIR, ARTIFACTS_DIR})

    entries: list[dict[str, Any]] = []
    with open_dir(base_dir) as base_fd:
        for name in sorted(expected_set):
            entries.append(_describe_path(base_fd, Path(name), required=True))
        added = {entry["path"] for entry in entries}

        # Capture additional files/directories that exist but were not declared.
        with os.scandir(base_fd) as listing:
            extra = sorted(
                item.name
                for item in listing
                if item.name not in added
                and item.name != MANIFEST_NAME
                and not is_atomic_temp(item.name)
                and not item.is_symlink()
            )
        for name in extra:
            entries.append(_describe_path(base_fd, Path(name), required=False))

        manifest = {
            "task_id": task_id,
            "generated_at": now_iso(),
            "entries": entries,
        }
        atomic_write_text(
            Path(MANIFEST_NAME),
            json.dumps(manifest, ensure_ascii=False, indent=2),
            dir_fd=base_fd,
        )
    return manifest


# -------------------------
# Helpers
# -------------------------


def _infer_type(rel_path: Path) -> str:
    normalized = rel_path.as_posix()
    if normalized == RESULTS_NAME:
        return "result"
    if normalized.startswith(f"{LOGS_DIR}/") or normalized == LOGS_DIR:
        return "logs"
    if normalized.startswith(f"{ARTIFACTS_DIR}/") or normalized == ARTIFACTS_DIR:
        return "artifact"
    if normalized.startswith(f"{SCRATCH_DIR}/") or normalized == SCRATCH_DIR:
        return "scratch"
    if rel_path.suffix:
        return "artifact"
    return "directory"


def _describe_path(base_fd: int, rel_path: Path, *, required: bool) -> dict[str, Any]:
    entry_type = _infer_type(rel_path)
    entry: dict[str, Any] = {
        "name": rel_path.as_posix(),
        "path": rel_path.as_posix(),
        "type": entry_type,
        "required": required,
    }
    stats = _stats(base_fd, rel_path)
    if stats is None:
        entry["status"] = "missing"
        return entry
    entry["status"] = "present"
    entry["updated_at"] = now_iso()
    entry.update(stats)
    return entry


def _stats(base_fd: int, rel_path: Path) -> dict[str, Any] | None:
    """The size and digest or file count of ``rel_path``, or None when it is missing
    or reached only through a link."""
    if rel_path.is_absolute() or not all(map(is_plain_segment, rel_path.parts)):
        return None
    try:
        with open_dir_at(base_fd, *rel_path.parent.parts) as dir_fd:
            st = os.stat(rel_path.name, dir_fd=dir_fd, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                return None
            if stat.S_ISREG(st.st_mode):
                if (opened := open_regular(dir_fd, rel_path.name)) is None:
                    return None
                with opened as fh:
                    return {
                        "size": os.fstat(fh.fileno()).st_size,
                        "sha256": _sha256(fh),
                    }
            if not stat.S_ISDIR(st.st_mode):
                return {"size": 0, "file_count": 0}
            with open_dir_at(dir_fd, rel_path.name) as top_fd:
                size, count = _directory_stats(top_fd)
            return {"size": size, "file_count": count}
    except (FileNotFoundError, PathRefused):
        return None


def _normalize_artifact_name(name: str) -> str:
    value = name.strip()
    if value.endswith("/"):
        value = value.rstrip("/")
    if value.startswith("./"):
        value = value[2:]
    return value or name


def _sha256(fh: BinaryIO) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: fh.read(8192), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def _directory_stats(top_fd: int) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    for _, dirs, others, dir_fd in walk(top_fd):
        dirs[:] = [name for name in dirs if not is_atomic_temp(name)]
        for name in others:
            if is_atomic_temp(name):
                continue
            try:
                st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(st.st_mode):
                total_size += st.st_size
                file_count += 1
    return total_size, file_count
