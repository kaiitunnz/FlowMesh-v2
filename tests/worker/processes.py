"""Process-tree observations shared by the worker tests that reap trees."""

from pathlib import Path

import psutil


def running(pid: int) -> bool:
    """Whether a process is alive, a zombie not counting as alive."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def descendants(pid: int) -> set[int]:
    """The pids below a process, or none once it is gone."""
    try:
        return {child.pid for child in psutil.Process(pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def recorded_pids(path: Path) -> list[int]:
    """The pids a test command wrote to ``path``, one per line."""
    return [int(line) for line in path.read_text().split()]
