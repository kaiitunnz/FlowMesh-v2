"""The layout of a task's output directory."""

import shutil
from pathlib import Path

from .nofollow import open_dir

RESULTS_NAME = "results.json"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"
SCRATCH_DIR = "scratch"

_SHARED_DIR_MODE = 0o0777


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
