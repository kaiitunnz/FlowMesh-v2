"""Filesystem helpers for tests that link files."""

import os
from pathlib import Path


def other_filesystem(path: Path) -> Path | None:
    """Return a writable directory on a filesystem other than ``path``'s, if any."""
    for candidate in (Path("/dev/shm"), Path("/run/user") / str(os.getuid())):
        try:
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                if os.stat(candidate).st_dev != os.stat(path).st_dev:
                    return candidate
        except OSError:
            continue
    return None
