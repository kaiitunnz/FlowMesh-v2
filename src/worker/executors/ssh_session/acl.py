"""POSIX ACL entries that keep a session account out of the worker's state.

A named-user ACL entry is checked before a file's "other" bits, so
``u:<uid>:---`` on a directory stops that one account from traversing into it
however permissive the modes below it are, and changes nothing for any other
principal sharing the directory.

Every entry applied is recorded before it is written, in a root-only state
file, so a worker that crashed mid-session can revoke exactly its own entries
and never one a peer worker applied to a shared volume. The record keeps the
mask the path had, which the entry leaves as it is and its revoke restores.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import NamedTuple

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

STATE_DIR = Path("/var/lib/flowmesh")
DENY_RECORD = STATE_DIR / "ssh-session-denies"
PROBE_UID = 65534
_ACL_TIMEOUT_SEC = 30.0
_DENIED_USER_RE = re.compile(r"^user:(\d+):---$")
_NAMED_USER_RE = re.compile(r"^(?:default:)?user:(\d+):")
_MASK_RE = re.compile(r"^mask::([rwx-]{3})$")
_NO_MASK = "-"
_record_lock = threading.Lock()


def find_setfacl() -> str | None:
    return shutil.which("setfacl")


def find_getfacl() -> str | None:
    return shutil.which("getfacl")


def tools_available() -> bool:
    return find_setfacl() is not None and find_getfacl() is not None


class Denial(NamedTuple):
    """A deny entry for ``uid`` on ``path``, and the mask ``path`` had before it."""

    uid: int
    path: str
    mask: str | None


def deny(uid: int, path: Path) -> None:
    """Deny ``uid`` every access to ``path``, leaving its mask as it is."""
    _setfacl(path, "-n", "-m", f"u:{uid}:---")


def grant_read(uid: int, path: Path) -> None:
    """Grant ``uid`` read access to ``path``."""
    _setfacl(path, "-m", f"u:{uid}:r")


def revoke(uid: int, path: Path, mask: str | None) -> None:
    """Remove ``uid``'s entry from ``path`` and put back the ``mask`` it had.

    Where it had none, the mask is dropped once no named entry needs it, since a
    mask left behind would make a later ``chmod`` on ``path`` change the mask
    instead of the group bits; ``setfacl`` refuses that while one does, which is
    the case to leave alone.
    """
    _setfacl(path, "-n", "-x", f"u:{uid}")
    if mask is not None:
        _setfacl(path, "-n", "-m", f"m::{mask}")
        return
    setfacl = _require(find_setfacl(), "setfacl")
    subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [setfacl, "-x", "m::", "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )


def mask(path: Path) -> str | None:
    """Return ``path``'s ACL mask, or ``None`` when it has none."""
    return parse_mask(_read_acl(path))


def denied_uids(path: Path) -> set[int]:
    """Return the uids that ``path``'s access ACL denies all permissions."""
    return parse_denied_uids(_read_acl(path))


def named_uids(path: Path) -> set[int]:
    """Return the uids named by a user entry in ``path``'s access or default ACL."""
    return parse_named_uids(_read_acl(path))


def parse_denied_uids(getfacl_output: str) -> set[int]:
    return _parse_uids(_DENIED_USER_RE, getfacl_output)


def parse_named_uids(getfacl_output: str) -> set[int]:
    return _parse_uids(_NAMED_USER_RE, getfacl_output)


def parse_mask(getfacl_output: str) -> str | None:
    for line in getfacl_output.splitlines():
        if match := _MASK_RE.match(line.strip()):
            return match.group(1)
    return None


def _parse_uids(pattern: re.Pattern[str], getfacl_output: str) -> set[int]:
    return {
        int(match.group(1))
        for line in getfacl_output.splitlines()
        if (match := pattern.match(line.strip()))
    }


def _read_acl(path: Path, *flags: str) -> str:
    getfacl = _require(find_getfacl(), "getfacl")
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [getfacl, "-n", "-c", "-p", *flags, "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutionError(
            f"Failed to read the ACL of {path.as_posix()}: {_stderr(result)}"
        )
    return result.stdout.decode("utf-8", errors="replace")


def probe(directory: Path) -> None:
    """Check that the filesystem backing ``directory`` stores a deny entry.

    The probe file is written and read without following a link, since another
    account may swap it for one in a directory it can write.
    """
    fd, name = tempfile.mkstemp(prefix=".flowmesh-acl-probe-", dir=directory)
    os.close(fd)
    target = Path(name)
    try:
        _setfacl(target, "-P", "-m", f"u:{PROBE_UID}:---")
        if PROBE_UID not in parse_denied_uids(_read_acl(target, "-P")):
            raise ExecutionError(
                f"ACL entries do not persist on the filesystem of {directory}"
            )
    finally:
        target.unlink(missing_ok=True)


def record(denial: Denial) -> None:
    """Note that ``denial`` is about to be applied."""
    with _record_lock:
        entries = {
            entry
            for entry in _read_records()
            if (entry.uid, entry.path) != (denial.uid, denial.path)
        }
        entries.add(denial)
        _write_records(entries)


def forget(denial: Denial) -> None:
    with _record_lock:
        entries = {
            entry
            for entry in _read_records()
            if (entry.uid, entry.path) != (denial.uid, denial.path)
        }
        _write_records(entries)


def recorded() -> set[Denial]:
    with _record_lock:
        return _read_records()


def _read_records() -> set[Denial]:
    try:
        text = DENY_RECORD.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise ExecutionError(f"Cannot read {DENY_RECORD.as_posix()}: {exc}") from exc
    entries: set[Denial] = set()
    for line in text.splitlines():
        uid, path, mask = [*line.split("\t", 2), _NO_MASK][:3]
        if uid.isdigit() and path:
            entries.add(Denial(int(uid), path, None if mask == _NO_MASK else mask))
    return entries


def _write_records(entries: set[Denial]) -> None:
    try:
        STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".ssh-session-denies-", dir=STATE_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.writelines(
                f"{uid}\t{path}\t{mask or _NO_MASK}\n"
                for uid, path, mask in sorted(entries, key=lambda e: (e.uid, e.path))
            )
        os.replace(tmp_name, DENY_RECORD)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot record SSH session ACL entries in {DENY_RECORD.as_posix()}: "
            f"{exc}"
        ) from exc


def _setfacl(path: Path, *options: str) -> None:
    setfacl = _require(find_setfacl(), "setfacl")
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [setfacl, *options, "--", path.as_posix()],
        capture_output=True,
        timeout=_ACL_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutionError(
            f"Failed to update the ACL of {path.as_posix()}: {_stderr(result)}"
        )


def _require(path: str | None, name: str) -> str:
    if path is None:
        raise ExecutionError(
            f"{name} is required to isolate SSH sessions but is missing from the "
            "worker image"
        )
    return path


def _stderr(result: "subprocess.CompletedProcess[bytes]") -> str:
    return result.stderr.decode("utf-8", errors="replace").strip()
