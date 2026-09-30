"""The OS identity a process-mode SSH session logs in as.

A root worker mints a throwaway account per session, so the session is a different
uid from the worker: it cannot read the worker's environment, where the task token
and every third-party API key live. An ACL entry denying that account each of the
worker's state roots keeps it out of the worker's files, which are shared with other
uids by mode and so readable to any account by default.
"""

import logging
import os
import pwd
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import psutil

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

ACCOUNT_PREFIX = "fmssn"
ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PRIVSEP_DIR = Path("/run/sshd")
# Root-only record of the last uid a session held, so no uid is handed out twice.
UID_HIGH_WATER = Path("/var/lib/flowmesh/ssh-session-uid")
FIRST_SESSION_UID = 200_000
# Where any account may leave files behind.
SHARED_TMP_DIRS = (Path("/", "tmp"), Path("/", "var", "tmp"), Path("/", "dev", "shm"))
_NOGROUP_GID = 65534
_KILL_GRACE_SEC = 5.0
_KILL_ROUNDS = 20
_KILL_ROUND_SEC = 0.1
_COMMAND_TIMEOUT_SEC = 30.0
# A uid no account holds, for probing whether a filesystem takes ACL entries.
_ACL_PROBE_UID = 2**31 - 3


def account_name_for(session_id: str) -> str:
    """Derive a valid Linux account name from a session id."""
    tail = re.sub(r"[^a-z0-9]", "", session_id.lower())[-16:]
    name = f"{ACCOUNT_PREFIX}{tail or secrets.token_hex(4)}"[:31]
    if not ACCOUNT_NAME_RE.match(name):
        raise ExecutionError(f"Cannot derive a valid account name from {session_id!r}")
    return name


class SessionAccount:
    """A throwaway account created for one session, denied the worker's state and
    deleted with the session."""

    def __init__(self, name: str, uid: int, gid: int, home: Path) -> None:
        self.name = name
        self.uid = uid
        self.gid = gid
        self.home = home
        self._denied: list[Path] = []

    @classmethod
    def create(
        cls, name: str, home: Path, state_roots: Iterable[Path]
    ) -> "SessionAccount":
        """Create the account and deny it every state root, or leave nothing behind."""
        _ensure_privsep_dir()
        useradd = _require_binary("useradd")
        _run(
            [
                useradd,
                "--no-create-home",
                "--no-user-group",
                "--uid",
                str(allocate_uid()),
                "--home-dir",
                home.as_posix(),
                "--shell",
                _login_shell(),
                name,
            ],
            f"create SSH session account {name}",
        )
        try:
            entry = pwd.getpwnam(name)
        except KeyError as exc:
            raise ExecutionError(f"SSH session account {name} was not created") from exc
        account = cls(name, entry.pw_uid, entry.pw_gid, home)
        try:
            # A fresh account's shadow entry is "!", which sshd reads as locked and
            # refuses even for public-key auth once UsePAM is off. An unguessable
            # hash leaves it unlocked without granting a usable password.
            _unlock(name)
            account.deny(state_roots)
            home.mkdir(mode=0o700)
            os.lchown(home, account.uid, account.gid)
        except BaseException:
            account.release()
            raise
        return account

    def deny(self, roots: Iterable[Path]) -> None:
        """Deny this account each root, failing on the first one that refuses."""
        setfacl = _require_binary("setfacl")
        for root in roots:
            _run(
                [setfacl, "-m", f"u:{self.uid}:---", root.as_posix()],
                f"deny SSH session account {self.name} {root.as_posix()}",
            )
            self._denied.append(root)

    def grant_read(self, path: Path) -> None:
        """Let this account read a file no other account may."""
        setfacl = _require_binary("setfacl")
        _run(
            [setfacl, "-m", f"u:{self.uid}:r", path.as_posix()],
            f"grant SSH session account {self.name} {path.as_posix()}",
        )

    def release(self) -> None:
        """Remove the account and everything it holds, or leave it locked and
        denied the worker's state when a process of it cannot be ended."""
        if not retire_account(self.name, self.uid, self._denied):
            raise ExecutionError(
                f"SSH session account {self.name} still runs a process it could not "
                "be rid of; it stays locked until a reap removes it"
            )
        self._denied.clear()


def retire_account(name: str, uid: int, roots: Iterable[Path]) -> bool:
    """End every process of an account, then remove its files from the shared
    temporary directories, its denials on ``roots``, and the account itself.

    An account with a process that outlives the kill is locked instead, keeping its
    denials, and ``False`` returned so a later reap can finish it.
    """
    if not kill_processes(uid):
        lock_account(name)
        logger.error(
            "SSH session account %s still runs a process after every kill; it stays "
            "locked and denied the worker's state",
            name,
        )
        return False
    remove_files_of(uid)
    lift_denials(uid, roots)
    delete_account(name)
    return True


def allocate_uid() -> int:
    """A uid no session on this worker has held, so a later session never inherits
    what an earlier one left running or owning."""
    try:
        last = int(UID_HIGH_WATER.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        last = FIRST_SESSION_UID - 1
    uid = max(last + 1, FIRST_SESSION_UID)
    while _uid_in_use(uid):
        uid += 1
    UID_HIGH_WATER.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    pending = UID_HIGH_WATER.with_name(f"{UID_HIGH_WATER.name}.tmp")
    pending.write_text(str(uid), encoding="utf-8")
    pending.chmod(0o600)
    pending.replace(UID_HIGH_WATER)
    return uid


def remove_files_of(uid: int) -> None:
    """Remove what ``uid`` left in the directories every account may write to."""
    for base in SHARED_TMP_DIRS:
        for current, dirs, files in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = os.path.join(current, name)
                if _owner(path) != uid:
                    continue
                dirs.remove(name)
                if os.path.islink(path):
                    _unlink(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
            for name in files:
                if _owner(path := os.path.join(current, name)) == uid:
                    _unlink(path)


def lock_account(name: str) -> None:
    """Lock an account so nothing can log in as it."""
    if (usermod := shutil.which("usermod")) is None:
        return
    try:
        _run([usermod, "--lock", "--expiredate", "1", name], f"lock account {name}")
    except ExecutionError:
        logger.warning("Failed to lock SSH session account %s", name)


def lift_denials(uid: int, roots: Iterable[Path]) -> None:
    """Remove ``uid``'s ACL entry from each root, so a reused uid inherits none."""
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        return
    for root in roots:
        if not root.exists():
            continue
        try:
            _run(
                [setfacl, "-x", f"u:{uid}", root.as_posix()],
                f"lift the SSH session denial on {root.as_posix()}",
            )
        except ExecutionError:
            logger.warning("Failed to lift the SSH session denial on %s", root)


def supports_denials(roots: Iterable[Path]) -> bool:
    """Whether every root takes an ACL entry."""
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        return False
    for root in roots:
        try:
            _run(
                [setfacl, "-m", f"u:{_ACL_PROBE_UID}:---", root.as_posix()],
                f"probe ACL support on {root.as_posix()}",
            )
        except ExecutionError:
            logger.info("SSH session ACLs are unsupported on %s", root)
            return False
        lift_denials(_ACL_PROBE_UID, [root])
    return True


def kill_processes(uid: int) -> bool:
    """End every process running as ``uid``: a terminate, then kills until none is
    left. Returns whether none is left."""
    if not (victims := _processes_of(uid)):
        return True
    for proc in victims:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            continue
    psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
    for _ in range(_KILL_ROUNDS):
        if not _processes_of(uid):
            return True
        _kill_all_as(uid)
        time.sleep(_KILL_ROUND_SEC)
    return not _processes_of(uid)


def delete_account(name: str) -> None:
    userdel = shutil.which("userdel")
    if userdel is None:
        logger.warning("userdel is missing; leaving account %s behind", name)
        return
    try:
        _run([userdel, name], f"delete SSH session account {name}")
    except ExecutionError:
        logger.warning("Failed to delete SSH session account %s", name)


def reap_stale_accounts(state_roots: Iterable[Path]) -> bool:
    """Retire every session account; returns whether each one is gone."""
    roots = list(state_roots)
    clean = True
    for entry in pwd.getpwall():
        if not (name := entry.pw_name).startswith(ACCOUNT_PREFIX):
            continue
        if retire_account(name, entry.pw_uid, roots):
            logger.info("Reaped stale SSH session account %s", name)
        else:
            clean = False
    return clean


def process_identity_available() -> bool:
    """Whether this worker can give a session an account of its own."""
    return os.getuid() == 0 and all(
        shutil.which(binary) for binary in ("useradd", "usermod", "userdel")
    )


def _processes_of(uid: int) -> list[psutil.Process]:
    return [p for p in psutil.process_iter(["uids"]) if _owned_by(p, uid)]


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    try:
        return proc.uids().real == uid
    except (psutil.Error, AttributeError):
        return False


def _kill_all_as(uid: int) -> None:
    """Kill every process of ``uid`` at once, from a child running as ``uid``.

    ``kill(-1)`` signals every process the caller may signal in one step, so a
    process forking as fast as its siblings are killed cannot outrun it.
    """
    if uid == 0 or os.getuid() != 0:
        return
    try:
        subprocess.run(  # nosec B603 - argv list, no shell=True, the running interpreter
            [sys.executable, "-c", "import os, signal; os.kill(-1, signal.SIGKILL)"],
            user=uid,
            group=_NOGROUP_GID,
            extra_groups=[],
            env={"PATH": os.defpath},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_COMMAND_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("Failed to kill the processes of uid %d", uid, exc_info=True)


def _uid_in_use(uid: int) -> bool:
    try:
        pwd.getpwuid(uid)
    except KeyError:
        return bool(_processes_of(uid))
    return True


def _owner(path: str) -> int | None:
    try:
        return os.lstat(path).st_uid
    except OSError:
        return None


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        logger.warning("Failed to remove %s", path)


def _ensure_privsep_dir() -> None:
    """Create sshd's privilege-separation directory, which a root sshd needs."""
    try:
        PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
        os.chown(PRIVSEP_DIR, 0, 0)
        PRIVSEP_DIR.chmod(0o755)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot prepare {PRIVSEP_DIR.as_posix()} for sshd: {exc}"
        ) from exc


def _login_shell() -> str:
    return "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"


def _unlock(name: str) -> None:
    usermod = _require_binary("usermod")
    _run(
        [usermod, "--password", _unusable_password_hash(), name],
        f"unlock SSH session account {name}",
    )


def _unusable_password_hash() -> str:
    """A valid but unguessable hash, so sshd does not treat the account as locked."""
    openssl = shutil.which("openssl")
    if openssl is None:
        # "*" and a leading "!" both read as locked to sshd; a bare salted
        # marker does not, and no password hashes to it.
        return f"$6$nologin${secrets.token_hex(16)}"
    # Hex, so the value never starts with "-", which openssl reads as an option.
    result = _run(
        [openssl, "passwd", "-6", secrets.token_hex(32)],
        "generate an unusable password hash",
    )
    return result.stdout.decode("utf-8", errors="replace").strip() or (
        f"$6$nologin${secrets.token_hex(16)}"
    )


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ExecutionError(
            f"{name} is required to give this SSH session its own account but is "
            "missing from the worker image"
        )
    return path


def _run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        argv, capture_output=True, timeout=_COMMAND_TIMEOUT_SEC, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionError(f"Failed to {what}: {detail}")
    return result
