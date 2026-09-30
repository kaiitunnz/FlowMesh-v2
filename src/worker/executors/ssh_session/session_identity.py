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
from collections.abc import Iterable
from pathlib import Path

import psutil

from ..base_executor import ExecutionError

logger = logging.getLogger(__name__)

ACCOUNT_PREFIX = "fmssn"
ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PRIVSEP_DIR = Path("/run/sshd")
_KILL_GRACE_SEC = 5.0
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
            if not root.exists():
                continue
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
        """End every process of the account, lift its denials, and delete it."""
        kill_processes(self.uid)
        lift_denials(self.uid, self._denied)
        self._denied.clear()
        delete_account(self.name)


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
    """Whether every existing root takes an ACL entry."""
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        return False
    for root in roots:
        if not root.exists():
            continue
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


def kill_processes(uid: int) -> None:
    """Terminate every process running as ``uid``, escalating after a grace period."""
    victims = [p for p in psutil.process_iter(["uids"]) if _owned_by(p, uid)]
    for proc in victims:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            continue
    if not victims:
        return
    _, alive = psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
    for proc in alive:
        try:
            proc.send_signal(signal.SIGKILL)
        except psutil.Error:
            continue


def delete_account(name: str) -> None:
    userdel = shutil.which("userdel")
    if userdel is None:
        logger.warning("userdel is missing; leaving account %s behind", name)
        return
    try:
        _run([userdel, name], f"delete SSH session account {name}")
    except ExecutionError:
        logger.warning("Failed to delete SSH session account %s", name)


def reap_stale_accounts(state_roots: Iterable[Path], keep: str | None = None) -> None:
    """Remove every session account but ``keep``: its processes, its denials, and the
    account itself."""
    roots = list(state_roots)
    for entry in pwd.getpwall():
        name = entry.pw_name
        if not name.startswith(ACCOUNT_PREFIX) or name == keep:
            continue
        kill_processes(entry.pw_uid)
        lift_denials(entry.pw_uid, roots)
        delete_account(name)
        logger.info("Reaped stale SSH session account %s", name)


def process_identity_available() -> bool:
    """Whether this worker can give a session an account of its own."""
    return os.getuid() == 0 and all(
        shutil.which(binary) for binary in ("useradd", "usermod", "userdel")
    )


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    try:
        return proc.uids().real == uid
    except (psutil.Error, AttributeError):
        return False


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
    for candidate in ("/bin/bash", "/bin/sh"):
        if Path(candidate).exists():
            return candidate
    return "/bin/sh"


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
    result = _run(
        [openssl, "passwd", "-6", secrets.token_urlsafe(32)],
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
