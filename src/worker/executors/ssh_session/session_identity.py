"""The OS identity a process-mode SSH session logs in as.

A root worker mints a throwaway account per session, so the session is a different uid
from the worker: it cannot read the worker's environment, where the task token and
every third-party API key live. An ACL entry denying that account each of the
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
import tempfile
from collections.abc import Iterable
from pathlib import Path

import psutil

from ..base_executor import ExecutionError
from . import acl

logger = logging.getLogger(__name__)

ACCOUNT_PREFIX = "fmssn"
ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
PRIVSEP_DIR = Path("/run/sshd")
# Debian's on-demand global range: above the uids distributions hand out to
# accounts, so a session never shares a uid with a principal of a shared volume,
# and inside the 65536 uids a user namespace maps.
SESSION_UID_MIN = 61000
SESSION_UID_MAX = 64999
_UID_ATTEMPTS = 16
_WORLD_WRITABLE_DIRS = (
    Path("/", "var", "tmp"),
    Path("/", "dev", "shm"),
    Path("/", "dev", "mqueue"),
)
_SYSV_IPC_DIR = Path("/", "proc", "sysvipc")
# Each /proc/sysvipc table and the column holding its object ids.
_SYSV_IPC_ID_COLUMNS = {"shm": "shmid", "msg": "msqid", "sem": "semid"}
_NOGROUP_GID = 65534
_KILL_GRACE_SEC = 5.0
_KILL_ROUNDS = 10
_KILL_ROUND_SEC = 0.5
# Prologue of every helper run as a session uid: argv is (uid, gid, *args).
_AS_UID_PROLOGUE = (
    "import os, sys\n"
    "uid = int(sys.argv[1])\n"
    "if os.getuid() != uid:\n"
    "    os.setgroups([])\n"
    "    os.setgid(int(sys.argv[2]))\n"
    "    os.setuid(uid)\n"
)
_KILL_ALL_SCRIPT = _AS_UID_PROLOGUE + (
    "import signal\n"
    "try:\n"
    "    os.kill(-1, signal.SIGKILL)\n"
    "except ProcessLookupError:\n"
    "    pass\n"
)
# argv[3:] are "<kind>:<id>" pairs; IPC_RMID is 0.
_REMOVE_IPC_SCRIPT = _AS_UID_PROLOGUE + (
    "import ctypes\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "failed = 0\n"
    "for spec in sys.argv[3:]:\n"
    "    kind, ipc_id = spec.split(':')\n"
    "    if kind == 'shm':\n"
    "        rc = libc.shmctl(int(ipc_id), 0, None)\n"
    "    elif kind == 'msg':\n"
    "        rc = libc.msgctl(int(ipc_id), 0, None)\n"
    "    else:\n"
    "        rc = libc.semctl(int(ipc_id), 0, 0)\n"
    "    if rc != 0:\n"
    "        print(spec, os.strerror(ctypes.get_errno()), file=sys.stderr)\n"
    "        failed = 1\n"
    "sys.exit(failed)\n"
)
_EXEC_AS_SCRIPT = _AS_UID_PROLOGUE + "os.execv(sys.argv[3], sys.argv[3:])\n"
_HELPER_TIMEOUT_SEC = 10.0
_COMMAND_TIMEOUT_SEC = 30.0


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

    @classmethod
    def create(
        cls, name: str, home: Path, denied_roots: Iterable[Path]
    ) -> "SessionAccount":
        """Create the account under a uid no ACL entry on ``denied_roots`` names,
        and deny it each of them, or leave nothing behind.

        A uid another worker sharing a root has denied it is never drawn, so
        lifting this account's entries can never lift that worker's.
        """
        roots = list(denied_roots)
        _ensure_privsep_dir()
        useradd = _require_binary("useradd")
        taken: set[int] = set()
        for root in roots:
            taken |= acl.named_uids(root)
        _add_account(useradd, name, home, frozenset(taken))
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
            account.deny(roots)
            home.mkdir(mode=0o700)
            os.lchown(home, account.uid, account.gid)
        except BaseException:
            account.release()
            raise
        return account

    def deny(self, roots: Iterable[Path]) -> None:
        """Deny this account each root, all or nothing.

        Each entry is recorded before it is written, so a worker that dies midway
        still lifts it.
        """
        applied: list[Path] = []
        try:
            for root in roots:
                acl.record(self.uid, root)
                applied.append(root)
                acl.deny(self.uid, root)
        except ExecutionError as exc:
            for root in applied:
                _revoke(self.uid, root)
            raise ExecutionError(
                f"Could not isolate SSH session account {self.name} from this "
                f"worker's state: {exc}"
            ) from exc

    def grant_read(self, path: Path) -> None:
        """Let this account read a file no other account may."""
        acl.grant_read(self.uid, path)

    def release(self) -> None:
        """Remove the account and everything it holds, or leave it locked and
        denied the worker's state when it cannot be."""
        if not retire_account(self.name, self.uid):
            raise ExecutionError(
                f"SSH session account {self.name} could not be removed; it stays "
                "locked and denied this worker's state until a reap removes it"
            )


def retire_account(name: str, uid: int) -> bool:
    """End every process of an account, delete it, then remove its files from the
    shared temporary directories and lift its denials.

    The denials stay until the account is gone, since lifting them while one of its
    processes lives would hand that process the worker's state. An account that
    cannot be ended or deleted is locked instead, keeping them, and ``False``
    returned so a later reap can finish it.
    """
    if not kill_processes(uid):
        lock_account(name)
        logger.error(
            "SSH session account %s still runs a process after every kill; it stays "
            "locked and denied the worker's state",
            name,
        )
        return False
    if not delete_account(name):
        lock_account(name)
        logger.error(
            "SSH session account %s could not be deleted; it stays locked and denied "
            "the worker's state",
            name,
        )
        return False
    purge_uid(uid)
    for recorded_uid, path in acl.recorded():
        if recorded_uid == uid:
            _revoke(uid, Path(path))
    return True


def reap_stale_accounts() -> bool:
    """Retire every session account, then lift every denial recorded for an account
    that no longer exists; returns whether each account is gone."""
    clean = True
    for entry in pwd.getpwall():
        if not (name := entry.pw_name).startswith(ACCOUNT_PREFIX):
            continue
        if retire_account(name, entry.pw_uid):
            logger.info("Reaped stale SSH session account %s", name)
        else:
            clean = False
    _revoke_orphaned_denies()
    return clean


def purge_uid(uid: int) -> None:
    """Delete what ``uid`` left behind that outlives its processes.

    Session uids are drawn at random and may come round again, so a later session
    must not inherit what an earlier one owned.
    """
    purge_uid_files(uid)
    purge_uid_ipc(uid)


def purge_uid_files(uid: int) -> None:
    """Delete what ``uid`` left in the shared scratch directories."""
    for base in (Path(tempfile.gettempdir()), *_WORLD_WRITABLE_DIRS):
        for current, dirs, files in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = os.path.join(current, name)
                if _owner(path) != uid:
                    continue
                dirs.remove(name)
                if os.path.islink(path):
                    _unlink(path)
                else:
                    remove_tree(Path(path))
            for name in files:
                if _owner(path := os.path.join(current, name)) == uid:
                    _unlink(path)


def purge_uid_ipc(uid: int) -> None:
    """Remove the System V IPC objects ``uid`` owns or created.

    Sessions share the worker's IPC namespace, and these objects persist after
    their creator exits. Only an owner, a creator or a holder of ``CAP_SYS_ADMIN``
    may remove one, and a container's root lacks that capability, so the removal
    runs as ``uid``.
    """
    if uid == 0:
        return
    specs = [
        f"{kind}:{ipc_id}"
        for kind, id_column in _SYSV_IPC_ID_COLUMNS.items()
        for ipc_id in owned_ipc_ids(_read_ipc_table(kind), id_column, uid)
    ]
    if not specs:
        return
    result = _run_as(uid, _REMOVE_IPC_SCRIPT, specs)
    if result is None or result.returncode != 0:
        detail = "" if result is None else _stderr_of(result)
        logger.warning(
            "Failed to remove System V IPC objects of uid %d: %s", uid, detail
        )


def owned_ipc_ids(table: str, id_column: str, uid: int) -> list[int]:
    """Ids in a ``/proc/sysvipc`` table whose owner or creator is ``uid``.

    The creator is matched too because the owner can hand an object to any uid.
    """
    lines = table.splitlines()
    if not lines:
        return []
    header = lines[0].split()
    try:
        id_at = header.index(id_column)
        uid_at = header.index("uid")
        cuid_at = header.index("cuid")
    except ValueError:
        return []
    ids: list[int] = []
    for line in lines[1:]:
        fields = line.split()
        try:
            if uid in (int(fields[uid_at]), int(fields[cuid_at])):
                ids.append(int(fields[id_at]))
        except (IndexError, ValueError):
            continue
    return ids


def _read_ipc_table(kind: str) -> str:
    try:
        return (_SYSV_IPC_DIR / kind).read_text(encoding="utf-8")
    except OSError:
        return ""


def remove_tree(path: Path) -> None:
    """Delete ``path`` and everything below it without following a link or crossing
    a mount, however deep a session nested it."""
    rm = shutil.which("rm")
    if rm is None:
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        _run(
            [rm, "-rf", "--one-file-system", "--", path.as_posix()],
            f"remove {path.as_posix()}",
        )
    except ExecutionError:
        logger.warning("Failed to remove %s", path, exc_info=True)


def lock_account(name: str) -> None:
    """Lock an account so nothing can log in as it."""
    if (usermod := shutil.which("usermod")) is None:
        return
    try:
        _run([usermod, "--lock", "--expiredate", "1", name], f"lock account {name}")
    except ExecutionError:
        logger.warning("Failed to lock SSH session account %s", name)


def kill_processes(uid: int) -> bool:
    """End every process running as ``uid``: a terminate, then kill rounds until none
    is left. Returns whether none is left."""
    if uid == 0:
        return False
    victims = _processes_of(uid)
    for proc in victims:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            continue
    if victims:
        psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
    # A snapshot can miss a process that forks and exits in a loop, so it is only
    # trusted once kill(-1) has left the uid unable to start another.
    for _ in range(_KILL_ROUNDS):
        _kill_all_as(uid)
        if not (survivors := _processes_of(uid)):
            return True
        for proc in survivors:
            try:
                proc.send_signal(signal.SIGKILL)
            except psutil.Error:
                continue
        psutil.wait_procs(survivors, timeout=_KILL_ROUND_SEC)
    return not _processes_of(uid)


def delete_account(name: str) -> bool:
    userdel = shutil.which("userdel")
    if userdel is None:
        logger.warning("userdel is missing; leaving account %s behind", name)
        return False
    try:
        _run([userdel, name], f"delete SSH session account {name}")
    except ExecutionError:
        logger.warning("Failed to delete SSH session account %s", name)
        return False
    return True


def exec_as(uid: int, gid: int, argv: list[str]) -> list[str]:
    """The command line running ``argv`` as ``uid``.

    The child drops to ``uid`` itself rather than through ``subprocess``'s ``user=``,
    which forces a plain ``fork()`` whose atfork handlers crash the child of a
    process running gRPC threads.
    """
    if uid == 0:
        raise ExecutionError("Refusing to run a session helper as root")
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _EXEC_AS_SCRIPT,
        str(uid),
        str(gid),
        *argv,
    ]


def process_identity_available() -> bool:
    """Whether this worker can give a session an account of its own."""
    return os.getuid() == 0 and all(
        shutil.which(binary) for binary in ("useradd", "usermod", "userdel")
    )


def _processes_of(uid: int) -> list[psutil.Process]:
    return [p for p in psutil.process_iter(["uids", "status"]) if _owned_by(p, uid)]


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    return _live_uid(proc) == uid


def _live_uid(proc: psutil.Process) -> int | None:
    """The real uid of ``proc``, or ``None`` once it has exited.

    A zombie is only an exit status waiting for its parent to reap it, and a PID 1
    that never reaps would otherwise keep a session account alive forever.
    """
    try:
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return int(proc.uids().real)
    except (psutil.Error, AttributeError):
        return None


def _kill_all_as(uid: int) -> None:
    """Have the kernel SIGKILL every process of ``uid`` in a single pass.

    A snapshot of the process table cannot catch a process forked after it was
    taken, so a fork loop outruns one. ``kill(-1)`` sent as ``uid`` reaches all of
    that uid's processes at once.
    """
    if os.geteuid() != 0 or uid in (0, os.getuid()):
        return
    result = _run_as(uid, _KILL_ALL_SCRIPT, [])
    if result is not None and result.returncode != 0:
        logger.warning(
            "Signalling every process of uid %d exited %d: %s",
            uid,
            result.returncode,
            _stderr_of(result),
        )


def _run_as(
    uid: int, script: str, args: list[str]
) -> "subprocess.CompletedProcess[bytes] | None":
    """Run ``script`` in a helper interpreter that drops to ``uid`` first.

    The helper drops privileges itself rather than through ``subprocess``'s
    ``user=``, which forces a plain ``fork()`` whose atfork handlers crash the
    child of a process running gRPC threads.
    """
    try:
        return subprocess.run(  # nosec B603 - argv list, no shell=True, the worker's own interpreter
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                script,
                str(uid),
                str(_NOGROUP_GID),
                *args,
            ],
            env={},
            cwd="/",
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_HELPER_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Failed to run a helper as uid %d", uid, exc_info=True)
        return None


def _stderr_of(result: "subprocess.CompletedProcess[bytes]") -> str:
    return result.stderr.decode("utf-8", errors="replace").strip()


def _add_account(
    useradd: str, name: str, home: Path, avoid_uids: frozenset[int]
) -> None:
    """Create ``name`` under a uid drawn at random from the session range, skipping
    one an account, a live process or ``avoid_uids`` holds."""
    detail = ""
    for _ in range(_UID_ATTEMPTS):
        uid = SESSION_UID_MIN + secrets.randbelow(SESSION_UID_MAX - SESSION_UID_MIN + 1)
        if uid in avoid_uids or _uid_exists(uid) or _processes_of(uid):
            continue
        try:
            _run(
                [
                    useradd,
                    "--no-create-home",
                    "--no-user-group",
                    "--uid",
                    str(uid),
                    "--home-dir",
                    home.as_posix(),
                    "--shell",
                    _login_shell(),
                    name,
                ],
                f"create SSH session account {name}",
            )
            return
        except ExecutionError as exc:
            detail = str(exc)
            if _account_exists(name):
                raise
    raise ExecutionError(
        f"Could not find a free uid for SSH session account {name}. {detail}".strip()
    )


def _revoke(uid: int, path: Path) -> None:
    try:
        if path.exists():
            acl.revoke(uid, path)
        acl.forget(uid, path)
    except ExecutionError:
        logger.warning(
            "Failed to lift the SSH session denial of uid %d on %s; the next reap "
            "retries",
            uid,
            path,
        )


def _revoke_orphaned_denies() -> None:
    try:
        records = acl.recorded()
    except ExecutionError:
        logger.warning("Cannot read recorded SSH session ACL entries", exc_info=True)
        return
    for uid, path in records:
        if not _uid_exists(uid):
            _revoke(uid, Path(path))


def _uid_exists(uid: int) -> bool:
    try:
        pwd.getpwuid(uid)
    except KeyError:
        return False
    return True


def _account_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
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
