"""The OS identity a process-mode SSH session logs in as.

A root worker mints a throwaway account per session, so the session cannot read the
worker's environment, where the task token and every third-party API key live, and
an ACL entry on each of the worker's state roots denies it the worker's files.
"""

import fcntl
import grp
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
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
_KILL_GRACE_SEC = 5.0
_KILL_ROUNDS = 10
_KILL_ROUND_SEC = 0.5
# Prepended to every helper script. It imports what the scripts use before
# switching to the uid and gid in argv[1:3], since that uid may be unable to read
# the interpreter's standard library.
_AS_UID_PREAMBLE = (
    "import ctypes, os, signal, sys\n"
    "uid = int(sys.argv[1])\n"
    "if uid == 0:\n"
    "    sys.exit('refusing to run a session helper as root')\n"
    "if os.getuid() != uid:\n"
    "    os.setgroups([])\n"
    "    os.setgid(int(sys.argv[2]))\n"
    "    os.setuid(uid)\n"
    "if os.getresuid() != (uid, uid, uid):\n"
    "    sys.exit('could not switch to the session uid')\n"
)
_KILL_ALL_SCRIPT = (
    "try:\n"
    "    os.kill(-1, signal.SIGKILL)\n"
    "except ProcessLookupError:\n"
    "    pass\n"
)
# argv[3:] is "<kind>:<id>" per object; 0 is IPC_RMID.
_REMOVE_IPC_SCRIPT = (
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
# argv[3:] is the command to exec as the uid.
_EXEC_SCRIPT = "os.execv(sys.argv[3], sys.argv[3:])\n"
_AS_UID_TIMEOUT_SEC = 10.0
_NOGROUP_GID = 65534
_USERADD_TIMEOUT_SEC = 30.0
_SCRATCH_DIRS = (
    Path("/", "tmp"),
    Path("/", "var", "tmp"),
    Path("/", "dev", "shm"),
    Path("/", "dev", "mqueue"),
    Path("/", "run", "lock"),
)
_SYSV_IPC_DIR = Path("/", "proc", "sysvipc")
_SYSV_IPC_ID_COLUMNS = {"shm": "shmid", "msg": "msqid", "sem": "semid"}
_LOCK_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_ROOT_LOCK_TIMEOUT_SEC = 30.0
_ROOT_LOCK_POLL_SEC = 0.05


def account_name_for(session_id: str) -> str:
    """Return a valid Linux account name derived from ``session_id``."""
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
        """Create the account and deny it ``denied_roots``, or leave nothing behind.

        It gets a uid no entry on those roots names, so its deny and revoke never
        replace or remove an entry someone else set, and a group of its own that no
        entry on them names, so no group grants it anything.
        """
        roots = list(denied_roots)
        _ensure_privsep_dir()
        # Workers in other containers draw from the same range and see only each
        # other's entries, so the draw and the denial it rests on happen at once.
        with _locked(roots):
            taken_uids: set[int] = set()
            taken_gids: set[int] = set()
            for root in roots:
                taken_uids |= acl.named_uids(root)
                taken_gids |= acl.named_gids(root)
            _add_account(name, home, frozenset(taken_uids), frozenset(taken_gids))
            try:
                entry = pwd.getpwnam(name)
            except KeyError as exc:
                raise ExecutionError(
                    f"SSH session account {name} was not created"
                ) from exc
            account = cls(name, entry.pw_uid, entry.pw_gid, home)
            try:
                # A fresh account's shadow entry is "!", which sshd reads as locked
                # and refuses even for public-key auth once UsePAM is off. An
                # unguessable hash leaves it unlocked without granting a password.
                _unlock(name)
                account.deny(roots)
                home.mkdir(mode=0o700)
                os.lchown(home, account.uid, account.gid)
            except BaseException:
                account.release()
                raise
        return account

    def deny(self, roots: Iterable[Path]) -> None:
        """Deny this account each of ``roots``, all or nothing."""
        applied: list[acl.Denial] = []
        try:
            for root in roots:
                denial = acl.Denial(self.uid, root.as_posix(), acl.mask(root))
                acl.record(denial)
                applied.append(denial)
                acl.deny(self.uid, root)
        except ExecutionError as exc:
            for denial in applied:
                _revoke(denial)
            raise ExecutionError(
                f"Could not isolate SSH session account {self.name} from this "
                f"worker's state: {exc}"
            ) from exc

    def grant_read(self, path: Path) -> None:
        """Grant this account read access to ``path``."""
        acl.grant_read(self.uid, path)

    def release(self) -> None:
        """Retire the account, raising if it stays locked instead."""
        if not retire_account(self.name, self.uid):
            raise ExecutionError(
                f"SSH session account {self.name} could not be removed; it stays "
                "locked and denied this worker's state until a reap removes it"
            )


@contextmanager
def _locked(roots: Iterable[Path]) -> Iterator[None]:
    """Hold an exclusive ``flock`` on each of ``roots``.

    They are taken in inode order, which every container agrees on whatever path it
    mounts a root at. A root whose filesystem refuses a directory lock, as NFS does,
    goes unlocked.
    """
    fds: list[tuple[tuple[int, int], int, Path]] = []
    try:
        for root in roots:
            try:
                fd = os.open(root, _LOCK_DIR_FLAGS)
            except OSError:
                logger.debug("Cannot open %s to lock it", root, exc_info=True)
                continue
            info = os.fstat(fd)
            fds.append(((info.st_dev, info.st_ino), fd, root))
        fds.sort(key=lambda held: held[0])
        seen: set[tuple[int, int]] = set()
        for key, fd, root in fds:
            if key not in seen:
                seen.add(key)
                _flock(fd, root)
        yield
    finally:
        for _, fd, _ in fds:
            os.close(fd)


def _flock(fd: int, root: Path) -> None:
    deadline = time.monotonic() + _ROOT_LOCK_TIMEOUT_SEC
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise ExecutionError(
                    f"Timed out waiting for another worker to finish a session "
                    f"account on {root}",
                    retryable=True,
                ) from None
            time.sleep(_ROOT_LOCK_POLL_SEC)
        except OSError:
            logger.debug("Cannot lock %s; drawing without it", root, exc_info=True)
            return


def retire_account(name: str, uid: int) -> bool:
    """Kill every process of the account, delete it, purge what its uid left, and
    revoke its recorded ACL entries; return whether it is gone.

    An account whose processes survive or that cannot be deleted is locked and
    keeps its entries, since a live process of it must stay denied. An account
    already deleted counts as gone, and once another account holds its uid, the
    entries recorded for that uid are the other account's.
    """
    if not _account_exists(name) and _uid_exists(uid):
        return True
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
    try:
        records = acl.recorded()
    except ExecutionError:
        logger.warning(
            "Cannot read the recorded SSH session ACL entries; the next reap lifts "
            "those of uid %d",
            uid,
            exc_info=True,
        )
        return True
    for denial in records:
        if denial.uid == uid:
            _revoke(denial)
    return True


def reap_stale_accounts() -> bool:
    """Retire every session account, then revoke recorded ACL entries whose uid no
    longer exists; return whether every account is gone."""
    clean = True
    for entry in pwd.getpwall():
        if not (name := entry.pw_name).startswith(ACCOUNT_PREFIX):
            continue
        if retire_account(name, entry.pw_uid):
            logger.info("Reaped stale SSH session account %s", name)
        else:
            clean = False
    for group in grp.getgrall():
        if group.gr_name.startswith(ACCOUNT_PREFIX) and not _account_exists(
            group.gr_name
        ):
            _delete_group(group.gr_name)
    _revoke_orphaned_denies()
    return clean


def purge_uid(uid: int) -> None:
    """Delete the files and System V IPC objects ``uid`` left where every session
    can reach them.

    Session uids are reused, so a later session with the same uid must not find
    them. A failure is logged, and does not stop the rest.
    """
    for purge in (purge_uid_files, purge_uid_ipc):
        try:
            purge(uid)
        except Exception:
            logger.warning("Failed to purge what uid %d left", uid, exc_info=True)


def purge_uid_files(uid: int) -> None:
    """Delete everything ``uid`` owns in the shared scratch directories.

    Links are removed, never followed.
    """
    for base in scratch_dirs():
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


def scratch_dirs() -> list[Path]:
    """Return the directories every account may write in, the temp dir first."""
    return list(dict.fromkeys((Path(tempfile.gettempdir()), *_SCRATCH_DIRS)))


def purge_uid_ipc(uid: int) -> None:
    """Remove the System V IPC objects that ``uid`` owns or created.

    Removal runs as ``uid``, since removing another user's object needs
    ``CAP_SYS_ADMIN``, which a container's root lacks. Failures are logged, not
    raised.
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
    """Return the ids in a ``/proc/sysvipc`` table owned or created by ``uid``.

    The creator is matched because an owner can give an object to another uid.
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
    """Delete ``path`` and everything below it, never following a link or crossing a
    mount.

    ``rm`` removes a tree of any depth, where ``shutil.rmtree`` runs out of file
    descriptors or recursion on one a session nested thousands deep.
    """
    rm = shutil.which("rm")
    if rm is None:
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        # A tree of millions of files takes minutes, so it gets no timeout.
        _run(
            [rm, "-rf", "--one-file-system", "--", path.as_posix()],
            f"remove {path.as_posix()}",
            timeout=None,
        )
    except ExecutionError:
        logger.warning("Failed to remove %s", path, exc_info=True)


def lock_account(name: str) -> None:
    """Lock ``name`` and expire it, so nothing can log in as it."""
    if (usermod := shutil.which("usermod")) is None:
        return
    try:
        _run([usermod, "--lock", "--expiredate", "1", name], f"lock account {name}")
    except ExecutionError:
        logger.warning("Failed to lock SSH session account %s", name)


def kill_processes(uid: int) -> bool:
    """SIGTERM every process of ``uid``, then SIGKILL until none is left; return
    whether none is."""
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
    # A snapshot can miss a fork loop's children; kill(-1) as the uid reaches all of
    # them in one pass, so only a snapshot taken after it is trusted.
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
    """Delete account ``name`` and its group; return whether both are gone."""
    userdel = shutil.which("userdel")
    if userdel is None:
        logger.warning("userdel is missing; leaving account %s behind", name)
        return False
    try:
        _run([userdel, name], f"delete SSH session account {name}")
    except ExecutionError:
        if _account_exists(name):
            logger.warning("Failed to delete SSH session account %s", name)
            return False
    return _delete_group(name)


def exec_as(uid: int, gid: int, argv: list[str]) -> list[str]:
    """Return the command line that runs ``argv`` as ``uid`` and ``gid``."""
    if uid == 0:
        raise ExecutionError("Refusing to run a session helper as root")
    return interpreter_argv(_AS_UID_PREAMBLE + _EXEC_SCRIPT, str(uid), str(gid), *argv)


def interpreter_argv(script: str, *args: str) -> list[str]:
    """Return the command line that runs ``script`` with ``args`` in the worker's
    own interpreter, isolated from the environment and ``site``."""
    return [sys.executable, "-I", "-S", "-c", script, *args]


def process_identity_available() -> bool:
    """Return whether this worker can create a session account."""
    return os.getuid() == 0 and all(
        shutil.which(binary)
        for binary in ("useradd", "usermod", "userdel", "groupadd", "groupdel")
    )


def _processes_of(uid: int) -> list[psutil.Process]:
    return [p for p in psutil.process_iter(["uids", "status"]) if _owned_by(p, uid)]


def _owned_by(proc: psutil.Process, uid: int) -> bool:
    return _live_uid(proc) == uid


def _live_uid(proc: psutil.Process) -> int | None:
    """Return the real uid of ``proc``, or ``None`` if it has exited or cannot be
    read.

    A zombie counts as exited, so an init that never reaps cannot keep a session
    account alive.
    """
    try:
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return int(proc.uids().real)
    except (psutil.Error, AttributeError):
        return None


def _kill_all_as(uid: int) -> None:
    """SIGKILL every process of ``uid`` with one ``kill(-1)`` sent as ``uid``.

    Does nothing unless the worker is root and ``uid`` is another user.
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
    """Run ``_AS_UID_PREAMBLE`` then ``script`` in a new interpreter, with ``uid``,
    a gid and ``args`` on its argv; return ``None`` if it cannot start.

    The interpreter starts as the worker and the preamble switches to ``uid``,
    because ``subprocess``'s ``user=`` forces a plain ``fork()``, which gRPC's fork
    handlers crash.
    """
    try:
        return subprocess.run(  # nosec B603 - argv list, no shell=True, the worker's own interpreter
            interpreter_argv(
                _AS_UID_PREAMBLE + script, str(uid), str(_NOGROUP_GID), *args
            ),
            env={},
            cwd="/",
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_AS_UID_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Failed to run a helper as uid %d", uid, exc_info=True)
        return None


def _stderr_of(result: "subprocess.CompletedProcess[bytes]") -> str:
    return result.stderr.decode("utf-8", errors="replace").strip()


def _add_account(
    name: str, home: Path, avoid_uids: frozenset[int], avoid_gids: frozenset[int]
) -> None:
    """Create ``name`` and a group of the same name, both under an id drawn at
    random from the session range that no account, group, live process or
    ``avoid_uids`` and ``avoid_gids`` holds."""
    groupadd = _require_binary("groupadd")
    useradd = _require_binary("useradd")
    detail = ""
    for _ in range(_UID_ATTEMPTS):
        uid = SESSION_UID_MIN + secrets.randbelow(SESSION_UID_MAX - SESSION_UID_MIN + 1)
        if (
            uid in avoid_uids
            or uid in avoid_gids
            or _uid_exists(uid)
            or _gid_exists(uid)
            or _processes_of(uid)
        ):
            continue
        try:
            _run(
                [groupadd, "--gid", str(uid), name],
                f"create SSH session group {name}",
            )
        except ExecutionError as exc:
            detail = str(exc)
            if _group_exists(name):
                raise
            continue
        try:
            _run(
                [
                    useradd,
                    "--no-create-home",
                    "--no-user-group",
                    "--gid",
                    str(uid),
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
            _delete_group(name)
    raise ExecutionError(
        f"Could not find a free uid for SSH session account {name}. {detail}".strip()
    )


def _delete_group(name: str) -> bool:
    """Delete group ``name`` if it exists; return whether it is gone."""
    if not _group_exists(name):
        return True
    if (groupdel := shutil.which("groupdel")) is None:
        logger.warning("groupdel is missing; leaving group %s behind", name)
        return False
    try:
        _run([groupdel, name], f"delete SSH session group {name}")
    except ExecutionError:
        logger.warning("Failed to delete SSH session group %s", name)
        return False
    return True


def _revoke(denial: acl.Denial) -> None:
    path = Path(denial.path)
    try:
        if path.exists():
            acl.revoke(denial.uid, path, denial.mask)
        acl.forget(denial)
    except ExecutionError:
        logger.warning(
            "Failed to lift the SSH session denial of uid %d on %s; the next reap "
            "retries",
            denial.uid,
            path,
        )


def _revoke_orphaned_denies() -> None:
    try:
        records = acl.recorded()
    except ExecutionError:
        logger.warning("Cannot read recorded SSH session ACL entries", exc_info=True)
        return
    for denial in records:
        if not _uid_exists(denial.uid):
            _revoke(denial)


def _uid_exists(uid: int) -> bool:
    try:
        pwd.getpwuid(uid)
    except KeyError:
        return False
    return True


def _gid_exists(gid: int) -> bool:
    try:
        grp.getgrgid(gid)
    except KeyError:
        return False
    return True


def _group_exists(name: str) -> bool:
    try:
        grp.getgrnam(name)
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
    """Return a valid, unguessable password hash, so sshd does not read the account
    as locked."""
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


def _run(
    argv: list[str], what: str, timeout: float | None = _USERADD_TIMEOUT_SEC
) -> "subprocess.CompletedProcess[bytes]":
    try:
        result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
            argv, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(f"Failed to {what}: timed out after {timeout}s") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionError(f"Failed to {what}: {detail}")
    return result
