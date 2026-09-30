"""A session account: its name, its uid, its password hash, and what retiring it
removes. Nothing here creates an account or signals a real process."""

import ctypes
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import psutil
import pytest

from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import session_identity


def test_the_password_openssl_hashes_never_reads_as_an_option() -> None:
    argv_seen: list[list[str]] = []

    def run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
        argv_seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"$6$salt$hash\n")

    with (
        patch.object(session_identity.shutil, "which", return_value="/usr/bin/openssl"),
        patch.object(session_identity, "_run", side_effect=run),
        patch.object(session_identity.secrets, "token_urlsafe", return_value="-Xa9"),
    ):
        assert session_identity._unusable_password_hash() == "$6$salt$hash"

    ((*_, password),) = argv_seen
    assert not password.startswith("-")


def test_an_account_name_is_derived_from_its_session() -> None:
    name: Any = session_identity.account_name_for(
        "ssn-0123456789abcdef0123456789abcdef"
    )

    assert name == "fmssn0123456789abcdef"
    assert session_identity.ACCOUNT_NAME_RE.match(name)


def _run_ok(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_a_uid_an_account_a_process_or_a_state_root_holds_is_not_drawn() -> None:
    draws = iter([0, 1, 2, 3])
    created: list[list[str]] = []

    def run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
        created.append(argv)
        return _run_ok(argv, what)

    first = session_identity.SESSION_UID_MIN
    with (
        patch.object(
            session_identity.secrets, "randbelow", side_effect=lambda n: next(draws)
        ),
        patch.object(
            session_identity, "_uid_exists", side_effect=lambda uid: uid == first + 1
        ),
        patch.object(
            session_identity,
            "_processes_of",
            side_effect=lambda uid: [MagicMock()] if uid == first + 2 else [],
        ),
        patch.object(session_identity, "_run", side_effect=run),
    ):
        session_identity._add_account(
            "/usr/sbin/useradd", "fmssn1", Path("/h"), frozenset({first})
        )

    ((*_, uid_flag, uid, _home_flag, _home, _shell_flag, _shell, name),) = created
    assert (uid_flag, uid, name) == ("--uid", str(first + 3), "fmssn1")


def test_a_new_account_avoids_every_uid_a_state_root_names(tmp_path: Path) -> None:
    avoided: list[frozenset[int]] = []
    roots = [tmp_path / "results", tmp_path / "hb"]
    named = {roots[0]: {61001}, roots[1]: {61002}}

    def add(useradd: str, name: str, home: Path, avoid: frozenset[int]) -> None:
        avoided.append(avoid)
        raise ExecutionError("stop here")

    with (
        patch.object(session_identity, "_ensure_privsep_dir"),
        patch.object(session_identity.shutil, "which", return_value="/bin/x"),
        patch.object(session_identity.acl, "named_uids", side_effect=named.get),
        patch.object(session_identity, "_add_account", side_effect=add),
        pytest.raises(ExecutionError, match="stop here"),
    ):
        session_identity.SessionAccount.create("fmssn1", tmp_path / "home", roots)

    assert avoided == [frozenset({61001, 61002})]


def test_the_uid_range_fits_a_user_namespace() -> None:
    assert (
        1000
        < session_identity.SESSION_UID_MIN
        <= session_identity.SESSION_UID_MAX
        < 65536
    )


def test_retiring_removes_only_the_files_the_account_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "tmp"
    deep = shared.joinpath("left-dir", *(["d"] * 20))
    deep.mkdir(parents=True)
    (shared / "left-file").write_text("secret")
    (shared / "left-link").symlink_to(tmp_path)
    kept = tmp_path / "kept"
    kept.write_text("kept")
    uid = os.getuid()
    monkeypatch.setattr(session_identity.tempfile, "tempdir", shared.as_posix())
    monkeypatch.setattr(session_identity, "_WORLD_WRITABLE_DIRS", ())

    session_identity.purge_uid_files(uid + 1)
    assert sorted(p.name for p in shared.iterdir()) == [
        "left-dir",
        "left-file",
        "left-link",
    ]
    session_identity.purge_uid_files(uid)

    assert list(shared.iterdir()) == []
    assert kept.read_text() == "kept"


def test_a_tree_deeper_than_python_recurses_is_removed(tmp_path: Path) -> None:
    root = tmp_path / "deep"
    root.mkdir()
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(1500):
            os.mkdir("d", dir_fd=fd)
            child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
    finally:
        os.close(fd)

    session_identity.remove_tree(root)

    assert not root.exists()


def _process() -> MagicMock:
    return MagicMock(spec=["send_signal"])


def test_the_kill_repeats_until_no_process_of_the_account_is_left() -> None:
    survivor = _process()
    rounds = iter([[survivor], [survivor], [survivor], []])
    with (
        patch.object(
            session_identity, "_processes_of", side_effect=lambda uid: next(rounds)
        ),
        patch.object(session_identity.psutil, "wait_procs"),
        patch.object(session_identity, "_kill_all_as") as kill_all,
    ):
        assert session_identity.kill_processes(61_000)

    assert kill_all.call_count == 3
    assert survivor.send_signal.call_count == 3


def test_the_kill_signals_the_account_at_once_even_when_no_process_is_seen() -> None:
    with (
        patch.object(session_identity, "_processes_of", return_value=[]),
        patch.object(session_identity, "_kill_all_as") as kill_all,
    ):
        assert session_identity.kill_processes(61_000)

    kill_all.assert_called_once_with(61_000)


def test_root_processes_are_never_the_account_s_to_kill() -> None:
    with patch.object(session_identity, "_processes_of") as processes:
        assert not session_identity.kill_processes(0)

    processes.assert_not_called()


def test_a_zombie_of_the_account_is_not_a_process_left_to_kill() -> None:
    child = subprocess.Popen(["true"])  # nosec B603 B607 - argv list, test-only
    try:
        deadline = time.monotonic() + 10
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        pids = {p.pid for p in session_identity._processes_of(os.getuid())}
    finally:
        child.wait()

    assert child.pid not in pids
    assert os.getpid() in pids


def _retire(
    killed: bool, deleted: bool, recorded: set[tuple[int, str]]
) -> tuple[bool, MagicMock, MagicMock, MagicMock]:
    with (
        patch.object(session_identity, "kill_processes", return_value=killed),
        patch.object(session_identity, "delete_account", return_value=deleted),
        patch.object(session_identity, "lock_account") as lock,
        patch.object(session_identity, "purge_uid") as purge,
        patch.object(session_identity.acl, "recorded", return_value=recorded),
        patch.object(session_identity, "_revoke") as revoke,
    ):
        retired = session_identity.retire_account("fmssn1", 61_001)
    return retired, lock, purge, revoke


def test_an_account_with_a_process_no_kill_ends_is_locked_and_kept_denied() -> None:
    retired, lock, purge, revoke = _retire(False, True, {(61_001, "/r")})

    assert not retired
    lock.assert_called_once_with("fmssn1")
    purge.assert_not_called()
    revoke.assert_not_called()


def test_an_account_that_cannot_be_deleted_is_locked_and_kept_denied() -> None:
    retired, lock, purge, revoke = _retire(True, False, {(61_001, "/r")})

    assert not retired
    lock.assert_called_once_with("fmssn1")
    purge.assert_not_called()
    revoke.assert_not_called()


def test_a_deleted_account_loses_only_its_own_recorded_denials() -> None:
    retired, lock, purge, revoke = _retire(
        True, True, {(61_001, "/r"), (61_001, "/hb"), (61_002, "/r")}
    )

    assert retired
    lock.assert_not_called()
    purge.assert_called_once_with(61_001)
    assert sorted(call.args for call in revoke.call_args_list) == [
        (61_001, Path("/hb")),
        (61_001, Path("/r")),
    ]


def test_a_released_account_that_cannot_be_retired_fails_loudly() -> None:
    account = session_identity.SessionAccount("fmssn1", 61_001, 61_001, Path("/h"))
    with (
        patch.object(session_identity, "retire_account", return_value=False),
        pytest.raises(ExecutionError, match="stays locked"),
    ):
        account.release()


def test_a_partial_denial_is_rolled_back(tmp_path: Path) -> None:
    recorded: list[Path] = []
    revoked: list[Path] = []

    def deny(uid: int, path: Path) -> None:
        if path.name == "second":
            raise ExecutionError("no ACL support")

    account = session_identity.SessionAccount("fmssn1", 61_001, 61_001, tmp_path)
    with (
        patch.object(
            session_identity.acl, "record", side_effect=lambda u, p: recorded.append(p)
        ),
        patch.object(session_identity.acl, "deny", side_effect=deny),
        patch.object(
            session_identity, "_revoke", side_effect=lambda u, p: revoked.append(p)
        ),
        pytest.raises(ExecutionError, match="Could not isolate"),
    ):
        account.deny([tmp_path / "first", tmp_path / "second", tmp_path / "third"])

    # Recorded before written, so the one that failed midway is revoked too.
    assert recorded == [tmp_path / "first", tmp_path / "second"]
    assert revoked == recorded


def test_the_reap_lifts_the_denials_of_accounts_that_no_longer_exist() -> None:
    revoked: list[tuple[int, Path]] = []
    with (
        patch.object(session_identity.pwd, "getpwall", return_value=[]),
        patch.object(
            session_identity.acl,
            "recorded",
            return_value={(61_001, "/gone"), (61_002, "/live")},
        ),
        patch.object(
            session_identity, "_uid_exists", side_effect=lambda uid: uid == 61_002
        ),
        patch.object(
            session_identity,
            "_revoke",
            side_effect=lambda uid, path: revoked.append((uid, path)),
        ),
    ):
        assert session_identity.reap_stale_accounts()

    assert revoked == [(61_001, Path("/gone"))]


def test_a_revoke_drops_the_mask_once_no_named_entry_needs_it(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with (
        patch.object(session_identity.acl.shutil, "which", return_value="/x/setfacl"),
        patch.object(session_identity.acl.subprocess, "run", side_effect=run),
    ):
        session_identity.acl.revoke(61_001, tmp_path)

    assert [argv[1:3] for argv in calls] == [["-x", "u:61001"], ["-x", "m::"]]


def test_getfacl_output_parses_to_uids() -> None:
    output = (
        "user::rwx\nuser:61001:---\nuser:1000:r-x\ngroup::r-x\nmask::r-x\n"
        "other::r-x\ndefault:user:61002:---\n"
    )
    assert session_identity.acl.parse_denied_uids(output) == {61001}
    assert session_identity.acl.parse_named_uids(output) == {61001, 1000, 61002}


def test_every_process_of_the_uid_is_signalled_from_a_helper_that_drops_to_it() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with (
        patch.object(session_identity.os, "geteuid", return_value=0),
        patch.object(session_identity.os, "getuid", return_value=0),
        patch.object(session_identity.subprocess, "run", side_effect=run),
    ):
        session_identity._kill_all_as(61_001)

    [(argv, kwargs)] = calls
    script, uid, gid = argv[-3:]
    assert "os.setuid(uid)" in script
    assert "os.kill(-1, signal.SIGKILL)" in script
    assert (uid, gid) == ("61001", "65534")
    assert not {"user", "group", "extra_groups", "preexec_fn"} & set(kwargs)


@pytest.mark.parametrize(
    ("euid", "uid", "target"), [(1000, 1000, 61_001), (0, 0, 0), (0, 61_001, 61_001)]
)
def test_the_kill_runs_only_as_root_and_never_as_itself(
    euid: int, uid: int, target: int
) -> None:
    with (
        patch.object(session_identity.os, "geteuid", return_value=euid),
        patch.object(session_identity.os, "getuid", return_value=uid),
        patch.object(session_identity.subprocess, "run") as run,
    ):
        session_identity._kill_all_as(target)

    run.assert_not_called()


def test_a_helper_drops_to_the_account_itself_and_never_runs_as_root() -> None:
    argv = session_identity.exec_as(61_001, 61_001, ["/usr/bin/tar", "-c"])

    assert argv[-4:] == ["61001", "61001", "/usr/bin/tar", "-c"]
    assert "os.execv(sys.argv[3], sys.argv[3:])" in argv[argv.index("-c") + 1]
    with pytest.raises(ExecutionError):
        session_identity.exec_as(0, 0, ["/usr/bin/tar"])


_SHM_TABLE = (
    "   key  shmid perms  size cpid lpid nattch   uid gid  cuid cgid\n"
    "     0     11   600  4096  100  100      0 61001 100 61001  100\n"
    "     0     12   600  4096  100  100      0  1000 100 61001  100\n"
    "     0     13   666  4096  100  100      0  1000 100  1000  100\n"
)


def test_the_ipc_objects_an_account_owns_or_created_are_selected() -> None:
    assert session_identity.owned_ipc_ids(_SHM_TABLE, "shmid", 61001) == [11, 12]
    assert session_identity.owned_ipc_ids("", "shmid", 61001) == []


def test_a_segment_the_uid_left_is_removed() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    ipc_private, ipc_creat = 0, 0o1000
    shmid = libc.shmget(ipc_private, 4096, ipc_creat | 0o600)
    if shmid < 0:
        pytest.skip("System V shared memory is unavailable")
    try:
        real = Path("/proc/sysvipc/shm").read_text(encoding="utf-8").splitlines()
        ours = [real[0]] + [line for line in real[1:] if line.split()[1] == str(shmid)]
        assert len(ours) == 2
        # Only this test's segment is visible, so no other object of this uid is
        # touched.
        with patch.object(
            session_identity,
            "_read_ipc_table",
            side_effect=lambda kind: "\n".join(ours) if kind == "shm" else "",
        ):
            session_identity.purge_uid_ipc(os.getuid())
        listed = Path("/proc/sysvipc/shm").read_text(encoding="utf-8")
        assert str(shmid) not in [line.split()[1] for line in listed.splitlines()[1:]]
    finally:
        libc.shmctl(shmid, 0, None)
