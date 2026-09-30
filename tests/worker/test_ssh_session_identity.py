"""A session account: its name, its uid, its password hash, and what retiring it
removes. Nothing here creates an account or signals a real process."""

import ctypes
import errno
import os
import subprocess
import threading
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


def test_an_id_an_account_a_group_a_process_or_a_state_root_holds_is_not_drawn() -> (
    None
):
    draws = iter(range(8))
    created: list[list[str]] = []

    def run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
        created.append(argv)
        return _run_ok(argv, what)

    first = session_identity.SESSION_UID_MIN
    with (
        patch.object(session_identity.shutil, "which", side_effect=lambda b: f"/x/{b}"),
        patch.object(
            session_identity.secrets, "randbelow", side_effect=lambda n: next(draws)
        ),
        patch.object(
            session_identity, "_uid_exists", side_effect=lambda uid: uid == first + 1
        ),
        patch.object(
            session_identity, "_gid_exists", side_effect=lambda gid: gid == first + 4
        ),
        patch.object(
            session_identity,
            "_processes_of",
            side_effect=lambda uid: [MagicMock()] if uid == first + 2 else [],
        ),
        patch.object(session_identity, "_run", side_effect=run),
    ):
        session_identity._add_account(
            "fmssn1", Path("/h"), frozenset({first, first + 3})
        )

    drawn = str(first + 5)
    assert created[0] == ["/x/groupadd", "--gid", drawn, "fmssn1"]
    useradd = created[1]
    assert useradd[0] == "/x/useradd" and useradd[-1] == "fmssn1"
    assert useradd[useradd.index("--uid") + 1] == drawn
    assert useradd[useradd.index("--gid") + 1] == drawn


def test_a_group_created_for_an_account_that_fails_is_removed() -> None:
    commands: list[str] = []
    groups: set[str] = set()

    def run(argv: list[str], what: str) -> "subprocess.CompletedProcess[bytes]":
        commands.append(Path(argv[0]).name)
        if argv[0].endswith("groupadd"):
            groups.add(argv[-1])
        elif argv[0].endswith("groupdel"):
            groups.discard(argv[-1])
        else:
            raise ExecutionError("useradd failed")
        return _run_ok(argv, what)

    with (
        patch.object(session_identity.shutil, "which", side_effect=lambda b: f"/x/{b}"),
        patch.object(session_identity, "_UID_ATTEMPTS", 1),
        patch.object(session_identity, "_uid_exists", return_value=False),
        patch.object(session_identity, "_gid_exists", return_value=False),
        patch.object(session_identity, "_processes_of", return_value=[]),
        patch.object(session_identity, "_account_exists", return_value=False),
        patch.object(
            session_identity, "_group_exists", side_effect=lambda n: n in groups
        ),
        patch.object(session_identity, "_run", side_effect=run),
        pytest.raises(ExecutionError, match="useradd failed"),
    ):
        session_identity._add_account("fmssn1", Path("/h"), frozenset())

    assert commands == ["groupadd", "useradd", "groupdel"]
    assert not groups


def test_a_new_account_avoids_every_uid_a_state_root_names(tmp_path: Path) -> None:
    avoided: list[frozenset[int]] = []
    roots = [tmp_path / "results", tmp_path / "hb"]
    named = {roots[0]: {61001}, roots[1]: {61002}}

    def add(name: str, home: Path, avoid: frozenset[int]) -> None:
        avoided.append(avoid)
        raise ExecutionError("stop here")

    with (
        patch.object(session_identity, "_ensure_privsep_dir"),
        patch.object(session_identity.acl, "named_uids", side_effect=named.get),
        patch.object(session_identity, "_add_account", side_effect=add),
        pytest.raises(ExecutionError, match="stop here"),
    ):
        session_identity.SessionAccount.create("fmssn1", tmp_path / "home", roots)

    assert avoided == [frozenset({61001, 61002})]


def _create_concurrently(tmp_path: Path) -> list[int]:
    """Create two accounts at once, as two workers in separate containers sharing
    the roots would: each sees only the other's ACL entries, never its account."""
    roots = [tmp_path / "results", tmp_path / "content"]
    for root in roots:
        root.mkdir()
    entries: dict[Path, set[int]] = {root: set() for root in roots}
    accounts: dict[str, int] = {}
    barrier = threading.Barrier(2)

    def named(root: Path) -> set[int]:
        found = set(entries[root])
        time.sleep(0.1)
        return found

    def add(name: str, home: Path, uids: frozenset[int]) -> None:
        accounts[name] = min(set(range(61_000, 61_010)) - uids)

    def deny(uid: int, root: Path) -> None:
        entries[root].add(uid)

    def create(name: str) -> None:
        barrier.wait()
        session_identity.SessionAccount.create(name, tmp_path / name, roots)

    with (
        patch.object(session_identity, "_ensure_privsep_dir"),
        patch.object(session_identity, "_unlock"),
        patch.object(session_identity.os, "lchown"),
        patch.object(session_identity.acl, "named_uids", side_effect=named),
        patch.object(session_identity.acl, "record"),
        patch.object(session_identity.acl, "deny", side_effect=deny),
        patch.object(session_identity, "_add_account", side_effect=add),
        patch.object(
            session_identity.pwd,
            "getpwnam",
            side_effect=lambda name: MagicMock(
                pw_uid=accounts[name], pw_gid=accounts[name]
            ),
        ),
    ):
        threads = [
            threading.Thread(target=create, args=(name,))
            for name in ("fmssna", "fmssnb")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return sorted(accounts.values())


def test_two_workers_sharing_a_root_never_draw_the_same_uid(tmp_path: Path) -> None:
    assert _create_concurrently(tmp_path) == [61_000, 61_001]


def test_a_root_that_refuses_a_lock_is_drawn_on_without_one(tmp_path: Path) -> None:
    with patch.object(
        session_identity.fcntl, "flock", side_effect=OSError(errno.ENOLCK, "no locks")
    ):
        assert len(_create_concurrently(tmp_path)) == 2


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
    monkeypatch.setattr(session_identity, "scratch_dirs", lambda: [shared])

    session_identity.purge_uid_files(uid + 1)
    assert sorted(p.name for p in shared.iterdir()) == [
        "left-dir",
        "left-file",
        "left-link",
    ]
    session_identity.purge_uid_files(uid)

    assert list(shared.iterdir()) == []
    assert kept.read_text() == "kept"


def test_the_scratch_dirs_include_tmp_when_the_temp_dir_is_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_identity.tempfile, "tempdir", tmp_path.as_posix())
    dirs = session_identity.scratch_dirs()

    assert dirs[0] == tmp_path
    assert {Path("/tmp"), Path("/run/lock"), Path("/dev/mqueue")} <= set(dirs)
    assert len(dirs) == len(set(dirs))
    monkeypatch.setattr(session_identity.tempfile, "tempdir", "/tmp")
    assert session_identity.scratch_dirs().count(Path("/tmp")) == 1


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
    killed: bool,
    deleted: bool,
    recorded: set[Any],
    recorded_error: Exception | None = None,
    **patches: Any,
) -> tuple[bool, MagicMock, MagicMock, MagicMock]:
    steps = MagicMock()
    steps.kill_processes.return_value = killed
    steps.delete_account.return_value = deleted
    steps.recorded.return_value = recorded
    steps.recorded.side_effect = recorded_error
    steps.purge_uid.return_value = None
    steps.lock_account.return_value = None
    steps._revoke.return_value = None
    defaults: dict[str, Any] = {
        "_account_exists": MagicMock(return_value=True),
        "_uid_exists": MagicMock(return_value=True),
        **{
            name: getattr(steps, name)
            for name in (
                "kill_processes",
                "delete_account",
                "lock_account",
                "purge_uid",
                "_revoke",
            )
        },
        **patches,
    }
    with (
        patch.multiple(session_identity, **defaults),
        patch.object(session_identity.acl, "recorded", steps.recorded),
    ):
        retired = session_identity.retire_account("fmssn1", 61_001)
    _retire.steps = steps  # type: ignore[attr-defined]
    return retired, steps.lock_account, steps.purge_uid, steps._revoke


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
        True,
        True,
        {(61_001, "/r"), (61_001, "/hb"), (61_002, "/r")},
    )

    assert retired
    lock.assert_not_called()
    purge.assert_called_once_with(61_001)
    assert sorted(call.args for call in revoke.call_args_list) == [
        (61_001, Path("/hb")),
        (61_001, Path("/r")),
    ]


def test_an_account_is_killed_then_deleted_then_purged_then_revoked() -> None:
    _retire(True, True, {(61_001, "/r")})

    assert [name for name, _, _ in _retire.steps.mock_calls] == [  # type: ignore[attr-defined]
        "kill_processes",
        "delete_account",
        "purge_uid",
        "recorded",
        "_revoke",
    ]


def test_an_account_already_deleted_finishes_its_purge_and_revoke() -> None:
    retired, lock, purge, revoke = _retire(
        True,
        True,
        {(61_001, "/r")},
        _account_exists=MagicMock(return_value=False),
        _uid_exists=MagicMock(return_value=False),
    )

    assert retired
    lock.assert_not_called()
    purge.assert_called_once_with(61_001)
    revoke.assert_called_once_with(61_001, Path("/r"))


def test_the_denials_of_a_uid_another_account_now_holds_are_left_alone() -> None:
    retired, lock, purge, revoke = _retire(
        True,
        True,
        {(61_001, "/r")},
        _account_exists=MagicMock(return_value=False),
    )

    assert retired
    assert not _retire.steps.kill_processes.called  # type: ignore[attr-defined]
    purge.assert_not_called()
    revoke.assert_not_called()


def test_an_unreadable_ledger_does_not_fail_a_retirement() -> None:
    retired, _, purge, revoke = _retire(
        True, True, set(), recorded_error=ExecutionError("unreadable")
    )

    assert retired
    purge.assert_called_once_with(61_001)
    revoke.assert_not_called()


def test_deleting_an_account_that_is_already_gone_succeeds() -> None:
    with (
        patch.object(session_identity.shutil, "which", return_value="/x/userdel"),
        patch.object(
            session_identity, "_run", side_effect=ExecutionError("does not exist")
        ),
        patch.object(session_identity, "_account_exists", return_value=False),
        patch.object(session_identity, "_group_exists", return_value=False),
    ):
        assert session_identity.delete_account("fmssn1")


def test_a_command_that_times_out_fails_as_an_execution_error() -> None:
    with (
        patch.object(
            session_identity.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["rm"], 30),
        ),
        pytest.raises(ExecutionError, match="timed out"),
    ):
        session_identity._run(["/bin/rm", "x"], "remove x")


def test_a_tree_removal_that_times_out_is_only_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stuck = tmp_path / "stuck"
    stuck.mkdir()
    with patch.object(
        session_identity.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(["rm"], 300),
    ):
        session_identity.remove_tree(stuck)

    assert "Failed to remove" in caplog.text


def test_a_failed_file_purge_still_purges_the_ipc_objects() -> None:
    with (
        patch.object(session_identity, "purge_uid_files", side_effect=OSError("stuck")),
        patch.object(session_identity, "purge_uid_ipc") as ipc,
    ):
        session_identity.purge_uid(61_001)

    ipc.assert_called_once_with(61_001)


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
    revoked: list[Any] = []
    with (
        patch.object(session_identity.pwd, "getpwall", return_value=[]),
        patch.object(session_identity.grp, "getgrall", return_value=[]),
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


def test_a_mask_is_redundant_only_when_it_narrows_nothing() -> None:
    acl = session_identity.acl
    assert acl.mask_is_redundant("user::rwx\ngroup::rwx\nmask::rwx\nother::r-x\n")
    assert acl.mask_is_redundant(
        "user::rwx\ngroup::r-x\nmask::rwx\nother::---\ndefault:user:1000:rwx\n"
    )
    assert not acl.mask_is_redundant(
        "user::rwx\ngroup::rwx\t#effective:r-x\nmask::r-x\nother::r-x\n"
    )
    assert not acl.mask_is_redundant(
        "user::rwx\nuser:1000:r-x\ngroup::r-x\nmask::r-x\nother::r-x\n"
    )
    assert not acl.mask_is_redundant("user::rwx\ngroup::r-x\nother::r-x\n")


def test_a_denial_and_its_revoke_never_recalculate_the_mask(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        calls.append(argv[1:-2])
        return subprocess.CompletedProcess(argv, 0, b"group::rwx\nmask::rwx\n", b"")

    with (
        patch.object(session_identity.acl.shutil, "which", side_effect=lambda b: b),
        patch.object(session_identity.acl.subprocess, "run", side_effect=run),
    ):
        session_identity.acl.deny(61_001, tmp_path)
        session_identity.acl.revoke(61_001, tmp_path)

    assert calls == [
        ["-n", "-m", "u:61001:---"],
        ["-n", "-x", "u:61001"],
        ["-n", "-c", "-p"],
        ["-x", "m::"],
    ]


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


def _helper(uid: int) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(  # nosec B603 - argv list, test-only
        session_identity.interpreter_argv(
            session_identity._AS_UID_PREAMBLE + "print('ran')", str(uid), "0"
        ),
        capture_output=True,
        check=False,
    )


def test_a_helper_refuses_to_run_as_root() -> None:
    refused = _helper(0)

    assert refused.returncode != 0
    assert b"ran" not in refused.stdout


def test_a_helper_runs_once_its_every_uid_is_the_session_s() -> None:
    if os.getuid() == 0:
        pytest.skip("runs the helper without switching uid")
    assert _helper(os.getuid()).stdout == b"ran\n"


def test_helpers_import_nothing_after_switching_uid() -> None:
    preamble = session_identity._AS_UID_PREAMBLE
    assert preamble.rindex("import") < preamble.index("os.setuid")
    for script in (
        session_identity._KILL_ALL_SCRIPT,
        session_identity._REMOVE_IPC_SCRIPT,
        session_identity._EXEC_SCRIPT,
    ):
        assert "import" not in script
