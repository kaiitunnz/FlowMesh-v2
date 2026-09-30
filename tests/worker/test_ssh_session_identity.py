"""A session account: its name, its uid, its password hash, and what retiring it
removes. Nothing here creates an account or signals a real process."""

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_no_uid_is_handed_out_twice(tmp_path: Path) -> None:
    with (
        patch.object(session_identity, "UID_HIGH_WATER", tmp_path / "state" / "uid"),
        patch.object(
            session_identity,
            "_uid_in_use",
            side_effect=lambda uid: uid == session_identity.FIRST_SESSION_UID + 1,
        ),
    ):
        first = session_identity.allocate_uid()
        second = session_identity.allocate_uid()

    assert first == session_identity.FIRST_SESSION_UID
    assert second == first + 2
    assert (tmp_path / "state" / "uid").read_text() == str(second)
    assert (tmp_path / "state" / "uid").stat().st_mode & 0o777 == 0o600


def test_retiring_removes_only_the_files_the_account_left(tmp_path: Path) -> None:
    shared = tmp_path / "tmp"
    (shared / "left-dir" / "inner").mkdir(parents=True)
    (shared / "left-file").write_text("secret")
    (shared / "left-link").symlink_to(tmp_path)
    kept = tmp_path / "kept"
    kept.write_text("kept")
    uid = os.getuid()

    with patch.object(session_identity, "SHARED_TMP_DIRS", (shared,)):
        session_identity.remove_files_of(uid + 1)
        assert sorted(p.name for p in shared.iterdir()) == [
            "left-dir",
            "left-file",
            "left-link",
        ]
        session_identity.remove_files_of(uid)

    assert list(shared.iterdir()) == []
    assert kept.read_text() == "kept"


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
        patch.object(session_identity.time, "sleep"),
    ):
        assert session_identity.kill_processes(200_000)

    assert kill_all.call_count == 2


def test_an_account_with_a_process_no_kill_ends_is_locked_not_deleted() -> None:
    with (
        patch.object(session_identity, "kill_processes", return_value=False),
        patch.object(session_identity, "lock_account") as lock,
        patch.object(session_identity, "lift_denials") as lift,
        patch.object(session_identity, "delete_account") as delete,
        patch.object(session_identity, "remove_files_of") as remove,
    ):
        assert not session_identity.retire_account("fmssn1", 200_000, [Path("/r")])

    lock.assert_called_once_with("fmssn1")
    lift.assert_not_called()
    delete.assert_not_called()
    remove.assert_not_called()


def test_a_released_account_that_cannot_be_retired_fails_loudly() -> None:
    account = session_identity.SessionAccount("fmssn1", 200_000, 200_000, Path("/h"))
    with (
        patch.object(session_identity, "retire_account", return_value=False),
        pytest.raises(ExecutionError, match="stays locked"),
    ):
        account.release()


@pytest.mark.parametrize(("worker_uid", "uid"), [(1000, 200_000), (0, 0)])
def test_the_kill_runs_only_as_root_and_never_as_root(
    worker_uid: int, uid: int
) -> None:
    with (
        patch.object(session_identity.os, "getuid", return_value=worker_uid),
        patch.object(session_identity.subprocess, "run") as run,
    ):
        session_identity._kill_all_as(uid)

    run.assert_not_called()
