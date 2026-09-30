"""The process session backend against a real sshd and real accounts.

It creates accounts and starts sshd, so it runs only as root inside a container,
never on a host.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import psutil
import pytest

from shared.utils import new_ssh_session_id
from tests.worker.factories import make_worker_config
from worker.config import WorkerConfig
from worker.executors.base_executor import ExecutionError, RunSignals
from worker.executors.ssh_session import (
    ProcessSessionBackend,
    ResolvedSSHInput,
    SessionRequest,
)
from worker.executors.ssh_session.backends import process as process_module
from worker.executors.ssh_session.config import SSHOutputConfig

pytestmark = pytest.mark.skipif(
    os.getuid() != 0 or not Path("/.dockerenv").exists(),
    reason="creates accounts and starts sshd: root inside a container only",
)

_WORKER_SECRET = "worker-secret-sentinel"
_SESSION_TOKEN = "session-token-value"


def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - argv list, test-only helper
        argv, capture_output=True, text=True, timeout=60, check=False, **kwargs
    )


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[WorkerConfig]:
    monkeypatch.setenv("WORKER_SECRET", _WORKER_SECRET)
    # Traversable by any uid, as a worker's results volume is.
    base = Path(tempfile.mkdtemp(dir="/var/lib"))
    base.chmod(0o755)
    roots = {name: base / name for name in ("results", "private", "content", "hf")}
    for path in roots.values():
        path.mkdir(mode=0o777)
    roots["private"].chmod(0o700)
    other = roots["results"] / "tsk-other"
    other.mkdir(mode=0o777)
    (other / "results.json").write_text('{"other": "tenant"}')
    (other / "results.json").chmod(0o666)
    (roots["content"] / "object").write_text("cached content")
    (roots["content"] / "object").chmod(0o666)
    (roots["private"] / "state").write_text("private state")
    hb_file = base / "worker.hb"
    hb_file.write_text("alive")
    hb_file.chmod(0o666)
    config = make_worker_config(
        results_dir=roots["results"],
        private_state_dir=roots["private"],
        content_dir=roots["content"],
        model_cache_dir=roots["hf"],
        hb_file=hb_file,
        home_dir=Path("/root"),
        ssh_relay_host="127.0.0.1",
    )
    yield config
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(Path(str(process_module.SAFE_MOUNT_ROOT)), ignore_errors=True)


@pytest.fixture
def client_key(tmp_path: Path) -> Path:
    key = tmp_path / "client_key"
    _run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key.as_posix()])
    return key


def _request(
    tmp_path: Path,
    client_key: Path,
    inputs: list[ResolvedSSHInput] | None = None,
    output: str | None = None,
) -> SessionRequest:
    cfg = MagicMock(
        interactive=True,
        user="flowmesh",
        authorized_keys=[client_key.with_suffix(".pub").read_text().strip()],
        extra_env={"TOKEN": _SESSION_TOKEN},
        output=None if output is None else SSHOutputConfig(output, None),
        gpu_device_ids=[],
        image=None,
    )
    return SessionRequest(
        task_id="tsk-ssh",
        session_id=new_ssh_session_id(),
        worker_name="worker-1",
        cfg=cfg,
        out_dir=tmp_path / "out",
        resolved_inputs=inputs or [],
        signals=RunSignals(),
    )


def _ssh(
    session: process_module.ProcessSession, key: Path, port: int, command: str
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-i",
            key.as_posix(),
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            f"{session.login_user()}@127.0.0.1",
            command,
        ]
    )


def _read(path: Path) -> str:
    """A shell probe printing the head of a file, or nothing when it cannot."""
    return f"cat {path.as_posix()} 2>/dev/null | head -c 64; true"


def _list(path: Path) -> str:
    """A shell probe listing a directory, or printing nothing when it cannot."""
    return f"ls -A {path.as_posix()} 2>/dev/null; true"


def _acl_users(path: Path) -> list[str]:
    out = _run(["getfacl", "-cp", path.as_posix()]).stdout
    return [
        line
        for line in out.splitlines()
        if line.startswith("user:") and "::" not in line
    ]


def test_a_session_logs_in_as_its_own_account_and_reaches_only_its_own_data(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    upstream = worker.results_dir / "tsk-up"
    upstream.mkdir()
    (upstream / "results.json").write_text('{"up": "stream"}')
    inputs = [
        ResolvedSSHInput(
            stage="up",
            task_id="tsk-up",
            source_path=upstream,
            mount_path="/mnt/flowmesh/inputs/up",
        )
    ]
    backend = ProcessSessionBackend(worker)
    backend.prepare()
    request = _request(tmp_path, client_key, inputs, "/mnt/flowmesh/output")
    session = backend.start_session(request)
    try:
        port = session.wait_ready(30)
        assert port is not None
        account = session.account

        uid = _ssh(session, client_key, port, "id -u")
        assert uid.returncode == 0, uid.stderr
        assert int(uid.stdout.strip()) == account.uid != 0

        # The file carrying the session's env is the session's to read alone.
        keys_file = (
            process_module.SESSIONS_ROOT / request.session_id / "authorized_keys"
        )
        as_nobody = ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups"]
        assert _run([*as_nobody, "cat", keys_file.as_posix()]).returncode != 0
        readable = _ssh(session, client_key, port, f"test -r {keys_file.as_posix()}")
        assert readable.returncode == 0

        env = _ssh(session, client_key, port, "env").stdout
        assert f"TOKEN={_SESSION_TOKEN}" in env
        assert _WORKER_SECRET not in env

        # The probes see a match on what the session is meant to reach.
        mounted = Path("/mnt/flowmesh/inputs/up")
        assert (
            '"up"'
            in _ssh(session, client_key, port, _read(mounted / "results.json")).stdout
        )
        assert "results.json" in _ssh(session, client_key, port, _list(mounted)).stdout
        for denied in (
            worker.results_dir / "tsk-other" / "results.json",
            worker.content_dir / "object",
            worker.private_state_dir / "state",
            worker.hb_file,
            Path("/proc/1/environ"),
        ):
            leaked = _ssh(session, client_key, port, _read(denied))
            assert leaked.stdout == "", (denied, leaked.stdout)
        for root in worker.state_roots:
            if root.is_dir():
                listed = _ssh(session, client_key, port, _list(root))
                assert listed.stdout == "", (root, listed.stdout)
        assert (
            _WORKER_SECRET
            not in _ssh(
                session,
                client_key,
                port,
                f"cat /proc/{os.getpid()}/environ 2>/dev/null; true",
            ).stdout
        )

        wrote = _ssh(
            session,
            client_key,
            port,
            "echo done > /mnt/flowmesh/output/out.txt && flowmesh-finish",
        )
        assert wrote.returncode == 0, wrote.stderr
        assert session.finish_requested()
        session.collect_output(tmp_path / "collected", None)
        assert (tmp_path / "collected" / "out.txt").read_text() == "done\n"
    finally:
        session.stop(1)
        session.cleanup()

    assert _run(["getent", "passwd", account.name]).returncode != 0
    for root in worker.state_roots:
        assert _acl_users(root) == [], root
    assert session.poll() is not None
    assert list(Path(str(process_module.SAFE_MOUNT_ROOT)).iterdir()) == []


def test_collection_reads_the_output_with_the_session_access_alone(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    root_only = Path(tempfile.mkdtemp(dir="/var/lib")) / "root-only"
    root_only.write_text("root only")
    root_only.chmod(0o600)
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(
        _request(tmp_path, client_key, output="/mnt/flowmesh/output")
    )
    account = session.account
    try:
        output = Path("/mnt/flowmesh/output")
        (output / "kept.txt").write_text("kept")
        os.mkfifo(output / "pipe")
        # A session links a file it cannot read where hard links are unprotected.
        os.link(root_only, output / "linked")
        for name in ("kept.txt", "pipe"):
            os.lchown(output / name, account.uid, account.gid)

        started = time.monotonic()
        session.collect_output(tmp_path / "collected", None)

        assert time.monotonic() - started < 30
        collected = tmp_path / "collected"
        assert sorted(p.name for p in collected.iterdir()) == ["kept.txt"]
        assert (collected / "kept.txt").read_text() == "kept"
    finally:
        session.stop(1)
        session.cleanup()
        shutil.rmtree(root_only.parent, ignore_errors=True)
    assert _run(["getent", "passwd", account.name]).returncode != 0


_FORK_CHAIN = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    if os.fork():
        os._exit(0)
    time.sleep(0.005)
"""


def _processes_of(uid: int) -> list[psutil.Process]:
    return [p for p in psutil.process_iter(["uids"]) if p.info["uids"].real == uid]


def test_a_session_leaves_the_next_one_nothing(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    first = backend.start_session(_request(tmp_path, client_key))
    account = first.account
    leftover = Path(tempfile.gettempdir()) / f"left-by-{account.name}"
    as_account: dict[str, Any] = {
        "user": account.uid,
        "group": account.gid,
        "extra_groups": [],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    subprocess.run(  # nosec B603 - argv list, test-only
        ["sh", "-c", f"echo secret > {leftover} && chmod 600 {leftover}"],
        check=True,
        **as_account,
    )
    # Ignores a terminate and moves to a new pid every few milliseconds, so a kill
    # of the pids it listed a moment ago misses it.
    subprocess.Popen(  # nosec B603 - argv list, test-only
        [sys.executable, "-c", _FORK_CHAIN], start_new_session=True, **as_account
    )
    time.sleep(0.5)

    first.stop(1)
    first.cleanup()

    assert _processes_of(account.uid) == []
    assert _run(["getent", "passwd", account.name]).returncode != 0
    assert not leftover.exists()
    second = backend.start_session(_request(tmp_path, client_key))
    try:
        assert second.account.uid != account.uid
    finally:
        second.stop(1)
        second.cleanup()


def test_a_planted_symlink_under_the_mount_root_is_never_followed(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    etc_owner = os.stat("/etc").st_uid
    root = Path(str(process_module.SAFE_MOUNT_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    (root / "output").symlink_to("/etc")
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(
        _request(tmp_path, client_key, output="/mnt/flowmesh/output/sub")
    )
    try:
        output = root / "output"
        assert output.is_dir() and not output.is_symlink()
        assert os.stat("/etc").st_uid == etc_owner
    finally:
        session.stop(1)
        session.cleanup()
    assert os.stat("/etc").st_uid == etc_owner


@pytest.mark.parametrize(
    "path", ["/mnt/flowmesh/../../etc", "/mnt/flowmesh/inputs/../../../etc"]
)
def test_a_mount_path_out_of_the_mount_root_is_refused(
    worker: WorkerConfig, tmp_path: Path, client_key: Path, path: str
) -> None:
    etc_owner = os.stat("/etc").st_uid
    backend = ProcessSessionBackend(worker)

    with pytest.raises(ExecutionError):
        backend.start_session(_request(tmp_path, client_key, output=path))

    assert os.stat("/etc").st_uid == etc_owner


def test_a_session_left_by_a_dead_worker_is_reaped_at_start(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(_request(tmp_path, client_key))
    assert session.wait_ready(30) is not None
    process = cast(Any, session)._process
    account = session.account

    # A new worker process finds what the old one left.
    ProcessSessionBackend(worker).reap_stale()

    process.wait(timeout=10)
    assert _run(["getent", "passwd", account.name]).returncode != 0
    for root in worker.state_roots:
        assert _acl_users(root) == [], root
    assert list(process_module.SESSIONS_ROOT.iterdir()) == []


def test_a_second_session_on_the_worker_is_refused(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(_request(tmp_path, client_key))
    try:
        with pytest.raises(ExecutionError, match="already has an SSH session"):
            backend.start_session(_request(tmp_path, client_key))
    finally:
        session.stop(1)
        session.cleanup()


def test_the_backend_is_available_on_a_root_worker(worker: WorkerConfig) -> None:
    assert ProcessSessionBackend.is_available(worker)
