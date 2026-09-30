"""The process session backend against a real sshd and real accounts.

It creates accounts and starts sshd, so it runs only as root inside a container,
never on a host.
"""

import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import threading
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
    session_identity,
)
from worker.executors.ssh_session.backends import process as process_module
from worker.executors.ssh_session.config import SSHOutputConfig

_ROOT_IN_CONTAINER = os.getuid() == 0 and any(
    Path(marker).exists() for marker in ("/.dockerenv", "/run/.containerenv")
)
# Set where the suite must run, so it fails rather than skips there.
if os.environ.get("FLOWMESH_TEST_PROCESS_SSH") == "1" and not _ROOT_IN_CONTAINER:
    raise RuntimeError("the process SSH backend suite needs root inside a container")

pytestmark = pytest.mark.skipif(
    not _ROOT_IN_CONTAINER,
    reason="creates accounts and starts sshd: root inside a container only",
)

_WORKER_SECRET = "worker-secret-sentinel"
_WORKER_TOKEN = "worker-token-sentinel"
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
    # The model cache as deployments share it: open to every uid, twice over.
    hub = roots["hf"] / "hub"
    hub.mkdir()
    for path in (roots["hf"], hub):
        path.chmod(0o777)
    (hub / "model").write_text("cached model")
    (hub / "model").chmod(0o666)
    roots["private"].chmod(0o700)
    other = roots["results"] / "tsk-other"
    other.mkdir(mode=0o777)
    (other / "results.json").write_text('{"other": "tenant"}')
    (other / "results.json").chmod(0o666)
    (roots["content"] / "object").write_text("cached content")
    (roots["content"] / "object").chmod(0o666)
    (roots["private"] / "state").write_text("private state")
    # Named after the worker token, in a directory any uid may list.
    hb_dir = base / "hb"
    hb_dir.mkdir(mode=0o755)
    hb_file = hb_dir / f"{_WORKER_TOKEN}.hb"
    hb_file.write_text("alive")
    hb_file.chmod(0o666)
    config = make_worker_config(
        results_dir=roots["results"],
        private_state_dir=roots["private"],
        content_dir=roots["content"],
        hb_file=hb_file,
        state_dirs=(Path("/root"), roots["hf"], hub),
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
        requested_image=None,
        requested_user=None,
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


def _acl_entries(path: Path) -> list[str]:
    """The named-user entries on ``path``, and the mask one leaves behind."""
    out = _run(["getfacl", "-cp", path.as_posix()]).stdout
    return [
        line
        for line in out.splitlines()
        if (line.startswith("user:") and "::" not in line) or line.startswith("mask:")
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

        launched = psutil.Process(cast(Any, session)._process.pid)
        for proc in (launched, *launched.children(recursive=True)):
            assert proc.nice() == 10
            assert Path(f"/proc/{proc.pid}/oom_score_adj").read_text().strip() == "1000"
        shell = _ssh(session, client_key, port, "nice; cat /proc/self/oom_score_adj")
        assert shell.stdout.split() == ["10", "1000"], shell.stderr

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
            worker.state_dirs[2] / "model",
            worker.hb_file,
            Path("/proc/1/environ"),
        ):
            leaked = _ssh(session, client_key, port, _read(denied))
            assert leaked.stdout == "", (denied, leaked.stdout)
        for root in process_module.denied_roots(worker):
            listed = _ssh(session, client_key, port, _list(root))
            assert listed.stdout == "", (root, listed.stdout)
        assert (
            _WORKER_TOKEN
            not in _ssh(session, client_key, port, _list(worker.hb_file.parent)).stdout
        )
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
    for root in process_module.denied_roots(worker):
        assert _acl_entries(root) == [], root
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


def _live_processes_of(uid: int) -> list[psutil.Process]:
    return [
        p
        for p in psutil.process_iter(["uids", "status"])
        if p.info["uids"].real == uid and p.info["status"] != psutil.STATUS_ZOMBIE
    ]


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

    assert _live_processes_of(account.uid) == []
    assert _run(["getent", "passwd", account.name]).returncode != 0
    assert not leftover.exists()
    second = backend.start_session(_request(tmp_path, client_key))
    try:
        assert second.account.uid != account.uid
    finally:
        second.stop(1)
        second.cleanup()


def test_a_state_root_the_worker_has_not_made_yet_is_denied(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    relocated = Path(tempfile.mkdtemp(dir="/var/lib"))
    relocated.chmod(0o755)
    content = relocated / "content"
    backend = ProcessSessionBackend(dataclasses.replace(worker, content_dir=content))
    session = backend.start_session(_request(tmp_path, client_key))
    try:
        port = session.wait_ready(30)
        assert port is not None
        # The worker makes its content cache, readable by mode, once it needs it.
        content.mkdir(exist_ok=True)
        content.chmod(0o755)
        (content / "object").write_text("cached content")
        (content / "object").chmod(0o644)

        assert _ssh(session, client_key, port, _read(content / "object")).stdout == ""
    finally:
        session.stop(1)
        session.cleanup()
        shutil.rmtree(relocated, ignore_errors=True)


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
    for root in process_module.denied_roots(worker):
        assert _acl_entries(root) == [], root
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


def test_a_uid_another_worker_denied_a_shared_root_is_never_drawn(
    worker: WorkerConfig,
    tmp_path: Path,
    client_key: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A live session of another worker sharing the results volume.
    peer_uid = session_identity.SESSION_UID_MIN + 7
    _run(["setfacl", "-m", f"u:{peer_uid}:---", worker.results_dir.as_posix()])
    draws = iter([7, 8])
    monkeypatch.setattr(session_identity.secrets, "randbelow", lambda n: next(draws))
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(_request(tmp_path, client_key))
    try:
        assert session.account.uid == session_identity.SESSION_UID_MIN + 8
    finally:
        session.stop(1)
        session.cleanup()

    assert _acl_entries(worker.results_dir)[0] == f"user:{peer_uid}:---"


def _zombies_of(uid: int) -> list[psutil.Process]:
    return [
        p
        for p in psutil.process_iter(["uids", "status"])
        if p.info["uids"].real == uid and p.info["status"] == psutil.STATUS_ZOMBIE
    ]


@pytest.mark.skipif(shutil.which("tini") is None, reason="needs tini")
def test_what_a_session_orphans_is_reaped_whatever_runs_as_pid_1(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(_request(tmp_path, client_key))
    account = session.account
    try:
        port = session.wait_ready(30)
        assert port is not None
        started = _ssh(
            session,
            client_key,
            port,
            "setsid sh -c 'while :; do (sleep 0.01 &); sleep 0.005; done' "
            ">/dev/null 2>&1 < /dev/null & sleep 1",
        )
        assert started.returncode == 0, started.stderr
    finally:
        session.stop(1)
        session.cleanup()

    assert _zombies_of(account.uid) == []


_DEEP_TREE = """
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)
for _ in range(3000):
    os.mkdir("d", dir_fd=fd)
    child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
    os.close(fd)
    fd = child
"""


def test_a_deep_tree_a_session_leaves_does_not_wedge_the_worker(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    request = _request(tmp_path, client_key, output="/mnt/flowmesh/output")
    first = backend.start_session(request)
    try:
        port = first.wait_ready(30)
        assert port is not None
        scratch = Path(tempfile.gettempdir()) / f"deep-{first.account.name}"
        built = _ssh(
            first,
            client_key,
            port,
            f"ulimit -n 1024; mkdir {scratch} && "
            f"python3 -c '{_DEEP_TREE}' /mnt/flowmesh/output && "
            f"python3 -c '{_DEEP_TREE}' {scratch}",
        )
        assert built.returncode == 0, built.stderr
    finally:
        first.stop(1)
        first.cleanup()

    assert not scratch.exists()
    assert not (process_module.SESSIONS_ROOT / request.session_id).exists()
    second = backend.start_session(_request(tmp_path, client_key))
    second.stop(1)
    second.cleanup()


def test_the_mount_root_holds_only_links_into_the_session_s_own_directory(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    request = _request(tmp_path, client_key, output="/mnt/flowmesh/out/data")
    session = backend.start_session(request)
    try:
        link = Path("/mnt/flowmesh/out/data")
        assert link.is_symlink() and os.lstat(link).st_uid == 0
        assert os.lstat(link.parent).st_uid == 0
        assert link.readlink().is_relative_to(process_module.SESSIONS_ROOT)
        assert os.stat(link).st_uid == session.account.uid
    finally:
        session.stop(1)
        session.cleanup()


def test_a_filesystem_mounted_below_the_mount_root_is_left_alone(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    data = Path(str(process_module.SAFE_MOUNT_ROOT)) / "data"
    data.mkdir(parents=True, exist_ok=True)
    if _run(["mount", "-t", "tmpfs", "tmpfs", data.as_posix()]).returncode != 0:
        pytest.skip("this container may not mount")
    try:
        (data / "kept").write_text("operator data")
        backend = ProcessSessionBackend(worker)
        with pytest.raises(ExecutionError, match="mounted below") as refused:
            backend.start_session(
                _request(tmp_path, client_key, output="/mnt/flowmesh/output")
            )
        assert refused.value.retryable
        assert (data / "kept").read_text() == "operator data"
    finally:
        _run(["umount", data.as_posix()])


def test_a_uid_drawn_again_inherits_no_ipc_object_of_its_last_holder(
    worker: WorkerConfig,
    tmp_path: Path,
    client_key: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_identity.secrets, "randbelow", lambda n: 11)
    backend = ProcessSessionBackend(worker)
    first = backend.start_session(_request(tmp_path, client_key))
    try:
        port = first.wait_ready(30)
        assert port is not None
        made = _ssh(first, client_key, port, "ipcmk -M 4096 && ipcmk -Q && ipcmk -S 1")
        assert made.returncode == 0, made.stderr
        assert len(_ipc_objects_of(first.account.uid)) == 3
    finally:
        first.stop(1)
        first.cleanup()

    assert _ipc_objects_of(first.account.uid) == []
    second = backend.start_session(_request(tmp_path, client_key))
    try:
        assert second.account.uid == first.account.uid
        assert _ipc_objects_of(second.account.uid) == []
    finally:
        second.stop(1)
        second.cleanup()


def _ipc_objects_of(uid: int) -> list[str]:
    owner_columns = {"shm": (7, 9), "msg": (7, 9), "sem": (4, 6)}
    found = []
    for table, columns in owner_columns.items():
        for line in Path("/proc/sysvipc", table).read_text().splitlines()[1:]:
            fields = line.split()
            if uid in {int(fields[column]) for column in columns}:
                found.append(f"{table} {fields[1]}")
    return found


def _useradd_default_gid() -> int:
    defaults = _run(["useradd", "-D"]).stdout
    return int(
        next(line for line in defaults.splitlines() if line.startswith("GROUP="))[6:]
    )


def test_a_session_cannot_swap_a_directory_above_worker_state(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    base = Path(tempfile.mkdtemp(dir="/var/lib"))
    base.chmod(0o755)
    # Writable by the group useradd hands an account by default, as on a lab host.
    lab = base / "lab"
    lab.mkdir()
    os.chown(lab, 0, _useradd_default_gid())
    lab.chmod(0o2775)
    (lab / "cache").mkdir(mode=0o755)
    config = dataclasses.replace(worker, state_dirs=(*worker.state_dirs, lab / "cache"))
    backend = ProcessSessionBackend(config)
    session = backend.start_session(_request(tmp_path, client_key))
    try:
        port = session.wait_ready(30)
        assert port is not None
        groups = _ssh(session, client_key, port, "id -G").stdout.split()
        assert groups == [str(session.account.gid)], groups
        moved = _ssh(session, client_key, port, f"mv {lab / 'cache'} {lab / 'mine'}")
        assert moved.returncode != 0
        assert (lab / "cache").is_dir() and not (lab / "mine").exists()
    finally:
        session.stop(1)
        session.cleanup()
        shutil.rmtree(base, ignore_errors=True)
    assert _run(["getent", "group", session.account.name]).returncode != 0


def test_state_below_a_directory_any_account_can_write_refuses_sessions(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    base = Path(tempfile.mkdtemp(dir="/var/lib"))
    base.chmod(0o755)
    (base / "open" / "sub" / "results").mkdir(parents=True)
    (base / "open").chmod(0o777)
    config = dataclasses.replace(
        worker, state_dirs=(*worker.state_dirs, base / "open" / "sub" / "results")
    )
    try:
        with pytest.raises(ExecutionError, match="world-writable") as refused:
            ProcessSessionBackend(config).start_session(_request(tmp_path, client_key))
        assert refused.value.retryable
        assert not ProcessSessionBackend.is_available(config)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert _run(["sh", "-c", "getent passwd | grep -c '^fmssn'"]).stdout.strip() == "0"


def test_a_stop_during_collection_lets_the_collection_finish(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(
        _request(tmp_path, client_key, output="/mnt/flowmesh/output")
    )
    collected = tmp_path / "collected"
    try:
        port = session.wait_ready(30)
        assert port is not None
        made = _ssh(
            session,
            client_key,
            port,
            "for i in $(seq 64); do head -c 4M /dev/urandom "
            "> /mnt/flowmesh/output/part-$i; done",
        )
        assert made.returncode == 0, made.stderr
        errors: list[BaseException] = []

        def collect() -> None:
            try:
                session.collect_output(collected, None)
            except BaseException as exc:
                errors.append(exc)

        collector = threading.Thread(target=collect)
        collector.start()
        deadline = time.monotonic() + 30
        while not any(collected.glob("part-*")) and time.monotonic() < deadline:
            time.sleep(0.01)
        session.stop(1)
        collector.join(timeout=120)

        assert errors == []
        assert len(list(collected.glob("part-*"))) == 64
        assert all(part.stat().st_size == 4 << 20 for part in collected.iterdir())
    finally:
        session.stop(1)
        session.cleanup()


def test_a_session_never_widens_what_a_root_s_acl_grants(
    worker: WorkerConfig, tmp_path: Path, client_key: Path
) -> None:
    results = worker.results_dir.as_posix()
    _run(["setfacl", "-m", "u:5000:rwx,g::rwx", results])
    _run(["setfacl", "-n", "-m", "m::r-x", results])
    before = _run(["getfacl", "-cp", results]).stdout
    backend = ProcessSessionBackend(worker)
    session = backend.start_session(_request(tmp_path, client_key))
    try:
        during = _run(["getfacl", "-cp", results]).stdout
        assert "mask::r-x" in during.splitlines(), during
    finally:
        session.stop(1)
        session.cleanup()

    assert _run(["getfacl", "-cp", results]).stdout == before
