"""The process session backend: which worker serves it, what a session is handed,
and what it leaves behind. Nothing here starts sshd or touches an account; the
in-container suite does that."""

import dataclasses
import io
import logging
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import typing
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from server.supervisor.adapters.ssh import SSHConfig as SupervisorSSHConfig
from shared.content.config import BACKEND_FILESYSTEM
from shared.schemas.worker import SSHBackendName
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    FakeContentPlane,
    make_live_worker_config,
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.config import WorkerConfig
from worker.executors.base_executor import (
    ExecutionError,
    RunSignals,
    TaskCancelledError,
)
from worker.executors.ssh_executor import SSHExecutor
from worker.executors.ssh_session import (
    DockerSessionBackend,
    ProcessSessionBackend,
    SessionRequest,
    select_backend_cls,
)
from worker.executors.ssh_session.backends import process as process_module
from worker.executors.ssh_session.base import (
    extract_output_archive,
    iter_tree,
    path_size_bytes,
)
from worker.executors.ssh_session.config import SSHOutputConfig
from worker.main import build_capabilities
from worker.runner import Runner

_SECRET = "tok-restored-SECRET"


def _config(backend: SSHBackendName, tmp_path: Path) -> WorkerConfig:
    return make_live_worker_config(tmp_path, ssh_session_backend=backend)


@pytest.mark.parametrize(
    ("backend", "docker", "process", "expected"),
    [
        (SSHBackendName.AUTO, True, True, DockerSessionBackend),
        (SSHBackendName.AUTO, False, True, ProcessSessionBackend),
        (SSHBackendName.AUTO, False, False, None),
        (SSHBackendName.DOCKER, False, True, None),
        (SSHBackendName.PROCESS, True, True, ProcessSessionBackend),
        (SSHBackendName.PROCESS, True, False, None),
        (SSHBackendName.OFF, True, True, None),
    ],
)
def test_the_backend_follows_the_setting_and_what_the_worker_can_isolate(
    tmp_path: Path,
    backend: SSHBackendName,
    docker: bool,
    process: bool,
    expected: type | None,
) -> None:
    with (
        patch.object(DockerSessionBackend, "is_available", return_value=docker),
        patch.object(ProcessSessionBackend, "is_available", return_value=process),
    ):
        assert select_backend_cls(_config(backend, tmp_path)) is expected


def test_a_non_root_worker_without_docker_serves_no_ssh(tmp_path: Path) -> None:
    config = _config(SSHBackendName.AUTO, tmp_path)
    with (
        patch.object(DockerSessionBackend, "is_available", return_value=False),
        patch.object(process_module, "process_identity_available", return_value=False),
    ):
        assert SSHExecutor.is_available(config) is False


def _servable(**patches: Any) -> Any:
    defaults: dict[str, Any] = {
        "process_identity_available": True,
        "find_sshd": "/usr/sbin/sshd",
        "find_ssh_keygen": "/usr/bin/ssh-keygen",
        "find_tar": "/usr/bin/tar",
        "_acl_ready": True,
        "_acquire_backend_lock": True,
        **patches,
    }
    return patch.multiple(
        process_module,
        **{name: MagicMock(return_value=value) for name, value in defaults.items()},
    )


@pytest.mark.parametrize("unready", ["_acl_ready", "_acquire_backend_lock"])
def test_a_worker_that_cannot_isolate_or_lock_serves_no_process_session(
    tmp_path: Path, unready: str
) -> None:
    config = make_live_worker_config(tmp_path, ssh_relay_host="10.0.0.9")
    with _servable():
        assert ProcessSessionBackend.is_available(config)
    with _servable(**{unready: False}):
        assert not ProcessSessionBackend.is_available(config)


def test_a_root_worker_its_supervisor_cannot_reach_serves_no_process_session(
    tmp_path: Path,
) -> None:
    with (
        _servable(),
        patch.object(process_module, "resolve_tailnet_address", return_value=None),
    ):
        assert not ProcessSessionBackend.is_available(make_live_worker_config(tmp_path))
        assert ProcessSessionBackend.is_available(
            make_live_worker_config(tmp_path, ssh_relay_host="10.0.0.9")
        )


@pytest.mark.parametrize(
    ("backend", "noninteractive"),
    [(SSHBackendName.DOCKER, True), (SSHBackendName.PROCESS, False)],
)
def test_the_worker_reports_whether_its_sessions_run_batch_tasks(
    tmp_path: Path, backend: SSHBackendName, noninteractive: bool
) -> None:
    with (
        patch.object(DockerSessionBackend, "is_available", return_value=True),
        patch.object(ProcessSessionBackend, "is_available", return_value=True),
        patch.object(ProcessSessionBackend, "reap_stale"),
    ):
        executor = SSHExecutor(_config(backend, tmp_path))

    capabilities = build_capabilities({"ssh": executor})

    assert TaskType.SSH in capabilities.supported_task_types
    assert capabilities.ssh_noninteractive is noninteractive


def _worker_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORKER_TOKEN", "tok")
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "127.0.0.1:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.delenv("SSH_SESSION_BACKEND", raising=False)


@pytest.mark.parametrize(
    ("enable_ssh", "expected"), [(False, None), (True, ProcessSessionBackend)]
)
def test_a_root_worker_without_docker_serves_ssh_only_where_it_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_ssh: bool,
    expected: type | None,
) -> None:
    _worker_env(monkeypatch, tmp_path)
    for name, value in (
        SupervisorSSHConfig(session_backend=None).to_env(enable_ssh).items()
    ):
        monkeypatch.setenv(name, value)

    with (
        patch.object(DockerSessionBackend, "is_available", return_value=False),
        patch.object(ProcessSessionBackend, "is_available", return_value=True),
    ):
        assert select_backend_cls(WorkerConfig.from_env()) is expected


def test_a_worker_given_no_backend_never_serves_a_process_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _worker_env(monkeypatch, tmp_path)

    with (
        patch.object(DockerSessionBackend, "is_available", return_value=False),
        patch.object(ProcessSessionBackend, "is_available", return_value=True),
    ):
        assert select_backend_cls(WorkerConfig.from_env()) is None


def test_an_unknown_session_backend_stops_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKER_TOKEN", "tok")
    monkeypatch.setenv("SUPERVISOR_GRPC_TARGET", "127.0.0.1:50051")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("WORKER_HB_FILE", str(tmp_path / "worker.hb"))
    monkeypatch.setenv("SSH_SESSION_BACKEND", "Process")
    assert WorkerConfig.from_env().ssh_session_backend is SSHBackendName.PROCESS

    monkeypatch.setenv("SSH_SESSION_BACKEND", "podman")
    with pytest.raises(SystemExit, match="SSH_SESSION_BACKEND"):
        WorkerConfig.from_env()


# ------------------------------------------------------------------ #
# The worker's state roots
# ------------------------------------------------------------------ #


def _path_typed(annotation: Any) -> bool:
    if annotation is Path:
        return True
    return any(_path_typed(arg) for arg in typing.get_args(annotation))


def test_every_worker_path_field_is_classified_exactly_once() -> None:
    hints = typing.get_type_hints(WorkerConfig)
    path_fields = {
        f.name for f in dataclasses.fields(WorkerConfig) if _path_typed(hints[f.name])
    }
    denied = set(process_module.DENIED_CONFIG_FIELDS)
    allowed = set(process_module.ALLOWED_CONFIG_FIELDS)

    assert not denied & allowed
    assert path_fields == denied | allowed


def _state_config(tmp_path: Path, **overrides: Any) -> WorkerConfig:
    fields: dict[str, Any] = {
        "results_dir": tmp_path / "results",
        "private_state_dir": tmp_path / "private",
        "content_dir": tmp_path / "content",
        "hb_file": tmp_path / "hb" / "worker-token.hb",
        "state_dirs": (tmp_path / "home", tmp_path / "hf"),
    }
    return make_worker_config(**{**fields, **overrides})


def test_the_denied_paths_cover_the_worker_state_and_a_filesystem_store(
    tmp_path: Path,
) -> None:
    config = _state_config(tmp_path)
    store = dataclasses.replace(
        config.object_store,
        backend=BACKEND_FILESYSTEM,
        filesystem_root=tmp_path / "store",
    )

    denied = process_module.denied_roots(
        dataclasses.replace(config, object_store=store)
    )

    assert set(denied) == {
        tmp_path / name
        for name in ("results", "private", "content", "hb", "home", "hf", "store")
    }


def test_the_heartbeat_directory_is_denied_not_just_the_file(tmp_path: Path) -> None:
    denied = process_module.denied_roots(_state_config(tmp_path))

    assert tmp_path / "hb" in denied
    assert tmp_path / "hb" / "worker-token.hb" not in denied


def test_the_state_dirs_come_from_home_the_caches_and_temp_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _worker_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_HOME", (tmp_path / "hf").as_posix())
    monkeypatch.setenv("TORCH_HOME", (tmp_path / "torch").as_posix())
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    dirs = WorkerConfig.from_env().state_dirs

    assert Path.home() in dirs
    assert {tmp_path / "hf", tmp_path / "torch"} <= set(dirs)
    assert Path(tempfile.gettempdir()) / "fastembed_cache" in dirs


def test_every_missing_state_root_is_created_private_to_the_worker(
    tmp_path: Path,
) -> None:
    config = _state_config(tmp_path, content_dir=tmp_path / "relocated" / "content")

    roots = process_module.ensure_state_roots(config)

    assert set(roots) == set(process_module.denied_roots(config))
    for root in roots:
        assert root.is_dir() and stat.S_IMODE(root.stat().st_mode) == 0o700, root


@pytest.mark.parametrize(
    "covering",
    [Path("/"), Path("/mnt"), Path(tempfile.gettempdir()), Path("/var/lib/flowmesh")],
)
def test_a_state_root_covering_what_sessions_need_is_refused(
    tmp_path: Path, covering: Path
) -> None:
    config = _state_config(tmp_path, state_dirs=(covering,))

    with pytest.raises(ExecutionError, match="would also deny"):
        process_module.ensure_state_roots(config)


def test_a_heartbeat_in_the_temp_dir_refuses_the_backend(tmp_path: Path) -> None:
    config = _state_config(
        tmp_path, hb_file=Path(tempfile.gettempdir()) / "worker-token.hb"
    )

    with pytest.raises(ExecutionError, match="would also deny"):
        process_module.ensure_state_roots(config)
    with patch.object(process_module.acl, "tools_available", return_value=True):
        assert not process_module._acl_ready(config)


def test_an_operator_link_is_resolved_to_its_target(tmp_path: Path) -> None:
    data = tmp_path / "data" / "results"
    data.mkdir(parents=True)
    (tmp_path / "results").symlink_to(data)

    roots = process_module.denied_roots(_state_config(tmp_path))

    assert data in roots
    assert process_module._root_problem(data) is None


@pytest.mark.parametrize("mode", [0o1777, 0o777])
def test_a_link_in_a_shared_dir_is_refused_not_followed(
    tmp_path: Path, mode: int
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(mode)
    victim = tmp_path / "victim"
    victim.mkdir()
    (shared / "cache").symlink_to(victim)
    config = _state_config(tmp_path, state_dirs=(shared / "cache",))

    with pytest.raises(ExecutionError, match="a link in a shared directory"):
        process_module.ensure_state_roots(config)


def test_a_path_through_a_link_in_a_shared_dir_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    (tmp_path / "elsewhere" / "results").mkdir(parents=True)
    (shared / "link").symlink_to(tmp_path / "elsewhere")
    config = _state_config(tmp_path, results_dir=shared / "link" / "results")

    with pytest.raises(ExecutionError, match="a link in a shared directory"):
        process_module.ensure_state_roots(config)


def _open_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o777)
    return path


def test_an_open_dir_anywhere_above_a_root_is_denied(tmp_path: Path) -> None:
    open_dir = _open_dir(tmp_path / "x" / "open")
    (open_dir / "sub" / "results").mkdir(parents=True)

    roots = process_module.ensure_state_roots(
        _state_config(tmp_path, results_dir=open_dir / "sub" / "results")
    )

    assert open_dir in roots
    assert open_dir / "sub" / "results" not in roots


def test_an_open_dir_above_a_missing_part_of_a_root_is_denied_not_created_through(
    tmp_path: Path,
) -> None:
    open_dir = _open_dir(tmp_path / "x" / "open")

    roots = process_module.ensure_state_roots(
        _state_config(tmp_path, results_dir=open_dir / "missing" / "results")
    )

    assert open_dir in roots
    assert not (open_dir / "missing").exists()


def test_an_open_dir_a_link_leads_through_is_denied_with_the_target(
    tmp_path: Path,
) -> None:
    open_dir = _open_dir(tmp_path / "far" / "open")
    (open_dir / "results").mkdir()
    (tmp_path / "results").symlink_to(open_dir / "results")
    closed = tmp_path / "closed"
    (closed / "real").mkdir(parents=True)
    _open_dir(tmp_path / "lead")
    (tmp_path / "lead" / "sub").mkdir()
    (tmp_path / "lead" / "sub" / "hop").symlink_to(closed / "real")

    roots = process_module.denied_roots(
        _state_config(tmp_path, content_dir=tmp_path / "lead" / "sub" / "hop")
    )

    assert open_dir in roots
    assert {tmp_path / "lead", closed / "real"} <= set(roots)


def test_a_root_others_can_write_in_a_shared_dir_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    (shared / "cache").mkdir()
    (shared / "cache").chmod(0o777)

    assert process_module._root_problem(shared / "cache") is not None
    assert process_module._root_problem(tmp_path / "missing") is None


def test_a_filesystem_that_takes_no_acl_offers_no_process_backend(
    tmp_path: Path,
) -> None:
    with (
        patch.object(process_module.acl, "tools_available", return_value=True),
        patch.object(
            process_module.acl, "probe", side_effect=ExecutionError("no ACLs here")
        ),
    ):
        assert not process_module._acl_ready(_state_config(tmp_path))


# ------------------------------------------------------------------ #
# The backend lock
# ------------------------------------------------------------------ #


def test_one_worker_per_lock_file_holds_the_process_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(process_module.tempfile, "tempdir", tmp_path.as_posix())
    monkeypatch.setattr(process_module, "_backend_lock_fd", None)
    lock = tmp_path / process_module._BACKEND_LOCK_NAME
    try:
        assert process_module._acquire_backend_lock()
        assert process_module._acquire_backend_lock()
        other = subprocess.run(  # nosec B603 - argv list, test-only
            [
                sys.executable,
                "-c",
                "import fcntl, os, sys\n"
                "fd = os.open(sys.argv[1], os.O_RDWR)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n",
                lock.as_posix(),
            ],
            capture_output=True,
            check=False,
        )
        assert other.returncode != 0
    finally:
        if (fd := process_module._backend_lock_fd) is not None:
            os.close(fd)


# ------------------------------------------------------------------ #
# Mount paths
# ------------------------------------------------------------------ #


def _request(tmp_path: Path, inputs: list[str], output: str | None) -> SessionRequest:
    cfg = MagicMock(
        output=None if output is None else SSHOutputConfig(output, None),
        interactive=True,
    )
    resolved = [
        MagicMock(mount_path=path, task_id=f"tsk-{index}")
        for index, path in enumerate(inputs)
    ]
    return SessionRequest(
        task_id="tsk-ssh",
        session_id="ssn-1",
        worker_name="worker-1",
        cfg=cfg,
        out_dir=tmp_path,
        resolved_inputs=cast(Any, resolved),
        signals=RunSignals(),
    )


@pytest.mark.parametrize(
    ("inputs", "output"),
    [
        (["/mnt/flowmesh/in"], "/mnt/flowmesh/in/out"),
        (["/mnt/flowmesh/out/in"], "/mnt/flowmesh/out"),
        (["/mnt/flowmesh"], None),
        (["/mnt/flowmesh/a", "/mnt/flowmesh/a/b"], None),
    ],
)
def test_mount_paths_that_nest_are_refused(
    tmp_path: Path, inputs: list[str], output: str | None
) -> None:
    with pytest.raises(ExecutionError):
        process_module._plan_mounts(_request(tmp_path, inputs, output))


def test_the_mount_root_is_emptied_without_following_a_link(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("keep")
    root = tmp_path / "mnt"
    (root / "old" / "deep").mkdir(parents=True)
    (root / "link").symlink_to(outside)
    (root / "old" / "link").symlink_to(outside)
    root.chmod(0o700)

    process_module._reset_mount_root(root, create=True)

    assert list(root.iterdir()) == []
    assert (outside / "precious").read_text() == "keep"
    assert stat.S_IMODE(root.stat().st_mode) == 0o755


def test_a_linked_mount_root_is_replaced_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("keep")
    root = tmp_path / "mnt"
    root.symlink_to(outside)

    process_module._reset_mount_root(root, create=True)

    assert root.is_dir() and not root.is_symlink()
    assert (outside / "precious").read_text() == "keep"


def test_a_filesystem_mounted_below_the_mount_root_is_never_emptied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mnt root"
    (root / "data").mkdir(parents=True)
    (root / "data" / "kept").write_text("operator data")
    mountinfo = tmp_path / "mountinfo"
    escaped = (root / "data").as_posix().replace(" ", "\\040")
    mountinfo.write_text(
        "23 28 0:22 / /proc rw,relatime - proc proc rw\n"
        f"90 28 8:1 /srv/data {escaped} rw,relatime - ext4 /dev/sda1 rw\n"
    )
    monkeypatch.setattr(process_module, "_MOUNTINFO", mountinfo)

    with pytest.raises(OSError, match="mounted"):
        process_module._reset_mount_root(root, create=True)

    assert (root / "data" / "kept").read_text() == "operator data"


def test_a_mount_on_another_device_below_the_root_is_never_emptied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mnt"
    (root / "data").mkdir(parents=True)
    (root / "data" / "kept").write_text("operator data")
    real_lstat = os.lstat

    def lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        info = real_lstat(path, *args, **kwargs)
        if Path(path) == root / "data":
            fields = list(info)
            fields[stat.ST_DEV] += 1
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(process_module, "_MOUNTINFO", tmp_path / "unreadable")
    monkeypatch.setattr(process_module.os, "lstat", lstat)

    with pytest.raises(OSError, match="mounted"):
        process_module._reset_mount_root(root, create=True)

    assert (root / "data" / "kept").read_text() == "operator data"


def test_a_mount_path_is_a_link_made_one_level_at_a_time(tmp_path: Path) -> None:
    root = tmp_path / "mnt"
    root.mkdir()
    target = tmp_path / "session" / "output"
    target.mkdir(parents=True)

    process_module._link_mount_path(root, (root / "a" / "b" / "out").as_posix(), target)

    link = root / "a" / "b" / "out"
    assert link.is_symlink() and link.readlink() == target
    assert stat.S_IMODE((root / "a").stat().st_mode) == 0o755


def test_a_mount_path_through_a_link_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "mnt"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "a").symlink_to(outside)

    with pytest.raises(ExecutionError, match="conflicts"):
        process_module._link_mount_path(root, (root / "a" / "out").as_posix(), tmp_path)

    assert list(outside.iterdir()) == []


# ------------------------------------------------------------------ #
# Output collection
# ------------------------------------------------------------------ #


def _archive(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> list[bytes]:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
            tar.addfile(info, None if data is None else io.BytesIO(data))
    raw = stream.getvalue()
    return [raw[i : i + 512] for i in range(0, len(raw), 512)]


def _member(
    name: str, kind: bytes = tarfile.REGTYPE, link: str = ""
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    return info


def test_collection_extracts_files_and_directories_and_nothing_else(
    tmp_path: Path,
) -> None:
    chunks = _archive(
        [
            (_member("./sub", tarfile.DIRTYPE), None),
            (_member("./result.txt"), b"ok"),
            (_member("./sub/nested.txt"), b"nested"),
            (_member("./secret-link", tarfile.SYMTYPE, "/etc/shadow"), None),
            (_member("./hard-link", tarfile.LNKTYPE, "./result.txt"), None),
            (_member("./pipe", tarfile.FIFOTYPE), None),
            (_member("/etc/escaped"), b"x"),
            (_member("../escaped"), b"x"),
        ]
    )
    destination = tmp_path / "collected"

    extract_output_archive(chunks, destination, None, lambda: None)

    collected = sorted(
        p.relative_to(destination).as_posix() for p in destination.rglob("*")
    )
    assert collected == ["etc", "etc/escaped", "result.txt", "sub", "sub/nested.txt"]
    assert (destination / "result.txt").read_text() == "ok"


def test_the_output_size_counts_regular_files_without_following_a_link(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("worker state")
    output = tmp_path / "output"
    (output / "sub").mkdir(parents=True)
    (output / "result.txt").write_text("ok")
    (output / "sub" / "nested.txt").write_text("nested")
    (output / "secret-link").symlink_to(outside / "secret")
    (output / "dir-link").symlink_to(outside)

    assert path_size_bytes(output) == len("ok") + len("nested")


def test_a_linked_output_root_is_not_walked(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "f").write_text("data")
    (tmp_path / "link").symlink_to(real)

    assert list(iter_tree(tmp_path / "link")) == []


def test_an_output_too_deep_to_walk_fails_the_size_check(tmp_path: Path) -> None:
    deepest = tmp_path.joinpath(*(["d"] * 70))
    deepest.mkdir(parents=True)
    (deepest / "f").write_bytes(b"x")

    with pytest.raises(ExecutionError, match="deeper"):
        path_size_bytes(tmp_path)


def test_collection_fails_past_the_output_limit(tmp_path: Path) -> None:
    chunks = _archive([(_member("./big.bin"), b"x" * 100)])

    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        extract_output_archive(chunks, tmp_path / "c", 10, lambda: None)


def test_a_cancel_during_collection_ends_it(tmp_path: Path) -> None:
    chunks = _archive([(_member("./a.bin"), b"x")])
    signals = RunSignals()

    with signals.running("tsk"):
        signals.cancel("tsk")
        with pytest.raises(TaskCancelledError):
            extract_output_archive(
                chunks, tmp_path / "c", None, signals.raise_if_cancelled
            )


# ------------------------------------------------------------------ #
# What a session is handed
# ------------------------------------------------------------------ #


def test_helpers_start_from_a_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKER_TOKEN", "worker-secret")
    monkeypatch.setenv("FLOWMESH_API_KEY", "api-secret")

    env = process_module._sanitized_spawn_env()

    assert "worker-secret" not in env.values()
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "TZ"}


def test_the_session_env_travels_on_its_keys() -> None:
    rendered, exported = process_module._render_authorized_keys(
        ["ssh-ed25519 AAAA one", " "],
        {"TOKEN": "abc", "BAD NAME": "x", "QUOTED": 'a"b'},
    )

    assert rendered == 'environment="TOKEN=abc" ssh-ed25519 AAAA one\n'
    assert exported == ["TOKEN"]


def test_sshd_admits_only_the_session_account_by_key() -> None:
    config = process_module._render_sshd_config(
        port=2222,
        session_dir=Path("/run/s"),
        host_key=Path("/run/s/key"),
        authorized_keys=Path("/run/s/authorized_keys"),
        login_user="fmssnabc",
        exported_env=["TOKEN"],
    )

    for line in (
        "AllowUsers fmssnabc",
        "PasswordAuthentication no",
        "PermitRootLogin no",
        "AllowTcpForwarding no",
        "PermitUserEnvironment TOKEN",
    ):
        assert line in config.splitlines()


# ------------------------------------------------------------------ #
# Readiness under a cancel or stop
# ------------------------------------------------------------------ #


def _session(signals: RunSignals, process: Any) -> process_module.ProcessSession:
    return process_module.ProcessSession(
        backend=MagicMock(),
        process=process,
        port=1,
        session_dir=Path("/nonexistent"),
        log_path=Path("/nonexistent/sshd.log"),
        output_path=None,
        finish_sentinel=Path("/nonexistent/finish"),
        account=MagicMock(),
        signals=signals,
    )


@pytest.mark.parametrize("kind", ["stop", "cancel"])
def test_a_signal_in_the_last_readiness_poll_ends_as_requested(kind: str) -> None:
    signals = RunSignals()
    process = MagicMock()
    process.poll.return_value = None

    def sleep(_seconds: float) -> None:
        getattr(signals, kind)("tsk")

    with (
        signals.running("tsk"),
        patch.object(process_module, "is_ssh_ready", return_value=False),
        patch.object(process_module.time, "sleep", side_effect=sleep),
    ):
        if kind == "cancel":
            with pytest.raises(TaskCancelledError):
                _session(signals, process).wait_ready(0.01)
        else:
            assert _session(signals, process).wait_ready(0.01) is None


def test_the_finish_sentinel_is_never_followed(tmp_path: Path) -> None:
    sentinel = tmp_path / ".flowmesh_finish"
    session = _session(RunSignals(), MagicMock())
    session._finish_sentinel = sentinel

    assert not session.finish_requested()
    sentinel.symlink_to(tmp_path / "missing")
    assert session.finish_requested()


def test_output_is_not_collected_while_a_process_of_the_session_lives(
    tmp_path: Path,
) -> None:
    session = _session(RunSignals(), MagicMock())
    session._output_path = tmp_path / "output"
    with (
        patch.object(process_module, "kill_processes", return_value=False),
        patch.object(process_module, "_archive_as") as archive,
        pytest.raises(ExecutionError, match="collect its output") as refused,
    ):
        session.collect_output(tmp_path / "collected", None)

    archive.assert_not_called()
    assert not refused.value.retryable


@pytest.mark.parametrize(
    "error", [OSError("disk gone"), tarfile.ReadError("truncated")]
)
def test_a_collection_failure_fails_the_session_for_good(
    tmp_path: Path, error: Exception
) -> None:
    session = _session(RunSignals(), MagicMock())
    session._output_path = tmp_path / "output"
    with (
        patch.object(process_module, "kill_processes", return_value=True),
        patch.object(process_module, "_archive_as"),
        patch.object(process_module, "_end_archiver"),
        patch.object(process_module, "extract_output_archive", side_effect=error),
        pytest.raises(ExecutionError, match="Failed to collect") as failed,
    ):
        session.collect_output(tmp_path / "collected", None)

    assert not failed.value.retryable


def test_the_session_is_stopped_before_its_sshd_and_again_after() -> None:
    order: list[str] = []
    session = _session(RunSignals(), MagicMock())

    def kill(uid: int) -> bool:
        order.append("kill")
        return True

    with (
        patch.object(process_module, "kill_processes", side_effect=kill),
        patch.object(
            process_module,
            "_terminate",
            side_effect=lambda process, timeout: order.append("terminate"),
        ),
    ):
        session.stop(1)

    assert order == ["kill", "terminate", "kill"]


@pytest.mark.parametrize("clean", [True, False])
def test_a_cleanup_releases_the_session_once(clean: bool) -> None:
    backend = MagicMock()
    session = _session(RunSignals(), MagicMock())
    session._backend = backend
    with patch.object(
        process_module, "_discard_session", return_value=clean
    ) as discard:
        session.cleanup()
        session.cleanup()

    discard.assert_called_once()
    backend._release.assert_called_once_with(session, clean)


def test_a_session_that_fails_to_start_and_cannot_be_discarded_stops_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_module, "SESSIONS_ROOT", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    request = SessionRequest(
        task_id="tsk-ssh",
        session_id="ssn-0123456789abcdef",
        worker_name="worker-1",
        cfg=MagicMock(interactive=True, output=None, requested_image=None),
        out_dir=tmp_path / "out",
        resolved_inputs=[],
        signals=RunSignals(),
    )
    with (
        patch.object(process_module, "find_sshd", return_value="/usr/sbin/sshd"),
        patch.object(process_module, "find_ssh_keygen", return_value="/usr/bin/k"),
        patch.object(process_module, "_make_private_dir"),
        patch.object(
            process_module, "ensure_state_roots", side_effect=ExecutionError("no")
        ),
        patch.object(process_module, "_discard_session", return_value=False),
        pytest.raises(ExecutionError, match="no"),
    ):
        backend.start_session(request)

    with pytest.raises(ExecutionError, match="could not be rid of"):
        backend.start_session(request)


def test_an_sshd_that_ignores_a_terminate_is_killed_with_its_children() -> None:
    process = MagicMock(pid=4321)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("sshd", 1), 0]
    with patch.object(process_module, "_kill_tree") as kill_tree:
        process_module._terminate(process, 1)

    process.terminate.assert_called_once_with()
    kill_tree.assert_called_once_with(4321)


def test_sshd_runs_niced_and_under_a_subreaper_when_tini_is_there(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_module, "_missing_tini_logged", False)
    with patch.object(process_module.shutil, "which", return_value="/usr/bin/tini"):
        argv = process_module._launch_argv(["/usr/sbin/sshd", "-D"])

    assert argv[-7:] == [
        "10",
        "1000",
        "/usr/bin/tini",
        "-s",
        "--",
        "/usr/sbin/sshd",
        "-D",
    ]
    assert "oom_score_adj" in argv[argv.index("-c") + 1]

    with (
        patch.object(process_module.shutil, "which", return_value=None),
        caplog.at_level(logging.WARNING),
    ):
        assert process_module._launch_argv(["/usr/sbin/sshd"])[-1] == "/usr/sbin/sshd"
        process_module._launch_argv(["/usr/sbin/sshd"])
    assert sum("tini is missing" in r.message for r in caplog.records) == 1


# ------------------------------------------------------------------ #
# Reaping what a dead worker left
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("ours", [True, False])
def test_a_reap_signals_only_the_sshd_its_manifest_names(
    tmp_path: Path, ours: bool
) -> None:
    session_dir = tmp_path / "ssn-1"
    session_dir.mkdir()
    process_module.SessionManifest(
        session_dir=session_dir, account="fmssn1", sshd_pid=4321
    ).write()

    with (
        patch.object(process_module, "_is_our_sshd", return_value=ours),
        patch.object(process_module, "_kill_tree") as kill,
        patch.object(
            process_module.pwd, "getpwnam", return_value=MagicMock(pw_uid=61_001)
        ),
        patch.object(process_module, "retire_account", return_value=True) as retire,
    ):
        assert process_module.reap_session(session_dir)

    assert kill.call_count == (1 if ours else 0)
    retire.assert_called_once_with("fmssn1", 61_001)
    assert not session_dir.exists()


def test_a_reap_keeps_a_session_whose_account_cannot_be_removed(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "ssn-1"
    session_dir.mkdir()
    process_module.SessionManifest(session_dir=session_dir, account="fmssn1").write()

    with (
        patch.object(
            process_module.pwd, "getpwnam", return_value=MagicMock(pw_uid=61_001)
        ),
        patch.object(process_module, "retire_account", return_value=False),
    ):
        assert not process_module.reap_session(session_dir)

    assert session_dir.exists()


def test_a_reap_kills_the_sshd_below_its_subreaper(tmp_path: Path) -> None:
    child = MagicMock()
    parent = MagicMock()
    parent.children.return_value = [child]
    with (
        patch.object(process_module.psutil, "Process", return_value=parent),
        patch.object(process_module.psutil, "wait_procs") as wait,
    ):
        process_module._kill_tree(4321)

    for proc in (parent, child):
        proc.send_signal.assert_called_once_with(process_module.signal.SIGKILL)
    wait.assert_called_once()


def test_a_worker_serves_no_session_until_a_stuck_account_is_reaped(
    tmp_path: Path,
) -> None:
    backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    with (
        patch.object(process_module.os, "getuid", return_value=0),
        patch.object(process_module, "SESSIONS_ROOT", tmp_path / "sessions"),
        patch.object(process_module, "_clear_mount_root"),
        patch.object(process_module, "reap_stale_accounts", return_value=False),
    ):
        backend.reap_stale()
    with pytest.raises(ExecutionError, match="serves no session") as refused:
        backend.start_session(_request(tmp_path, [], None))
    assert refused.value.retryable

    with (
        patch.object(process_module.os, "getuid", return_value=0),
        patch.object(process_module, "SESSIONS_ROOT", tmp_path / "sessions"),
        patch.object(process_module, "_clear_mount_root"),
        patch.object(process_module, "reap_stale_accounts", return_value=True),
        patch.object(backend, "_create_session") as create,
    ):
        backend.reap_stale()
        backend.start_session(_request(tmp_path, [], None))
    create.assert_called_once()


def test_a_recycled_pid_is_not_taken_for_the_session_sshd(tmp_path: Path) -> None:
    config = (tmp_path / "sshd_config").as_posix()

    assert not process_module._is_our_sshd(os.getpid(), config)


def test_constructing_the_backend_reaps_nothing(tmp_path: Path) -> None:
    with (
        patch.object(process_module.os, "getuid", return_value=0),
        patch.object(process_module, "reap_session") as reap,
        patch.object(process_module, "reap_stale_accounts") as reap_accounts,
        patch.object(process_module, "_clear_mount_root") as reset,
    ):
        ProcessSessionBackend(make_live_worker_config(tmp_path))

    reap.assert_not_called()
    reap_accounts.assert_not_called()
    reset.assert_not_called()


def test_a_worker_that_is_not_root_reaps_nothing(tmp_path: Path) -> None:
    backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    with (
        patch.object(process_module.os, "getuid", return_value=1000),
        patch.object(process_module, "reap_session") as reap,
        patch.object(process_module, "reap_stale_accounts") as reap_accounts,
        patch.object(process_module, "_clear_mount_root") as reset,
    ):
        backend.reap_stale()

    reap.assert_not_called()
    reap_accounts.assert_not_called()
    reset.assert_not_called()


def test_the_executor_reaps_what_a_dead_worker_left_when_it_starts(
    tmp_path: Path,
) -> None:
    with (
        patch.object(ProcessSessionBackend, "is_available", return_value=True),
        patch.object(ProcessSessionBackend, "reap_stale") as reap,
    ):
        SSHExecutor(_config(SSHBackendName.PROCESS, tmp_path))

    reap.assert_called_once_with()


# ------------------------------------------------------------------ #
# Credentials in what a failing session reports
# ------------------------------------------------------------------ #


def test_a_session_failure_reports_no_restored_credential(tmp_path: Path) -> None:
    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    lifecycle.content_plane = FakeContentPlane(MagicMock())
    msg = make_worker_task_message(
        {
            "taskType": "ssh",
            "authorizedKeys": ["ssh-ed25519 AAAA key"],
            "env": {"TOKEN": _SECRET},
        },
        task_type=TaskType.SSH,
        task_id="tsk-1",
        credential_pointers={"tsk-1": ["/env/TOKEN"]},
    )
    backend = MagicMock()
    backend.start_session.side_effect = ExecutionError(
        f"sshd exited immediately.\nsshd output:\nenvironment TOKEN={_SECRET}"
    )
    with (
        patch.object(ProcessSessionBackend, "is_available", return_value=True),
        patch.object(ProcessSessionBackend, "reap_stale"),
    ):
        executor = SSHExecutor(
            _config(SSHBackendName.PROCESS, tmp_path), lifecycle=None
        )
    executor._backend = backend

    Runner(
        lifecycle=lifecycle,
        task_stream=[msg],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"ssh": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    ).start()

    reported = lifecycle.set_failed.call_args.args[1]
    assert _SECRET not in reported and "[REDACTED]" in reported
