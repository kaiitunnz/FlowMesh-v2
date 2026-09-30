"""The process session backend: which worker serves it, what a session is handed,
and what it leaves behind. Nothing here starts sshd or touches an account; the
in-container suite does that."""

import dataclasses
import io
import os
import tarfile
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
from worker.executors.ssh_session.base import extract_output_archive, tree_size_bytes
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


def test_a_worker_whose_state_takes_no_acl_serves_no_process_session(
    tmp_path: Path,
) -> None:
    with (
        patch.object(process_module, "process_identity_available", return_value=True),
        patch.object(process_module, "find_sshd", return_value="/usr/sbin/sshd"),
        patch.object(process_module, "find_ssh_keygen", return_value="/usr/bin/k"),
        patch.object(process_module, "ensure_state_roots"),
        patch.object(process_module, "supports_denials", return_value=False),
    ):
        assert not ProcessSessionBackend.is_available(
            make_live_worker_config(tmp_path, ssh_relay_host="10.0.0.9")
        )


def test_a_root_worker_its_supervisor_cannot_reach_serves_no_process_session(
    tmp_path: Path,
) -> None:
    with (
        patch.object(process_module, "process_identity_available", return_value=True),
        patch.object(process_module, "find_sshd", return_value="/usr/sbin/sshd"),
        patch.object(process_module, "find_ssh_keygen", return_value="/usr/bin/k"),
        patch.object(process_module, "find_tar", return_value="/usr/bin/tar"),
        patch.object(process_module, "ensure_state_roots"),
        patch.object(process_module, "supports_denials", return_value=True),
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


def _path_fields(cls: type) -> set[str]:
    hints = typing.get_type_hints(cls)
    return {
        f.name
        for f in dataclasses.fields(cls)
        if Path in {hints[f.name], *typing.get_args(hints[f.name])}
    }


def test_every_worker_path_field_is_classified_as_state_or_not() -> None:
    classified = WorkerConfig.STATE_ROOT_FIELDS | WorkerConfig.NON_STATE_PATH_FIELDS

    assert _path_fields(WorkerConfig) == classified
    assert not WorkerConfig.STATE_ROOT_FIELDS & WorkerConfig.NON_STATE_PATH_FIELDS


def test_every_missing_state_root_is_created_private_to_the_worker(
    tmp_path: Path,
) -> None:
    config = make_worker_config(
        results_dir=tmp_path / "results",
        private_state_dir=tmp_path / "private",
        content_dir=tmp_path / "relocated" / "content",
        hb_file=tmp_path / "hb" / "worker.hb",
        home_dir=tmp_path / "home",
        model_cache_dir=tmp_path / "hf",
    )

    process_module.ensure_state_roots(config)

    for root in config.state_root_dirs:
        assert root.is_dir() and root.stat().st_mode & 0o777 == 0o700, root
    assert config.hb_file.is_file()


def test_the_state_roots_cover_the_worker_state_and_a_filesystem_store(
    tmp_path: Path,
) -> None:
    config = make_worker_config(
        results_dir=tmp_path / "results",
        private_state_dir=tmp_path / "private",
        content_dir=tmp_path / "content",
        hb_file=tmp_path / "worker.hb",
        home_dir=tmp_path / "home",
        model_cache_dir=tmp_path / "hf",
    )
    store = dataclasses.replace(
        config.object_store,
        backend=BACKEND_FILESYSTEM,
        filesystem_root=tmp_path / "store",
    )

    roots = set(dataclasses.replace(config, object_store=store).state_roots)

    assert {getattr(config, name) for name in WorkerConfig.STATE_ROOT_FIELDS} == (
        roots - {tmp_path / "store"}
    )
    assert roots == {
        tmp_path / name
        for name in ("results", "private", "content", "worker.hb", "home", "hf")
    } | {tmp_path / "store"}


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


def test_a_mount_path_through_a_symlink_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "planted").symlink_to(target)

    with pytest.raises(ExecutionError, match="not a directory"):
        process_module._make_dirs_nofollow(tmp_path / "planted" / "output")

    assert list(target.iterdir()) == []


def test_mount_components_are_created_one_level_at_a_time(tmp_path: Path) -> None:
    fd = process_module._make_dirs_nofollow(tmp_path / "a" / "b" / "c")
    os.close(fd)

    assert (tmp_path / "a" / "b" / "c").is_dir()


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

    assert tree_size_bytes(output) == len("ok") + len("nested")


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
        session_dir=session_dir,
        account="fmssn1",
        roots=[(tmp_path / "results").as_posix()],
        uid=1234,
        sshd_pid=4321,
    ).write()

    with (
        patch.object(process_module, "_is_our_sshd", return_value=ours),
        patch.object(process_module.os, "kill") as kill,
        patch.object(process_module, "retire_account", return_value=True) as retire,
    ):
        assert process_module.reap_session(session_dir)

    assert kill.call_count == (1 if ours else 0)
    retire.assert_called_once_with("fmssn1", 1234, [tmp_path / "results"])
    assert not session_dir.exists()


def test_a_reap_keeps_a_session_whose_account_still_runs_a_process(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "ssn-1"
    session_dir.mkdir()
    process_module.SessionManifest(
        session_dir=session_dir, account="fmssn1", roots=[], uid=1234
    ).write()

    with patch.object(process_module, "retire_account", return_value=False):
        assert not process_module.reap_session(session_dir)

    assert session_dir.exists()


def test_a_worker_serves_no_session_until_a_stuck_account_is_reaped(
    tmp_path: Path,
) -> None:
    backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    with (
        patch.object(process_module.os, "getuid", return_value=0),
        patch.object(process_module, "SESSIONS_ROOT", tmp_path / "sessions"),
        patch.object(process_module, "_reset_mount_root"),
        patch.object(process_module, "reap_stale_accounts", return_value=False),
    ):
        backend.reap_stale()
    with pytest.raises(ExecutionError, match="serves no session") as refused:
        backend.start_session(_request(tmp_path, [], None))
    assert refused.value.retryable

    with (
        patch.object(process_module.os, "getuid", return_value=0),
        patch.object(process_module, "SESSIONS_ROOT", tmp_path / "sessions"),
        patch.object(process_module, "_reset_mount_root"),
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
        patch.object(process_module, "_reset_mount_root") as reset,
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
        patch.object(process_module, "_reset_mount_root") as reset,
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
