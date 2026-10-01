"""The process session backend: which worker serves it, what a session is handed,
and what it leaves behind. Nothing here starts sshd or touches an account; the
in-container suite does that."""

import dataclasses
import fcntl
import io
import logging
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import psutil
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
    no_mediated_op,
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
    render_authorized_keys,
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


@pytest.fixture(autouse=True)
def _no_account_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_module, "lock_account", MagicMock())


def _servable(**patches: Any) -> Any:
    defaults: dict[str, Any] = {
        "process_identity_available": True,
        "find_sshd": "/usr/sbin/sshd",
        "find_ssh_keygen": "/usr/bin/ssh-keygen",
        "find_tar": "/usr/bin/tar",
        "find_tini": "/usr/bin/tini",
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
    config = make_live_worker_config(tmp_path)
    with _servable():
        assert ProcessSessionBackend.is_available(config)
    with _servable(**{unready: False}):
        assert not ProcessSessionBackend.is_available(config)


@pytest.mark.parametrize(
    "missing", ["find_sshd", "find_ssh_keygen", "find_tar", "find_tini"]
)
def test_a_worker_missing_a_tool_serves_no_process_session(
    tmp_path: Path, missing: str
) -> None:
    config = make_live_worker_config(tmp_path)
    with _servable(**{missing: None}):
        assert not ProcessSessionBackend.is_available(config)


def test_a_worker_without_the_acl_tools_cannot_isolate_a_session(
    tmp_path: Path,
) -> None:
    with patch.object(process_module.acl, "tools_available", return_value=False):
        assert not process_module._acl_ready(make_live_worker_config(tmp_path))


def test_the_filesystem_holding_the_session_state_is_probed_for_acls(
    tmp_path: Path,
) -> None:
    state = tmp_path / "flowmesh"
    state.mkdir()
    with (
        patch.object(process_module.acl, "tools_available", return_value=True),
        patch.object(process_module.acl, "STATE_DIR", state),
        patch.object(
            process_module.acl,
            "probe",
            side_effect=lambda path: _refuse_probe(path, state),
        ),
    ):
        assert not process_module._acl_ready(_state_config(tmp_path))


def _refuse_probe(path: Path, refused: Path) -> None:
    if path == refused:
        raise ExecutionError("ACL entries do not persist")


def test_a_state_dir_someone_else_owns_is_refused(tmp_path: Path) -> None:
    if os.getuid() == 0:
        pytest.skip("root owns what it creates")
    (tmp_path / "state").mkdir()

    with pytest.raises(ExecutionError, match="not a root-owned directory"):
        process_module._make_private_dir(tmp_path / "state", 0o711)


def test_a_root_worker_with_no_routable_address_serves_process_sessions(
    tmp_path: Path,
) -> None:
    """The worker reaches a relayed session itself, so nothing needs to dial it."""
    with (
        _servable(),
        patch.object(process_module, "resolve_tailnet_address", return_value=None),
    ):
        assert ProcessSessionBackend.is_available(make_live_worker_config(tmp_path))


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


def _with_store(config: WorkerConfig, root: Path) -> WorkerConfig:
    store = dataclasses.replace(
        config.object_store, backend=BACKEND_FILESYSTEM, filesystem_root=root
    )
    return dataclasses.replace(config, object_store=store)


def test_a_shared_store_root_is_never_created_and_refuses_sessions_until_it_exists(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    config = _with_store(_state_config(tmp_path), store)

    with (
        patch.object(process_module.acl, "tools_available", return_value=True),
        patch.object(process_module.acl, "probe") as probe,
        patch.object(process_module.acl, "STATE_DIR", tmp_path / "flowmesh"),
    ):
        assert process_module._acl_ready(config)
    assert not store.exists()
    probed = [call.args[0] for call in probe.call_args_list]
    assert tmp_path in probed and tmp_path / "results" in probed
    with pytest.raises(ExecutionError, match="does not exist yet") as refused:
        process_module.ensure_state_roots(config)
    assert refused.value.retryable
    assert not store.exists()

    store.mkdir()
    assert store in process_module.ensure_state_roots(config)


def test_a_state_root_below_a_missing_store_root_never_creates_the_store(
    tmp_path: Path,
) -> None:
    """The store is the content plane's to create; a local stand-in made as some
    nested root's parent would pass for the shared mount."""
    store = tmp_path / "store"
    config = _with_store(_state_config(tmp_path, content_dir=store / "cache"), store)

    with (
        patch.object(process_module.acl, "tools_available", return_value=True),
        patch.object(process_module.acl, "probe"),
        patch.object(process_module.acl, "STATE_DIR", tmp_path / "flowmesh"),
    ):
        assert process_module._acl_ready(config)
    assert not store.exists()
    with pytest.raises(ExecutionError, match="does not exist yet") as refused:
        process_module.ensure_state_roots(config)
    assert refused.value.retryable
    assert not store.exists()

    store.mkdir()
    roots = process_module.ensure_state_roots(config)
    assert store in roots and store / "cache" not in roots
    assert not (store / "cache").exists()


def test_a_missing_store_root_inside_the_results_dir_is_left_out(
    tmp_path: Path,
) -> None:
    config = _state_config(tmp_path)
    (tmp_path / "results").mkdir()
    store = tmp_path / "results" / "shared-content"
    config = _with_store(config, store)

    roots = process_module.ensure_state_roots(config)

    assert not store.exists()
    assert store not in roots and tmp_path / "results" in roots


def test_a_missing_cache_inside_an_existing_cache_root_is_left_out(
    tmp_path: Path,
) -> None:
    hub = tmp_path / "hf" / "hub"
    (tmp_path / "hf").mkdir()
    config = _state_config(
        tmp_path, state_dirs=(tmp_path / "home", tmp_path / "hf", hub)
    )

    roots = process_module.ensure_state_roots(config)

    assert not hub.exists()
    assert hub not in roots and tmp_path / "hf" in roots


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

    assert data in process_module.denied_roots(_state_config(tmp_path))
    assert process_module._path_problem(tmp_path / "results") is None


def _refusal(config: WorkerConfig) -> str:
    with pytest.raises(ExecutionError) as refused:
        process_module.ensure_state_roots(config)
    assert refused.value.retryable
    return str(refused.value)


@pytest.mark.parametrize("mode", [0o1777, 0o777])
def test_a_link_in_a_shared_dir_is_refused_not_followed(
    tmp_path: Path, mode: int
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (shared / "cache").symlink_to(victim)
    shared.chmod(mode)

    refusal = _refusal(_state_config(tmp_path, state_dirs=(shared / "cache",)))

    assert ("world-writable" if mode == 0o777 else "shared directory") in refusal
    assert list(victim.iterdir()) == []


def test_an_owned_root_in_a_sticky_dir_is_accepted(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    (shared / "cache").mkdir(parents=True)
    shared.chmod(0o1777)

    assert process_module._path_problem(shared / "cache") is None
    assert process_module._path_problem(shared / "missing") is None


@pytest.mark.parametrize("depth", [1, 2])
def test_a_root_anywhere_below_a_world_writable_dir_is_refused(
    tmp_path: Path, depth: int
) -> None:
    shared = tmp_path / "open"
    root = shared.joinpath(*["sub"] * (depth - 1), "results")
    root.mkdir(parents=True)
    shared.chmod(0o777)

    assert "world-writable" in _refusal(_state_config(tmp_path, results_dir=root))


def test_a_missing_part_below_a_world_writable_dir_is_refused_not_created(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "open"
    shared.mkdir()
    shared.chmod(0o777)

    assert "world-writable" in _refusal(
        _state_config(tmp_path, results_dir=shared / "missing" / "results")
    )
    assert not (shared / "missing").exists()


def test_a_world_writable_dir_a_link_leads_through_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    (shared / "results").mkdir(parents=True)
    shared.chmod(0o777)
    (tmp_path / "hop").symlink_to("shared/results")
    (tmp_path / "results").symlink_to(tmp_path / "hop")

    assert "world-writable" in _refusal(_state_config(tmp_path))


def test_parent_components_after_a_link_are_resolved_physically(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    (shared / "inner").mkdir(parents=True)
    (shared / "results").mkdir()
    shared.chmod(0o777)
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe" / "link").symlink_to(shared / "inner")

    assert "world-writable" in (
        process_module._path_problem(tmp_path / "safe" / "link" / ".." / "results")
        or ""
    )


def test_a_link_loop_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")

    assert "too many links" in (process_module._path_problem(tmp_path / "a") or "")


def _shared_cache(tmp_path: Path) -> Path:
    """A model cache as deployments share it: a host directory open to every uid,
    with the hub below it open too."""
    cache = tmp_path / "huggingface"
    (cache / "hub").mkdir(parents=True)
    for path in (cache, cache / "hub"):
        path.chmod(0o777)
    return cache


def test_a_world_writable_dir_inside_a_denied_root_is_accepted(
    tmp_path: Path,
) -> None:
    cache = _shared_cache(tmp_path)

    roots = process_module.ensure_state_roots(
        _state_config(tmp_path, state_dirs=(cache, cache / "hub"))
    )

    assert {cache, cache / "hub"} <= set(roots)


def test_a_cache_on_the_default_volume_is_accepted(tmp_path: Path) -> None:
    # The supervisor's named volume: a worker-owned 0755 root, its hub opened by
    # the entrypoint's prefetch.
    cache = tmp_path / "huggingface"
    (cache / "hub").mkdir(parents=True)
    cache.chmod(0o755)
    (cache / "hub").chmod(0o777)

    for state_dirs in ((cache, cache / "hub"), (cache / "hub",)):
        roots = process_module.ensure_state_roots(
            _state_config(tmp_path, state_dirs=state_dirs)
        )
        assert set(state_dirs) <= set(roots)


def test_the_same_world_writable_dir_outside_any_denied_root_is_refused(
    tmp_path: Path,
) -> None:
    cache = _shared_cache(tmp_path)

    assert "world-writable" in _refusal(
        _state_config(tmp_path, state_dirs=(cache / "hub",))
    )


def test_a_link_out_of_a_denied_root_to_a_world_writable_dir_is_refused(
    tmp_path: Path,
) -> None:
    cache = _shared_cache(tmp_path)
    outside = tmp_path / "outside"
    (outside / "models").mkdir(parents=True)
    outside.chmod(0o777)
    (cache / "models").symlink_to(outside / "models")

    assert "world-writable" in _refusal(
        _state_config(tmp_path, state_dirs=(cache, cache / "models"))
    )


def test_a_root_is_created_without_following_a_link_planted_on_the_way(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "new").symlink_to(elsewhere)

    with pytest.raises(OSError):
        process_module._create_state_root(tmp_path / "new" / "state")
    assert list(elsewhere.iterdir()) == []


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


def _lock_held_elsewhere(lock: Path) -> bool:
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
    return other.returncode != 0


@pytest.fixture
def lock_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(process_module, "_backend_lock_fds", None)
    yield
    for fd in process_module._backend_lock_fds or ():
        os.close(fd)


def test_one_worker_per_lock_file_holds_the_process_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_state: None
) -> None:
    monkeypatch.setattr(process_module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(process_module.tempfile, "tempdir", tmp_path.as_posix())

    assert process_module._acquire_backend_lock()
    assert process_module._acquire_backend_lock()
    assert _lock_held_elsewhere(tmp_path / process_module._BACKEND_LOCK_NAME)


def test_a_worker_sharing_the_state_dir_with_another_serves_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_state: None
) -> None:
    run, state = tmp_path / "run", tmp_path / "state"
    run.mkdir()
    state.mkdir()
    monkeypatch.setattr(process_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(process_module, "_RUN_DIR", run)
    monkeypatch.setattr(process_module.acl, "STATE_DIR", state)
    monkeypatch.setattr(process_module, "_make_private_dir", lambda path, mode: None)
    monkeypatch.setattr(process_module, "_root_only_dir", lambda path: True)
    peer = os.open(
        state / process_module._BACKEND_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600
    )
    fcntl.flock(peer, fcntl.LOCK_EX)
    try:
        assert not process_module._acquire_backend_lock()
        assert not _lock_held_elsewhere(run / process_module._BACKEND_LOCK_NAME)
    finally:
        os.close(peer)

    assert process_module._acquire_backend_lock()
    assert _lock_held_elsewhere(state / process_module._BACKEND_LOCK_NAME)


def test_a_run_dir_another_account_can_write_takes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_state: None
) -> None:
    monkeypatch.setattr(process_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(process_module, "_RUN_DIR", Path(tempfile.gettempdir()))
    monkeypatch.setattr(process_module.acl, "STATE_DIR", tmp_path)
    monkeypatch.setattr(process_module, "_make_private_dir", lambda path, mode: None)

    assert not process_module._acquire_backend_lock()
    assert process_module._root_only_dir(Path("/"))
    assert not process_module._root_only_dir(tmp_path)


def test_the_lock_is_taken_before_any_state_root_is_touched(tmp_path: Path) -> None:
    config = make_live_worker_config(tmp_path)
    with (
        _servable(_acquire_backend_lock=False),
        patch.object(process_module, "_release_backend_lock") as release,
    ):
        assert not ProcessSessionBackend.is_available(config)
        cast(MagicMock, process_module._acl_ready).assert_not_called()
    release.assert_not_called()
    with (
        _servable(_acl_ready=False),
        patch.object(process_module, "_release_backend_lock") as release,
    ):
        assert not ProcessSessionBackend.is_available(config)
    release.assert_called_once_with()


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


@pytest.mark.skipif(os.getuid() == 0, reason="root opens any directory")
def test_an_output_dir_the_walk_cannot_open_fails_the_size_check(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "output" / "sub" / "locked"
    locked.mkdir(parents=True)
    (locked / "f").write_bytes(b"x" * 1024)
    locked.chmod(0)
    try:
        with pytest.raises(ExecutionError, match="Permission denied: 'sub/locked'"):
            path_size_bytes(tmp_path / "output")
    finally:
        locked.chmod(0o700)


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
    rendered, exported = render_authorized_keys(
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
        bind_host="127.0.0.1",
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


# Writes a tar of two files, or dies of a signal after the first one, which cuts
# the stream on a header boundary, as a reader the collector never killed would.
_ARCHIVER = """
import io, os, signal, sys, tarfile
def member(name):
    data = name.encode() * 1000
    info = tarfile.TarInfo(name)
    info.size = len(data)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()[: 512 + -(-len(data) // 512) * 512]
out = sys.stdout.buffer
out.write(member("first"))
out.flush()
if sys.argv[1] == "die":
    os.kill(os.getpid(), signal.SIGKILL)
out.write(member("second") + bytes(1024))
out.flush()
"""


def _archiver(outcome: str) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(  # nosec B603 - argv list, test-only
        [sys.executable, "-c", _ARCHIVER, outcome], stdout=subprocess.PIPE
    )


@pytest.mark.parametrize("outcome", ["exit", "die"])
def test_an_archiver_killed_by_what_the_collector_never_sent_fails_the_task(
    tmp_path: Path, outcome: str
) -> None:
    session = _session(RunSignals(), MagicMock())
    session._output_path = tmp_path / "output"
    collected = tmp_path / "collected"
    with (
        patch.object(process_module, "kill_processes", return_value=True),
        patch.object(process_module, "_archive_as", return_value=_archiver(outcome)),
    ):
        if outcome == "die":
            with pytest.raises(ExecutionError, match="killed by signal 9") as failed:
                session.collect_output(collected, None)
            assert not failed.value.retryable
        else:
            session.collect_output(collected, None)
            assert sorted(p.name for p in collected.iterdir()) == ["first", "second"]


def test_a_stop_during_collection_leaves_the_reader_running(tmp_path: Path) -> None:
    session = _session(RunSignals(), MagicMock())
    session._output_path = tmp_path / "output"
    kills: list[str] = []

    def kill(uid: int) -> bool:
        kills.append("kill")
        return True

    def extract(*args: Any) -> None:
        kills.clear()
        session.stop(1)

    with (
        patch.object(process_module, "kill_processes", side_effect=kill),
        patch.object(process_module, "_end_sshd"),
        patch.object(process_module, "_archive_as", return_value=_archiver("exit")),
        patch.object(process_module, "extract_output_archive", side_effect=extract),
    ):
        session.collect_output(tmp_path / "collected", None)
        assert kills == []
        session.stop(1)

    assert kills == ["kill", "kill"]


def test_a_cancel_during_collection_ends_it_cancelled(tmp_path: Path) -> None:
    signals = RunSignals()
    session = _session(signals, MagicMock())
    session._output_path = tmp_path / "output"
    with (
        signals.running("tsk"),
        patch.object(process_module, "kill_processes", return_value=True),
        patch.object(process_module, "_end_sshd"),
        patch.object(process_module, "_archive_as", return_value=_archiver("exit")),
        patch.object(
            process_module,
            "extract_output_archive",
            side_effect=lambda *args: signals.cancel("tsk") and args[3](),
        ),
        pytest.raises(TaskCancelledError),
    ):
        session.collect_output(tmp_path / "collected", None)


# Writes a header promising more than it sends, then stops itself, as a reader a
# process of the session stopped would.
_STALLED_ARCHIVER = """
import os, signal, sys, tarfile
info = tarfile.TarInfo("big")
info.size = 1 << 20
sys.stdout.buffer.write(info.tobuf() + bytes(512))
sys.stdout.buffer.flush()
os.kill(os.getpid(), signal.SIGSTOP)
"""


def _collect_in_background(
    session: process_module.ProcessSession, destination: Path
) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def collect() -> None:
        try:
            session.collect_output(destination, None)
        except BaseException as exc:
            errors.append(exc)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    return collector, errors


def _stalled_archiver() -> "subprocess.Popen[bytes]":
    archiver = subprocess.Popen(  # nosec B603 - argv list, test-only
        [sys.executable, "-c", _STALLED_ARCHIVER], stdout=subprocess.PIPE
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if psutil.Process(archiver.pid).status() == psutil.STATUS_STOPPED:
            break
        time.sleep(0.01)
    return archiver


def test_a_cancel_ends_a_collection_whose_reader_is_stuck(tmp_path: Path) -> None:
    signals = RunSignals()
    session = _session(signals, MagicMock())
    session._output_path = tmp_path / "output"
    archiver = _stalled_archiver()
    try:
        with (
            signals.running("tsk"),
            patch.object(process_module, "kill_processes", return_value=True),
            patch.object(process_module, "_end_sshd"),
            patch.object(process_module, "_archive_as", return_value=archiver),
        ):
            collector, errors = _collect_in_background(session, tmp_path / "c")
            time.sleep(0.2)
            signals.cancel("tsk")
            session.stop(1)
            collector.join(timeout=10)
            assert not collector.is_alive()
        assert [type(error) for error in errors] == [TaskCancelledError]
    finally:
        archiver.kill()
        archiver.wait()


def test_a_reader_that_writes_nothing_for_too_long_fails_the_collection_for_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_module, "_ARCHIVE_IDLE_TIMEOUT_SEC", 0.5)
    session = _session(RunSignals(), MagicMock())
    session._output_path = tmp_path / "output"
    archiver = _stalled_archiver()
    try:
        with (
            patch.object(process_module, "kill_processes", return_value=True),
            patch.object(process_module, "_end_sshd"),
            patch.object(process_module, "_archive_as", return_value=archiver),
        ):
            collector, errors = _collect_in_background(session, tmp_path / "c")
            collector.join(timeout=10)
            assert not collector.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], ExecutionError)
        assert "no progress" in str(errors[0])
        assert not errors[0].retryable
        assert archiver.poll() == -9
    finally:
        archiver.kill()
        archiver.wait()


def test_collection_bars_logins_before_it_ends_sshd_and_the_session(
    tmp_path: Path,
) -> None:
    session = _session(RunSignals(), MagicMock())
    session._session_dir = tmp_path
    session._output_path = tmp_path / "output"
    keys = tmp_path / process_module._AUTHORIZED_KEYS_NAME
    keys.write_text("key\n")
    order: list[Any] = []

    def kill(uid: int) -> bool:
        order.append("kill")
        return True

    with (
        patch.object(
            process_module,
            "lock_account",
            side_effect=lambda name: order.append(("lock", keys.exists())),
        ),
        patch.object(
            process_module, "_end_sshd", side_effect=lambda _: order.append("end")
        ),
        patch.object(process_module, "kill_processes", side_effect=kill),
        patch.object(process_module, "_archive_as", return_value=_archiver("exit")),
    ):
        session.collect_output(tmp_path / "collected", None)

    assert order == [("lock", False), "end", "kill"]


def test_the_session_is_stopped_before_its_sshd_and_again_after() -> None:
    order: list[str] = []
    session = _session(RunSignals(), MagicMock())

    def kill(uid: int) -> bool:
        order.append("kill")
        return True

    with (
        patch.object(process_module, "kill_processes", side_effect=kill),
        patch.object(
            process_module, "_end_sshd", side_effect=lambda _: order.append("end")
        ),
    ):
        session.stop(1)

    assert order == ["kill", "end", "kill"]


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


def test_a_session_s_files_get_their_modes_whatever_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        bin_dir = process_module._install_finish_helper(tmp_path, tmp_path / "finish")
        process_module._write_private(tmp_path / "authorized_keys", "key\n")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(bin_dir.stat().st_mode) == 0o711
    assert stat.S_IMODE((tmp_path / "authorized_keys").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        process_module._write_private(tmp_path / "authorized_keys", "other\n")


def test_ending_sshd_kills_its_whole_tree() -> None:
    process = MagicMock(pid=4321)
    process.poll.return_value = None
    root = MagicMock()
    with (
        patch.object(process_module.psutil, "Process", return_value=root) as find,
        patch.object(process_module, "_kill_tree") as kill_tree,
    ):
        process_module._end_sshd(process)

    find.assert_called_once_with(4321)
    kill_tree.assert_called_once_with(root)
    process.wait.assert_called_once()


def test_an_ended_sshd_is_left_alone() -> None:
    process = MagicMock()
    process.poll.return_value = 0
    with patch.object(process_module, "_kill_tree") as kill_tree:
        process_module._end_sshd(process)

    kill_tree.assert_not_called()


def test_sshd_runs_niced_and_under_tini_as_a_subreaper() -> None:
    argv = process_module._launch_argv("/usr/bin/tini", ["/usr/sbin/sshd", "-D"])

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


# ------------------------------------------------------------------ #
# Reaping what a dead worker left
# ------------------------------------------------------------------ #


def _proc_running(argv: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.cmdline.return_value = argv
    return proc


def test_a_reap_kills_only_the_sshd_run_with_the_session_s_config(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "ssn-1"
    session_dir.mkdir()
    process_module.SessionManifest(session_dir=session_dir, account="fmssn1").write()
    config = (session_dir / "sshd_config").as_posix()
    commands = [
        ["/usr/bin/tini", "-s", "--", "/usr/sbin/sshd", "-f", config],
        ["sshd:", "/usr/sbin/sshd", "-D", "-e", "-f", config],
        ["/usr/sbin/sshd", "-f", "/other/sshd_config"],
        ["python3", config],
        ["/usr/sbin/sshd", "-f", config + ".bak"],
    ]
    procs = [_proc_running(argv) for argv in commands]
    gone = MagicMock()
    gone.cmdline.side_effect = psutil.NoSuchProcess(9)

    with (
        patch.object(
            process_module.psutil, "process_iter", return_value=[*procs, gone]
        ),
        patch.object(process_module, "_kill_tree") as kill,
        patch.object(
            process_module.pwd, "getpwnam", return_value=MagicMock(pw_uid=61_001)
        ),
        patch.object(process_module, "retire_account", return_value=True) as retire,
    ):
        assert process_module.reap_session(session_dir)

    assert [call.args for call in kill.call_args_list] == [(procs[0],), (procs[1],)]
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


def test_a_tree_is_frozen_then_killed_from_the_bottom_up() -> None:
    order: list[str] = []
    child = MagicMock()
    child.status.return_value = psutil.STATUS_RUNNING
    child.kill.side_effect = lambda: order.append("child")
    root = MagicMock()
    root.children.side_effect = [[child], []]
    root.suspend.side_effect = lambda: order.append("suspend")
    root.kill.side_effect = lambda: order.append("root")
    with patch.object(process_module.time, "sleep"):
        process_module._kill_tree(root)

    assert order == ["suspend", "child", "root"]


def test_a_process_that_outlives_the_kill_deadline_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stuck = MagicMock(pid=4242)
    stuck.status.return_value = psutil.STATUS_RUNNING
    stuck.name.return_value = "sshd-session"
    root = MagicMock(pid=4200)
    root.children.return_value = [stuck]
    clock = iter(range(0, 1000, 1))
    with (
        patch.object(process_module.time, "sleep"),
        patch.object(process_module.time, "monotonic", side_effect=lambda: next(clock)),
        caplog.at_level(logging.WARNING, logger=process_module.logger.name),
    ):
        process_module._kill_tree(root)

    assert "4242 (sshd-session)" in caplog.text
    root.kill.assert_called_once()


def test_a_tree_s_processes_all_die(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    parent = subprocess.Popen(  # nosec B603 - argv list, test-only
        [
            sys.executable,
            "-c",
            "import pathlib, subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "pathlib.Path(sys.argv[1]).touch()\n"
            "time.sleep(60)\n",
            ready.as_posix(),
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        root = psutil.Process(parent.pid)
        (child,) = root.children()

        process_module._kill_tree(root)

        assert parent.wait(timeout=5) == -9
        psutil.wait_procs([child], timeout=5)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        parent.kill()
        parent.wait()


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

    assert not process_module._is_our_sshd(psutil.Process(), config)


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
    lifecycle.client.next_mediated_op.side_effect = no_mediated_op
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


@pytest.mark.parametrize(
    ("mode", "bind", "scope"),
    [
        ("direct", "0.0.0.0", "network"),
        ("proxy", "127.0.0.1", "loopback"),
        ("forward", "127.0.0.1", "loopback"),
    ],
)
def test_only_a_direct_session_listens_beyond_loopback(
    tmp_path: Path, mode: str, bind: str, scope: str
) -> None:
    backend = ProcessSessionBackend(make_live_worker_config(tmp_path))
    assert backend.session_bind_host(mode) == bind
    assert backend.session_scope(mode) == scope
    config = process_module._render_sshd_config(
        port=2222,
        session_dir=Path("/run/s"),
        host_key=Path("/run/s/key"),
        authorized_keys=Path("/run/s/authorized_keys"),
        login_user="fmssnabc",
        exported_env=[],
        bind_host=backend.session_bind_host(mode),
    )
    assert f"ListenAddress {bind}" in config.splitlines()
    if scope == "loopback":
        assert backend.session_address(mode) == "127.0.0.1"


@pytest.mark.parametrize(
    ("title", "serves"),
    [
        (["sshd-session: fmssn61000 [priv]"], True),
        (["sshd-session: fmssn61000@pts/0"], True),
        (["sshd: fmssn61000@notty"], True),
        (["sshd-session:", "fmssn61000", "[priv]"], True),
        (["sshd-session: fmssn610001 [priv]"], False),
        (["sshd-session: fmssn6100 [priv]"], False),
        (["bash", "fmssn61000"], False),
        (["sshd: /usr/sbin/sshd -D -f /run/s/sshd_config"], False),
    ],
)
def test_a_connection_process_is_matched_by_its_account_as_a_whole_word(
    title: list[str], serves: bool
) -> None:
    proc = MagicMock()
    proc.cmdline.return_value = title
    assert process_module._serves_account(proc, "fmssn61000") is serves


def test_stopping_a_session_whose_listener_is_gone_kills_its_connections() -> None:
    """A privileged connection process outlives a listener that exited on its own."""
    monitor = MagicMock()
    monitor.cmdline.return_value = ["sshd-session: fmssn61000 [priv]"]
    other = MagicMock()
    other.cmdline.return_value = ["sshd-session: fmssn61001 [priv]"]
    process = MagicMock()
    process.poll.return_value = 0
    session = _session(RunSignals(), process)
    session.account.name = "fmssn61000"
    with (
        patch.object(process_module, "kill_processes"),
        patch.object(
            process_module.psutil, "process_iter", return_value=[monitor, other]
        ),
        patch.object(process_module, "_kill_tree") as kill_tree,
    ):
        session.stop(1)

    kill_tree.assert_called_once_with(monitor)
