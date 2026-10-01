"""An interactive SSH session is stopped once it has had no connection for its idle
timeout, and never on missing evidence."""

import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from shared.tasks.specs import SSHSpecStrict
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import (
    DEFAULT_WORKER_CONFIG,
    make_live_worker_config,
    make_ssh_executor,
)
from worker.executors import ssh_executor as ssh_module
from worker.executors.base_executor import ExecutionError
from worker.executors.ssh_session import (
    SessionRequest,
    SSHConfig,
    SSHSession,
    SSHSessionBackend,
)
from worker.executors.ssh_session.backends import process as process_module
from worker.executors.ssh_session.backends.docker import DockerSession
from worker.executors.ssh_session.base import count_established_connections

_TASK_ID = "tsk-idle"


class _FakeSession(SSHSession):
    """A session whose observable state the test drives."""

    def __init__(self, connections: list[int | None], output_size: int | None) -> None:
        self._connections = connections
        self._output_size = output_size
        self.stops: list[float] = []

    def login_user(self) -> str:
        return "flowmesh"

    def wait_ready(self, timeout_sec: float) -> int:
        return 2222

    def poll(self) -> int | None:
        return None

    def finish_requested(self) -> bool:
        return False

    def established_connections(self) -> int | None:
        if len(self._connections) > 1:
            return self._connections.pop(0)
        return self._connections[0]

    def output_size_bytes(self) -> int | None:
        return self._output_size

    def collect_output(self, destination: Path, max_bytes: int | None) -> None:
        return None

    def stop(self, timeout_sec: float) -> None:
        self.stops.append(timeout_sec)

    def cleanup(self) -> None:
        return None


class _FakeBackend(SSHSessionBackend):
    def __init__(self, session: _FakeSession) -> None:
        super().__init__(DEFAULT_WORKER_CONFIG)
        self.session = session

    @classmethod
    def is_available(cls, config: Any) -> bool:
        return True

    def prepare(self) -> None:
        return None

    def start_session(self, request: SessionRequest) -> SSHSession:
        return self.session

    def teardown(self, worker_name: str) -> None:
        return None


def _task(**spec: Any) -> WorkerTaskMessage:
    return WorkerTaskMessage.model_validate(
        {
            "task_id": _TASK_ID,
            "workflow_id": "wfl-1",
            "owner_id": "owner",
            "assigned_worker": "worker-1",
            "dispatched_at": "2026-03-22T00:00:00Z",
            "task": {
                "apiVersion": "mloc/v1",
                "kind": "Task",
                "metadata": {"name": "wf:s"},
                "spec": {
                    "taskType": "ssh",
                    "authorizedKeys": ["ssh-ed25519 AAAA test"],
                    **spec,
                },
            },
        }
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    connections: list[int | None],
    output_size: int | None = None,
    **spec: Any,
) -> tuple[_FakeSession, float]:
    monkeypatch.setenv("SSH_POLL_INTERVAL_SEC", "0.01")
    executor = make_ssh_executor(make_live_worker_config(tmp_path), lifecycle=None)
    session = _FakeSession(connections, output_size)
    executor._backend = _FakeBackend(session)
    started = time.monotonic()
    with (
        patch.object(executor, "emit_update"),
        patch.object(ssh_module, "maybe_upload_artifacts"),
    ):
        executor.run(_task(**spec), tmp_path / "out")
    return session, time.monotonic() - started


def test_a_session_nobody_connects_to_is_stopped_at_its_idle_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, elapsed = _run(
        tmp_path, monkeypatch, [0], ttlSeconds=30, idleTimeoutSeconds=0.2
    )

    assert elapsed < 5.0
    assert session.stops[0] == 1


def test_the_idle_clock_restarts_while_a_connection_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected = [1] * 40

    session, elapsed = _run(
        tmp_path, monkeypatch, [*connected, 0], ttlSeconds=30, idleTimeoutSeconds=0.2
    )

    assert 0.4 <= elapsed < 5.0
    assert session.stops[0] == 1


@pytest.mark.parametrize("connections", [None, 1])
def test_a_session_is_not_reaped_without_evidence_it_is_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, connections: int | None
) -> None:
    _, elapsed = _run(
        tmp_path, monkeypatch, [connections], ttlSeconds=1, idleTimeoutSeconds=0.05
    )

    # Only its TTL ends it.
    assert elapsed >= 1.0


def test_a_zero_idle_timeout_leaves_only_the_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, elapsed = _run(tmp_path, monkeypatch, [0], ttlSeconds=1, idleTimeoutSeconds=0)

    assert elapsed >= 1.0


def test_an_output_limit_breach_fails_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExecutionError, match="exceeded maxBytes"):
        _run(
            tmp_path,
            monkeypatch,
            [1],
            output_size=11,
            ttlSeconds=30,
            sshOutput={"maxBytes": 10},
        )


def _cfg(**spec: Any) -> SSHConfig:
    payload = {"taskType": "ssh", "authorizedKeys": ["ssh-ed25519 AAAA"], **spec}
    return SSHConfig.from_spec(
        cast(SSHSpecStrict, SSHSpecStrict.model_validate(payload)),
        DEFAULT_WORKER_CONFIG,
    )


def test_the_idle_timeout_never_exceeds_the_ttl() -> None:
    assert _cfg(ttlSeconds=60, idleTimeoutSeconds=3600).idle_sec == 60
    assert _cfg(ttlSeconds=120).idle_sec == 120
    assert _cfg(ttlSeconds=3600, idleTimeoutSeconds=120).idle_sec == 120
    assert _cfg(ttlSeconds=3600, idleTimeoutSeconds=0).idle_sec == 0


_PROC_NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid
   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0
   1: 0100007F:0016 0100007F:D3A2 01 00000000:00000000 00:00000000 00000000     0
   2: 0100007F:0017 0100007F:D3A4 01 00000000:00000000 00:00000000 00000000     0
"""


def test_only_established_connections_to_the_session_port_count() -> None:
    assert count_established_connections(_PROC_NET_TCP, 22) == 1


def test_a_docker_session_reads_its_own_connection_table() -> None:
    container = MagicMock()
    container.exec_run.return_value = MagicMock(
        exit_code=0, output=_PROC_NET_TCP.encode()
    )
    session = DockerSession(
        MagicMock(), container, MagicMock(), None, MagicMock(), MagicMock()
    )

    assert session.established_connections() == 1
    # Without an IPv6 table, cat fails after printing the IPv4 one.
    container.exec_run.return_value = MagicMock(
        exit_code=1, output=_PROC_NET_TCP.encode()
    )
    assert session.established_connections() == 1
    container.exec_run.return_value = MagicMock(exit_code=1, output=b"")
    assert session.established_connections() is None


def _docker_sshd_config() -> str:
    script = Path("src/worker/docker/ssh-session.sh").read_text(encoding="utf-8")
    body = script.split("flowmesh.conf << 'EOF'\n", 1)[1]
    return body.split("\nEOF\n", 1)[0]


def _process_sshd_config() -> str:
    return process_module._render_sshd_config(
        port=2222,
        session_dir=Path("/s"),
        host_key=Path("/s/key"),
        authorized_keys=Path("/s/keys"),
        login_user="fmssn61000",
        exported_env=[],
        bind_host="127.0.0.1",
    )


@pytest.mark.parametrize("render", [_docker_sshd_config, _process_sshd_config])
def test_sshd_ends_a_connection_whose_peer_stops_answering(render: Any) -> None:
    """A relayed connection whose origin vanished must not hold the session open."""
    options = dict(line.split(" ", 1) for line in render().splitlines() if " " in line)

    assert int(options["ClientAliveInterval"]) > 0
    assert int(options["ClientAliveCountMax"]) > 0
