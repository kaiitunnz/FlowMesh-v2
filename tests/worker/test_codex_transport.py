"""The live Codex transport reaps its app-server however a close meets its spawn.

The SDK client is faked after ``openai_codex.CodexClient``'s start and close, and its
process after the supervisor that runs the app-server; no process is started or
signalled.
"""

import subprocess  # nosec B404 - only for its TimeoutExpired
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

from worker.executors.harness import codex_transport  # noqa: E402
from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    CodexTransportError,
    RealCodexAppServerTransport,
)
from worker.utils.subreaper import REAPED, UNPROVED  # noqa: E402


class _Proc:
    """A supervisor whose app-server exits once its stdin closes, or ignores that and
    leaves the supervisor's SIGTERM to end the tree."""

    exit_delay = 0.0
    exits_on_eof = True
    proves = True

    def __init__(self) -> None:
        self.exited = threading.Event()
        self.signalled = threading.Event()
        self.stdin = SimpleNamespace(close=self._on_eof)
        self.returncode: int | None = None

    def _on_eof(self) -> None:
        if self.exits_on_eof:
            threading.Timer(self.exit_delay, self._exit).start()

    def _exit(self) -> None:
        self.returncode = REAPED if self.proves else UNPROVED
        self.exited.set()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired("codex", timeout or 0)
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signalled.set()
        self._exit()


class _Client:
    """Mirrors the SDK client: close does nothing until the spawn assigned a process."""

    spawning = threading.Event()
    proceed = threading.Event()
    made: list["_Client"] = []

    def __init__(self, config: Any) -> None:
        self._proc: _Proc | None = None
        self.made.append(self)

    def start(self) -> None:
        self.spawning.set()
        assert self.proceed.wait(5)
        self._proc = _Proc()

    def initialize(self) -> None:
        pass

    def thread_start(self) -> Any:
        return SimpleNamespace(thread=SimpleNamespace(id="thr-1"))

    def close(self) -> None:
        self._proc = None


@pytest.fixture
def transport(tmp_path: Path) -> Any:
    _Client.spawning = threading.Event()
    _Client.proceed = threading.Event()
    _Client.made = []
    config = CodexTransportConfig(
        base_url="http://127.0.0.1:1",
        model="m",
        codex_home=tmp_path / "home",
        initial_input="task",
        task_id="tsk-1",
        exit_grace_sec=0.2,
        reap_budget_sec=0.2,
    )
    _Proc.exit_delay = 0.0
    _Proc.exits_on_eof = True
    _Proc.proves = True
    with patch.object(codex_transport, "CodexClient", _Client):
        yield RealCodexAppServerTransport(config)


def test_a_close_during_the_spawn_reaps_the_app_server_it_started(
    transport: RealCodexAppServerTransport,
) -> None:
    errors: list[BaseException] = []

    def open_thread() -> None:
        try:
            transport.thread_start()
        except BaseException as exc:
            errors.append(exc)

    opener = threading.Thread(target=open_thread, daemon=True)
    opener.start()
    assert _Client.spawning.wait(5)
    closer = threading.Thread(target=transport.close, daemon=True)
    closer.start()
    _Client.proceed.set()
    opener.join(5)
    closer.join(5)

    (client,) = _Client.made
    assert all(isinstance(exc, CodexTransportError) for exc in errors)
    assert client._proc is None
    with pytest.raises(CodexTransportError):
        transport.thread_start()
    assert len(_Client.made) == 1


def test_every_close_returns_once_the_app_server_exited(
    transport: RealCodexAppServerTransport,
) -> None:
    _Client.proceed.set()
    transport.thread_start()
    (client,) = _Client.made
    proc = client._proc
    assert proc is not None
    proc.exit_delay = 0.1
    first = threading.Thread(target=transport.close, daemon=True)
    first.start()

    assert transport.close()
    assert proc.exited.is_set()
    first.join(5)


def test_the_app_server_ends_on_end_of_input_before_any_signal(
    transport: RealCodexAppServerTransport,
) -> None:
    _Client.proceed.set()
    transport.thread_start()
    (client,) = _Client.made
    proc = client._proc
    assert proc is not None

    assert transport.quiesce()
    assert not proc.signalled.is_set()


def test_an_app_server_ignoring_end_of_input_is_ended_by_its_supervisor(
    transport: RealCodexAppServerTransport,
) -> None:
    _Proc.exits_on_eof = False
    _Client.proceed.set()
    transport.thread_start()
    (client,) = _Client.made
    proc = client._proc
    assert proc is not None

    assert transport.quiesce()
    assert proc.signalled.is_set()


def test_an_unproved_reap_keeps_the_tree_for_a_later_close(
    transport: RealCodexAppServerTransport,
) -> None:
    _Proc.proves = False
    _Client.proceed.set()
    transport.thread_start()
    (client,) = _Client.made
    proc = client._proc
    assert proc is not None

    assert not transport.quiesce()
    # The SDK's close, which would kill a still-draining supervisor, never ran.
    assert client._proc is proc
    assert not transport.close()


def _wait_for(condition: Any) -> bool:
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True
