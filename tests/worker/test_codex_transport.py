"""The live Codex transport reaps its app-server however a close meets its spawn.

The SDK client is faked after ``openai_codex.CodexClient``'s start and close, and no
process is started or signalled.
"""

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


class _Proc:
    exit_delay = 0.0

    def __init__(self) -> None:
        self.exited = threading.Event()

    def terminate(self) -> None:
        time.sleep(self.exit_delay)
        self.exited.set()


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
        if (proc := self._proc) is None:
            return
        self._proc = None
        proc.terminate()


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
    )
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
    proc.exit_delay = 0.3
    first = threading.Thread(target=transport.close, daemon=True)
    first.start()
    assert _wait_for(lambda: client._proc is None)

    transport.close()

    assert proc.exited.is_set()
    first.join(5)


def _wait_for(condition: Any) -> bool:
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True
