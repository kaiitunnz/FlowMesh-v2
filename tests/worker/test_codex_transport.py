"""The live Codex app-server transport's teardown."""

import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    RealCodexAppServerTransport,
)


def test_a_cancel_returns_while_the_app_server_is_still_exiting(
    tmp_path: Path,
) -> None:
    transport = RealCodexAppServerTransport(
        CodexTransportConfig(
            base_url="http://127.0.0.1:1/v1",
            model="codex-model",
            codex_home=tmp_path / "codex_home",
            initial_input="task",
            task_id="tsk-1",
        )
    )
    exiting = threading.Event()
    exited = threading.Event()

    def slow_exit() -> None:
        exiting.set()
        exited.wait(5)

    cast(Any, transport)._finalizer = slow_exit
    started = time.monotonic()

    transport.cancel("thr-1")

    returned_after = time.monotonic() - started
    assert exiting.wait(5)
    exited.set()
    assert returned_after < 1.0
