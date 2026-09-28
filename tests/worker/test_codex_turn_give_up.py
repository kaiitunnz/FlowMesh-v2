"""A shutdown gives up a held Codex turn: the turn ends, and its permit never egresses.

A real runner drives the agent-episode executor over the Codex adapter, whose fake
app-server holds a model turn on the worker-local facade until the app-server is closed,
as a live one does. The worker is stopped while that turn waits on its permit.
"""

import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from shared.harness import HarnessBackendKey
from shared.tasks.task_type import TaskType
from shared.tools.contract import MediatedOperationPermit
from shared.tools.model.schema import ModelCompletion
from shared.utils.ids import new_mediated_permit_id
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker import runner as runner_module
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.harness import register_adapter
from worker.executors.harness.codex import (
    CodexAppServerHarnessAdapter,
    CodexEvent,
    CodexInjectItem,
)
from worker.lifecycle import Lifecycle
from worker.main import run_until_exit
from worker.model_turn import ResponsesFacade
from worker.runner import Runner

_TASK = "tsk-codex"
_TURN_SEC = 4.0


class _Sidecar:
    def __init__(self) -> None:
        self.egressed: list[str] = []

    def egress_now(self, permit: MediatedOperationPermit) -> ModelCompletion:
        self.egressed.append(permit.permit_id)
        return ModelCompletion(content="hi")

    def stop(self) -> None:
        pass


class _AppServer:
    """A Codex app-server whose turn makes one model call through the facade and
    ends when the call returns, or fails once the app-server is closed."""

    def __init__(self, facade: ResponsesFacade, on_held: Any) -> None:
        self._facade = facade
        self._on_held = on_held
        self.closed = threading.Event()

    def thread_start(self) -> str:
        return "thr-1"

    def thread_resume(self, thread_id: str, rollout_ref: str) -> None:
        pass

    def thread_inject_items(
        self, thread_id: str, items: Sequence[CodexInjectItem]
    ) -> None:
        pass

    def turn_start(self, thread_id: str) -> str:
        return "turn-1"

    def next_event(self, thread_id: str, turn_id: str) -> CodexEvent:
        token = self._facade.register_episode(_TASK, "http://model/v1", "m", [])
        replied = threading.Event()

        def call_model() -> None:
            try:
                self._facade.handle_turn(
                    _TASK, token, {"input": [{"role": "user", "content": "hi"}]}
                )
            except Exception:  # noqa: BLE001 - a failed turn is still a reply
                pass
            replied.set()

        threading.Thread(target=call_model, daemon=True).start()
        self._on_held()
        deadline = time.monotonic() + _TURN_SEC
        while time.monotonic() < deadline and not replied.is_set():
            if self.closed.is_set():
                raise RuntimeError("the Codex app-server closed mid-turn")
            time.sleep(0.01)
        return CodexEvent(kind="completed", value="done")

    def cancel(self, thread_id: str) -> None:
        self.closed.set()


def test_a_shutdown_gives_up_a_held_codex_turn(tmp_path: Path) -> None:
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    client = cast(MagicMock, lifecycle.client)
    client.worker_id = "wrk-test"
    client.incarnation = 1
    client.iter_interrupts.return_value = []
    client.iter_stops.return_value = []
    client.iter_mediated_ops.return_value = []
    client.create_task_log_emitter.return_value = None
    stamps: dict[str, float] = {}
    client.unregister.side_effect = lambda *_, **__: stamps.setdefault(
        "unregister", time.monotonic()
    )
    executor = AgentEpisodeExecutor(make_worker_config(), lifecycle=lifecycle)
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[
            make_worker_task_message(
                {"taskType": "agent"},
                task_id=_TASK,
                task_type=TaskType.AGENT,
                agent_episode={"backend": {"backend": "fake-codex", "version": "v1"}},
            )
        ],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"agent_episode": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    sidecar = _Sidecar()
    runner._mediated_sidecar = cast(Any, sidecar)
    rendezvous = runner._model_turn_rendezvous

    def stop_once_held() -> None:
        while not any(task == _TASK for task, _ in rendezvous._waiters):
            time.sleep(0.01)
        stamps["sigterm"] = time.monotonic()
        runner.stop()

    servers: list[_AppServer] = []

    def build(
        backend: HarnessBackendKey, task: Any, config: Any, facade: Any, *_: Any
    ) -> CodexAppServerHarnessAdapter:
        servers.append(_AppServer(facade, stop_once_held))
        return CodexAppServerHarnessAdapter(servers[-1])

    register_adapter("fake-codex", build)
    with (
        patch.object(runner_module, "_STOP_BUDGET_SEC", 2.5),
        patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", 1.5),
    ):
        run_until_exit(runner, lifecycle, MagicMock())
    assert not rendezvous._waiters
    held = next(iter(rendezvous._held))

    runner._route_mediated_op(
        "permit",
        MediatedOperationPermit(
            permit_id=new_mediated_permit_id(),
            agent_task_id=held[0],
            call_correlation=held[1],
            interface="model",
            subject="model",
            invocation_id="inv-1",
            idempotency_key="idm-1",
            request_digest="d",
            target_id="wrk-test",
            target_generation=1,
            deadline_epoch=2_000_000_000.0,
            max_results=1,
            timeout_sec=10.0,
            result_char_cap=4000,
        ).model_dump(mode="json"),
    )

    time.sleep(0.2)  # a waiter handed the permit would egress it on its own thread
    assert servers[0].closed.is_set()
    client.task_cancelled.assert_called_once()
    client.task_failed.assert_not_called()
    assert stamps["unregister"] - stamps["sigterm"] < 2.5
    assert sidecar.egressed == []
    assert lifecycle.pending_egress_requests.occurrences() == []
