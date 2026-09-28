"""A shutdown gives up a held Codex turn: the turn ends, and its permit never egresses.

A real runner drives the agent-episode executor over the Codex adapter, whose fake
app-server holds a model turn on the worker-local facade until the app-server is closed,
as a live one does. The worker is stopped while that turn waits on its permit.
"""

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from shared.harness import BoundaryEventKind, HarnessBackendKey
from shared.tasks.task_type import TaskType
from shared.tools.contract import MediatedOperationPermit
from shared.tools.facade import FacadeDescriptor
from shared.tools.model.schema import ModelCompletion, ModelToolCall
from shared.tools.search.schema import SEARCH_INTERFACE
from shared.utils.ids import new_mediated_permit_id
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker import runner as runner_module
from worker.egress import PendingEgressRequestStore
from worker.executors import agent_episode_executor as aee
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
_STOP_BUDGET_SEC = 5.0
_BOUNDARY_DRAIN_SEC = 3.0


def _eventually(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


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
            except Exception:
                pass
            replied.set()

        threading.Thread(target=call_model, daemon=True).start()
        self._on_held()
        deadline = time.monotonic() + _TURN_SEC
        # A closed app-server delivers no later event, whatever its call answered.
        while time.monotonic() < deadline:
            if self.closed.is_set():
                raise RuntimeError("the Codex app-server closed mid-turn")
            if replied.is_set():
                break
            time.sleep(0.01)
        return CodexEvent(kind="completed", value="done")

    def cancel(self, thread_id: str | None) -> None:
        self.closed.set()


def _codex_runner(
    tmp_path: Path, backend: str
) -> tuple[Runner, Lifecycle, MagicMock, AgentEpisodeExecutor]:
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    client = cast(MagicMock, lifecycle.client)
    client.worker_id = "wrk-test"
    client.incarnation = 1
    client.iter_interrupts.return_value = []
    client.iter_stops.return_value = []
    client.iter_mediated_ops.return_value = []
    client.create_task_log_emitter.return_value = None
    executor = AgentEpisodeExecutor(make_worker_config(), lifecycle=lifecycle)
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[
            make_worker_task_message(
                {"taskType": "agent"},
                task_id=_TASK,
                task_type=TaskType.AGENT,
                agent_episode={"backend": {"backend": backend, "version": "v1"}},
            )
        ],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"agent_episode": executor},
        default_executor=executor,
        logger=MagicMock(),
    )
    return runner, lifecycle, client, executor


def test_a_shutdown_gives_up_a_held_codex_turn(tmp_path: Path) -> None:
    runner, lifecycle, client, _ = _codex_runner(tmp_path, "fake-codex")
    stamps: dict[str, float] = {}
    client.unregister.side_effect = lambda *_, **__: stamps.setdefault(
        "unregister", time.monotonic()
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
        patch.object(runner_module, "_STOP_BUDGET_SEC", _STOP_BUDGET_SEC),
        patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", _BOUNDARY_DRAIN_SEC),
    ):
        run_until_exit(runner, lifecycle, MagicMock())
    # The step reports on its own thread, which a loaded host may run after the exit.
    assert _eventually(lambda: client.task_cancelled.called)
    assert _eventually(lambda: not rendezvous._waiters)
    held = next(iter(rendezvous._held))

    # With no waiter armed, a permit for the held turn is dropped where it is routed.
    runner._route_mediated_op("permit", _permit("model", *held))

    assert servers[0].closed.is_set()
    client.task_cancelled.assert_called_once()
    client.task_failed.assert_not_called()
    assert stamps["unregister"] - stamps["sigterm"] < _STOP_BUDGET_SEC
    assert sidecar.egressed == []
    assert lifecycle.pending_egress_requests.occurrences() == []


_SEARCH = FacadeDescriptor(
    name="web_search",
    kind=BoundaryEventKind.INVOCATION,
    interface=SEARCH_INTERFACE,
    tool_schema=json.dumps(
        {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
    ),
)


class _SearchingSidecar:
    """Answers the turn's model call with a web search, and runs a search permit by
    dropping its request, as a committed outcome does."""

    def __init__(self, pending: PendingEgressRequestStore) -> None:
        self._pending = pending
        self.searched: list[tuple[str, str]] = []

    def egress_now(self, permit: MediatedOperationPermit) -> ModelCompletion:
        return ModelCompletion(
            content="let me look",
            tool_calls=(
                ModelToolCall(
                    call_id="c1", name="web_search", arguments='{"query": "q"}'
                ),
            ),
        )

    def submit_permit(self, permit: MediatedOperationPermit) -> None:
        self.searched.append((permit.agent_task_id, permit.call_correlation))
        self._pending.delete(permit.agent_task_id, permit.call_correlation)

    def stop(self) -> None:
        pass


class _SearchingAppServer:
    """A Codex app-server whose turn makes one model call that asks for a search, so
    the facade captures a group and completes the turn clean."""

    def __init__(self, facade: ResponsesFacade, route_permit: Any) -> None:
        self._facade = facade
        self._route_permit = route_permit

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
        token = self._facade.register_episode(_TASK, "http://model/v1", "m", [_SEARCH])
        turn = threading.Thread(
            target=self._facade.handle_turn,
            args=(_TASK, token, {"input": [{"role": "user", "content": "find"}]}),
            daemon=True,
        )
        turn.start()
        self._route_permit("model", _TASK)
        turn.join(_TURN_SEC)
        return CodexEvent(kind="completed", value="done")

    def cancel(self, thread_id: str | None) -> None:
        pass


def _permit(interface: str, task_id: str, call_correlation: str) -> dict[str, Any]:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=task_id,
        call_correlation=call_correlation,
        interface=interface,
        subject=interface,
        invocation_id="inv-1",
        idempotency_key="idm-1",
        request_digest="d",
        target_id="wrk-test",
        target_generation=1,
        deadline_epoch=2_000_000_000.0,
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=4000,
    ).model_dump(mode="json")


def test_a_give_up_during_the_seal_keeps_the_searches_the_turn_asked_for(
    tmp_path: Path,
) -> None:
    runner, lifecycle, client, executor = _codex_runner(tmp_path, "fake-codex-seal")
    sidecar = _SearchingSidecar(lifecycle.pending_egress_requests)
    runner._mediated_sidecar = cast(Any, sidecar)
    rendezvous = runner._model_turn_rendezvous

    def route_model_permit(interface: str, task_id: str) -> None:
        deadline = time.monotonic() + _TURN_SEC
        while time.monotonic() < deadline and not rendezvous._waiters:
            time.sleep(0.005)
        (key,) = rendezvous._waiters
        runner._route_mediated_op("permit", _permit(interface, *key))

    def build(
        backend: HarnessBackendKey, task: Any, config: Any, facade: Any, *_: Any
    ) -> CodexAppServerHarnessAdapter:
        return CodexAppServerHarnessAdapter(
            _SearchingAppServer(facade, route_model_permit)
        )

    register_adapter("fake-codex-seal", build)

    gave_up: list[bool] = []

    class _Holder:
        """Seals while the worker is told to stop, returning once the give-up has
        reached the facade."""

        def seal(self, state: Any, attachment: Any) -> None:
            runner.stop()
            facade = lifecycle.responses_facade
            assert facade is not None
            deadline = time.monotonic() + _TURN_SEC
            while time.monotonic() < deadline and _TASK in facade._episodes:
                time.sleep(0.005)
            gave_up.append(executor._signals.cancelled)

    def run_the_searches(*_: Any, **kwargs: Any) -> None:
        group = kwargs["metadata"]["agent_episode_facade_group"]
        for member in group["members"]:
            runner._route_mediated_op(
                "permit", _permit("search/v1", _TASK, member["call_correlation"])
            )

    client.task_succeeded.side_effect = run_the_searches
    with (
        patch.object(
            AgentEpisodeExecutor,
            "_open_private_state",
            lambda self, dispatch: (MagicMock(), _Holder()),
        ),
        patch.object(aee, "_attachment", lambda dispatch: MagicMock()),
        patch.object(Runner, "_write_results", lambda self, msg, out_dir, out: {}),
        patch.object(runner_module, "_STOP_BUDGET_SEC", _STOP_BUDGET_SEC),
        patch.object(runner_module, "_BOUNDARY_DRAIN_SEC", _BOUNDARY_DRAIN_SEC),
    ):
        run_until_exit(runner, lifecycle, MagicMock())

    assert gave_up == [True]
    assert _eventually(lambda: client.task_succeeded.called)
    client.task_cancelled.assert_not_called()
    metadata = client.task_succeeded.call_args.kwargs["metadata"]
    (member,) = metadata["agent_episode_facade_group"]["members"]
    assert sidecar.searched == [(_TASK, member["call_correlation"])]
    assert lifecycle.pending_egress_requests.occurrences() == []
