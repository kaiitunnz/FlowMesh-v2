"""The agent-episode executor drives one adapter step behind a backend key.

An agent whose dispatch carries a harness backend key routes to the episode executor
and advertises the AGENT capability even when the legacy UTU executor cannot import; a
step returns the backend's :class:`HarnessResult`; and a native-bypass backend is
refused.
"""

from pathlib import Path

import pytest

from shared.harness import (
    BoundaryEventKind,
    BoundaryRequest,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessResult,
    HarnessResultKind,
)
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.agent_episode_executor import (
    AgentEpisodeExecutor,
    AgentEpisodeResult,
)
from worker.executors.base_executor import ExecutionError, Executor
from worker.harness import UnknownHarnessBackendError, register_adapter
from worker.main import build_capabilities
from worker.runner import Runner


class _FakeAdapter(HarnessAdapter):
    def __init__(self, step: HarnessResult, *, bypass: bool = True) -> None:
        self._step = step
        self._bypass = bypass
        self.started: list[str | None] = []
        self.cancelled: list[str] = []

    def backend_key(self) -> HarnessBackendKey:
        return HarnessBackendKey(backend="fake", version="v1")

    def start(self, activation_id, *, capsule, outcomes) -> HarnessResult:
        self.started.append(capsule.blob if capsule else None)
        return self._step

    def cancel(self, activation_id: str) -> None:
        self.cancelled.append(activation_id)

    def bypass_disabled(self) -> bool:
        return self._bypass


def _dispatch_msg(**episode: object) -> WorkerTaskMessage:
    return make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={
            "backend": {"backend": "fake", "version": "v1"},
            **episode,
        },
    )


def test_agent_episode_key_is_registered() -> None:
    assert "agent_episode" in EXECUTOR_REGISTRY
    cls = EXECUTOR_REGISTRY.get("agent_episode")
    assert cls is not None and cls.supported_task_types == frozenset({TaskType.AGENT})


def test_worker_advertises_agent_without_the_utu_executor() -> None:
    # The episode executor is dependency-light, so a CPU worker that cannot import the
    # UTU agent executor still advertises AGENT through it.
    cls = EXECUTOR_REGISTRY.get("agent_episode")
    assert cls is not None
    caps = build_capabilities({"agent_episode": cls(make_worker_config())})
    assert TaskType.AGENT in caps.supported_task_types


def test_step_returns_the_harness_result(tmp_path: Path) -> None:
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value="done")
    register_adapter("fake", lambda backend, task, config: _FakeAdapter(completion))
    ex = AgentEpisodeExecutor(make_worker_config())
    out = ex.run(_dispatch_msg(capsule_blob="after:c0"), tmp_path)
    assert isinstance(out, AgentEpisodeResult)
    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value == "done"


def test_boundary_step_carries_no_terminal_value(tmp_path: Path) -> None:
    boundary = HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="model"
        ),
    )
    register_adapter("fake", lambda backend, task, config: _FakeAdapter(boundary))
    ex = AgentEpisodeExecutor(make_worker_config())
    out = ex.run(_dispatch_msg(), tmp_path)
    assert out.harness_result.kind is HarnessResultKind.BOUNDARY and out.value is None


def test_native_bypass_backend_is_refused(tmp_path: Path) -> None:
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value="x")
    register_adapter(
        "fake", lambda backend, task, config: _FakeAdapter(completion, bypass=False)
    )
    ex = AgentEpisodeExecutor(make_worker_config())
    with pytest.raises(ExecutionError, match="bypass"):
        ex.run(_dispatch_msg(), tmp_path)


def test_missing_dispatch_context_is_an_error(tmp_path: Path) -> None:
    msg = make_worker_task_message({"taskType": "agent"}, task_type=TaskType.AGENT)
    ex = AgentEpisodeExecutor(make_worker_config())
    with pytest.raises(ExecutionError, match="without an agent-episode"):
        ex.run(msg, tmp_path)


def test_unknown_backend_has_no_binding(tmp_path: Path) -> None:
    ex = AgentEpisodeExecutor(make_worker_config())
    msg = make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={"backend": {"backend": "nonesuch", "version": "v1"}},
    )
    with pytest.raises(UnknownHarnessBackendError):
        ex.run(msg, tmp_path)


class _RecordingExecutor(Executor):
    def __init__(self, result: object) -> None:
        super().__init__(make_worker_config())
        self._result = result
        self.ran = False

    def run(self, task, out_dir):  # type: ignore[no-untyped-def]
        self.ran = True
        return self._result


def _runner(tmp_path: Path, executors: dict[str, Executor]) -> Runner:
    from unittest.mock import MagicMock

    from tests.worker.factories import make_worker_hardware

    lifecycle = MagicMock()
    lifecycle.worker_id = "wrk-test"
    lifecycle.cost_per_hour = 1.0
    lifecycle.client.create_task_log_emitter.return_value = None
    lifecycle.client.iter_interrupts.return_value = []
    lifecycle.client.iter_stops.return_value = []
    return Runner(
        lifecycle=lifecycle,
        task_stream=[],
        results_dir=tmp_path,
        hardware=make_worker_hardware(),
        executors=executors,
        default_executor=executors["agent"],
        logger=MagicMock(),
    )


def _agent_result() -> AgentEpisodeResult:
    return AgentEpisodeResult(
        harness_result=HarnessResult(kind=HarnessResultKind.COMPLETION, value="done"),
        value="done",
    )


def test_runner_routes_an_episode_message_to_the_episode_executor(
    tmp_path: Path,
) -> None:
    # An agent message carrying an agent-episode dispatch routes to the episode
    # executor; a bare agent message routes to the UTU executor. This is the production
    # seam the dispatcher and runner select, exercised end-to-end on the real stack.
    episode = _RecordingExecutor(_agent_result())
    utu = _RecordingExecutor(_agent_result())
    runner = _runner(tmp_path, {"agent_episode": episode, "agent": utu})

    with_episode = make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={"backend": {"backend": "scripted", "version": "v1"}},
    )
    runner.task_stream = [with_episode]
    runner.start()
    assert episode.ran and not utu.ran

    episode.ran = utu.ran = False
    bare = make_worker_task_message({"taskType": "agent"}, task_type=TaskType.AGENT)
    runner.task_stream = [bare]
    runner.start()
    assert utu.ran and not episode.ran
