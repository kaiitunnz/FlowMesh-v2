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
from worker.executors.base_executor import ExecutionError
from worker.harness import register_adapter
from worker.main import build_capabilities


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
    from worker.harness import UnknownHarnessBackendError

    ex = AgentEpisodeExecutor(make_worker_config())
    msg = make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={"backend": {"backend": "nonesuch", "version": "v1"}},
    )
    with pytest.raises(UnknownHarnessBackendError):
        ex.run(msg, tmp_path)
