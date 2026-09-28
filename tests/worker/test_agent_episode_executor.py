"""The agent-episode executor drives one adapter step behind a backend key.

An agent whose dispatch carries a harness backend key routes to the episode executor
and advertises the AGENT capability; a step returns the backend's result; and a
native-bypass backend is refused.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.harness import (
    REQUIRED_MEDIATED_FACADES,
    BoundaryEventKind,
    BoundaryRequest,
    EgressHandoffMode,
    HarnessAdapter,
    HarnessBackendKey,
    HarnessResult,
    HarnessResultKind,
    MediatedFacade,
)
from shared.private_state import PrivateStateUnavailable, PrivateStateUnavailableReason
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import WorkerTaskMessage
from shared.tools.facade import (
    FacadeCallMember,
    FacadeCompletionMode,
    FacadeDescriptor,
    FacadeTurnGroup,
)
from shared.tools.model.schema import ModelCompletion, ModelToolCall
from shared.tools.search.schema import SEARCH_INTERFACE, parse_search_request
from tests.worker.factories import (
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker.egress import PendingEgressRequestStore
from worker.executors import EXECUTOR_REGISTRY
from worker.executors import agent_episode_executor as aee
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.base_executor import ExecutionError, Executor, ExecutorTask
from worker.executors.episode_support import EpisodeStepResult
from worker.executors.harness import UnknownHarnessBackendError, register_adapter
from worker.lifecycle import Lifecycle
from worker.main import build_capabilities
from worker.model_turn import ResponsesFacade
from worker.resident import ResidentRequestStore
from worker.runner import Runner

_SEARCH = FacadeDescriptor(
    name="web_search",
    kind=BoundaryEventKind.INVOCATION,
    interface=SEARCH_INTERFACE,
    tool_schema=json.dumps(
        {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
    ),
)


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

    def mediated_facades(self) -> frozenset[MediatedFacade]:
        return (
            REQUIRED_MEDIATED_FACADES
            if self._bypass
            else frozenset({MediatedFacade.MODEL})
        )


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


def test_worker_advertises_agent_through_the_episode_executor() -> None:
    # The episode executor is dependency-light, so a CPU worker advertises AGENT through
    # it — the only executor that services the task type.
    cls = EXECUTOR_REGISTRY.get("agent_episode")
    assert cls is not None
    caps = build_capabilities({"agent_episode": cls(make_worker_config())})
    assert TaskType.AGENT in caps.supported_task_types


def test_step_returns_the_harness_result(tmp_path: Path) -> None:
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value="done")
    register_adapter(
        "fake",
        lambda backend, task, config, facade, state, sandbox: _FakeAdapter(completion),
    )
    ex = AgentEpisodeExecutor(make_worker_config())
    out = ex.run(_dispatch_msg(capsule_blob="after:c0"), tmp_path)
    assert isinstance(out, EpisodeStepResult)
    assert out.harness_result.kind is HarnessResultKind.COMPLETION
    assert out.value == "done"


def test_boundary_step_carries_no_terminal_value(tmp_path: Path) -> None:
    boundary = HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION, call_correlation="c0", interface="model"
        ),
    )
    register_adapter(
        "fake",
        lambda backend, task, config, facade, state, sandbox: _FakeAdapter(boundary),
    )
    ex = AgentEpisodeExecutor(make_worker_config())
    out = ex.run(_dispatch_msg(), tmp_path)
    assert out.harness_result.kind is HarnessResultKind.BOUNDARY and out.value is None


def test_native_bypass_backend_is_refused(tmp_path: Path) -> None:
    completion = HarnessResult(kind=HarnessResultKind.COMPLETION, value="x")
    register_adapter(
        "fake",
        lambda backend, task, config, facade, state, sandbox: _FakeAdapter(
            completion, bypass=False
        ),
    )
    ex = AgentEpisodeExecutor(make_worker_config())
    with pytest.raises(ExecutionError, match="mediate"):
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
        default_executor=executors["default"],
        logger=MagicMock(),
    )


def _agent_result() -> EpisodeStepResult:
    return EpisodeStepResult(
        harness_result=HarnessResult(kind=HarnessResultKind.COMPLETION, value="done"),
        value="done",
    )


def test_runner_routes_an_episode_message_to_the_episode_executor(
    tmp_path: Path,
) -> None:
    # An agent message carrying an agent-episode dispatch routes to the episode
    # executor; a bare agent with no dispatch context fails without falling back to any
    # other executor. This is the production seam the dispatcher and runner select.
    episode = _RecordingExecutor(_agent_result())
    default = _RecordingExecutor(_agent_result())
    runner = _runner(tmp_path, {"agent_episode": episode, "default": default})

    with_episode = make_worker_task_message(
        {"taskType": "agent"},
        task_type=TaskType.AGENT,
        agent_episode={"backend": {"backend": "scripted", "version": "v1"}},
    )
    runner.task_stream = [with_episode]
    runner.start()
    assert episode.ran and not default.ran

    episode.ran = default.ran = False
    bare = make_worker_task_message({"taskType": "agent"}, task_type=TaskType.AGENT)
    runner.task_stream = [bare]
    runner.start()
    assert not episode.ran and not default.ran


class _CapturingAdapter(_FakeAdapter):
    """A held turn that captures a search group through the facade, then either ends
    the step or raises."""

    def __init__(self, facade: ResponsesFacade, raises: bool) -> None:
        super().__init__(HarnessResult(kind=HarnessResultKind.COMPLETION, value="ok"))
        self._facade = facade
        self._raises = raises

    def start(self, activation_id, *, capsule, outcomes) -> HarnessResult:
        token = self._facade.register_episode(
            activation_id, "http://up/v1", "m", [_SEARCH]
        )
        self._facade.handle_turn(activation_id, token, {"input": "find it"})
        if self._raises:
            raise RuntimeError("the app-server died after the turn")
        return self._step


@pytest.mark.parametrize("raises", [False, True])
def test_a_step_that_raises_drops_the_requests_its_turn_stashed(
    tmp_path: Path, raises: bool
) -> None:
    pending = PendingEgressRequestStore()
    held = MagicMock()
    held.run.return_value = ModelCompletion(
        content="searching",
        tool_calls=(
            ModelToolCall(call_id="c1", name="web_search", arguments='{"query": "q"}'),
        ),
    )
    facade = ResponsesFacade(held_egress=held, pending=pending)
    lifecycle = MagicMock()
    lifecycle.responses_facade = facade
    register_adapter(
        "fake",
        lambda backend, task, config, _facade, state, sandbox: _CapturingAdapter(
            facade, raises
        ),
    )
    ex = AgentEpisodeExecutor(make_worker_config(), lifecycle=lifecycle)
    msg = _dispatch_msg()

    if raises:
        with pytest.raises(RuntimeError):
            ex.run(msg, tmp_path)
        assert pending.occurrences() == []
    else:
        out = ex.run(msg, tmp_path)
        assert out.facade_group is not None
        assert pending.occurrences() == [
            (msg.task_id, m.call_correlation) for m in out.facade_group.members
        ]
    assert facade.take_captured_group(msg.task_id) is None


class _YieldingAdapter(_FakeAdapter):
    def egress_handoff_mode(self) -> EgressHandoffMode:
        return EgressHandoffMode.DURABLE_PRE_EGRESS_YIELD


class _FailingSeal:
    def seal(self, state: object, attachment: object) -> None:
        raise PrivateStateUnavailable(
            PrivateStateUnavailableReason.STALE_EPOCH,
            "superseded",
            reference_id="aps-x",
        )


@pytest.mark.parametrize("resident", [False, True])
def test_a_step_whose_seal_fails_holds_no_request_for_control(
    tmp_path: Path, resident: bool
) -> None:
    boundary = HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface="model" if resident else SEARCH_INTERFACE,
            request_payload=(
                '{"messages": []}' if resident else '{"query": "q", "max_results": 3}'
            ),
        ),
    )
    register_adapter(
        "fake",
        lambda backend, task, config, facade, state, sandbox: _YieldingAdapter(
            boundary
        ),
    )
    lifecycle = MagicMock()
    lifecycle.pending_egress_requests = PendingEgressRequestStore()
    lifecycle.resident_requests = ResidentRequestStore()
    lifecycle.responses_facade = None
    ex = AgentEpisodeExecutor(make_worker_config(), lifecycle=lifecycle)
    msg = _dispatch_msg(
        model_binding={"mode": "resident"} if resident else None,
    )
    with (
        patch.object(
            AgentEpisodeExecutor,
            "_open_private_state",
            lambda self, dispatch: (MagicMock(), _FailingSeal()),
        ),
        patch.object(aee, "_attachment", lambda dispatch: MagicMock()),
        pytest.raises(PrivateStateUnavailable),
    ):
        ex.run(msg, tmp_path)

    assert lifecycle.pending_egress_requests.occurrences() == []
    assert lifecycle.resident_requests.occurrences() == []


class _CapturingExecutor(Executor):
    """A step that holds one request for control of the given kind."""

    def __init__(self, lifecycle: Lifecycle, kind: str) -> None:
        super().__init__(make_worker_config(), lifecycle=lifecycle)
        self._kind = kind

    def run(self, task: ExecutorTask, out_dir: Path) -> EpisodeStepResult:
        assert self._lifecycle is not None
        request = BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="c0",
            interface=SEARCH_INTERFACE,
            request_digest="d",
        )
        if self._kind == "facade_group":
            self._lifecycle.pending_egress_requests.put(
                task.task_id, "c1", parse_search_request('{"query": "q"}')
            )
            return EpisodeStepResult(
                harness_result=HarnessResult(
                    kind=HarnessResultKind.COMPLETION, value="ok"
                ),
                facade_group=FacadeTurnGroup(
                    group_id="grp-1",
                    activation_id=task.task_id,
                    turn_id="turn-1",
                    members=(
                        FacadeCallMember(
                            ordinal=0,
                            kind=BoundaryEventKind.INVOCATION,
                            completion_mode=FacadeCompletionMode.AWAIT_OUTCOME,
                            call_correlation="c1",
                            harness_call_id="call-1",
                            tool_name="web_search",
                            interface_or_region=SEARCH_INTERFACE,
                            request_digest="d",
                        ),
                    ),
                ),
            )
        if self._kind == "resident":
            self._lifecycle.resident_requests.put(task.task_id, "c0", "{}")
        else:
            self._lifecycle.pending_egress_requests.put(
                task.task_id, "c0", parse_search_request('{"query": "q"}')
            )
        return EpisodeStepResult(
            harness_result=HarnessResult(
                kind=HarnessResultKind.BOUNDARY, request=request
            )
        )


@pytest.mark.parametrize("kind", ["search", "resident", "facade_group"])
@pytest.mark.parametrize("reported", [False, True])
def test_a_step_whose_report_fails_holds_no_request_for_control(
    tmp_path: Path, kind: str, reported: bool
) -> None:
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    client = lifecycle.client
    assert isinstance(client, MagicMock)
    client.worker_id = "wrk-test"
    client.create_task_log_emitter.return_value = None
    client.iter_interrupts.return_value = []
    client.iter_stops.return_value = []
    client.iter_mediated_ops.return_value = []
    executor = _CapturingExecutor(lifecycle, kind)
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[_dispatch_msg()],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={"agent_episode": executor, "default": executor},
        default_executor=executor,
        logger=MagicMock(),
    )

    def write_results(self: Runner, msg: object, out_dir: object, out: object) -> dict:
        if not reported:
            raise ExecutionError("the result store is unavailable", retryable=True)
        return {}

    with patch.object(Runner, "_write_results", write_results):
        runner.start()

    held = (
        lifecycle.pending_egress_requests.occurrences()
        + lifecycle.resident_requests.occurrences()
    )
    if reported:
        client.task_succeeded.assert_called_once()
        assert len(held) == 1
    else:
        client.task_failed.assert_called_once()
        assert held == []
