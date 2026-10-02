"""The held turn answers an agent's commands locally, without a control round trip.

A ``run_command`` call the model emits is resolved by the worker inside the turn: it is
never captured as a turn-group member, never stashed for the egress path, and never
settles through control. Only the model calls between commands are mediated, as they
already were.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.harness import BoundaryEventKind, HarnessResult, HarnessResultKind
from shared.sandbox import (
    SANDBOX_EXECUTE_INTERFACE,
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
    SandboxUnavailable,
)
from shared.tools.contract import AgentModelTurnProposal, MediatedOperationPermit
from shared.tools.facade import FacadeDescriptor, FacadeResolution
from shared.tools.model.schema import (
    ModelCompletion,
    ModelRequest,
    ModelToolCall,
    model_request_digest,
)
from shared.tools.search.schema import SEARCH_INTERFACE
from shared.utils.ids import new_mediated_permit_id
from tests.worker.factories import make_worker_config
from tests.worker.test_agent_episode_executor import _dispatch_msg, _FakeAdapter
from worker.egress import HeldEgressReject, PendingEgressRequestStore
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.executors.harness import register_adapter
from worker.model_turn import HeldModelEgress, ResponsesFacade
from worker.model_turn.facade import _MAX_TURN_COMMANDS, FacadeTurnError
from worker.model_turn.rendezvous import ModelTurnRendezvous

_TASK = "tsk-agent"
_RUN_COMMAND = FacadeDescriptor(
    name="run_command",
    kind=BoundaryEventKind.STATE_ACCESS,
    interface=SANDBOX_EXECUTE_INTERFACE,
    tool_schema=json.dumps(
        {"type": "function", "name": "run_command", "parameters": {"type": "object"}}
    ),
    resolution=FacadeResolution.LOCAL_INLINE,
)
_SEARCH = FacadeDescriptor(
    name="web_search",
    kind=BoundaryEventKind.INVOCATION,
    interface=SEARCH_INTERFACE,
    tool_schema=json.dumps(
        {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
    ),
)


class _ScriptedEgress:
    """Returns one completion per model round and records every egress it ran."""

    def __init__(self, completions: list[ModelCompletion]) -> None:
        self._completions = completions
        self.seen: list[tuple[str, str, Any]] = []

    def run(
        self,
        task_id: str,
        correlation: str,
        request: Any,
        episode: str,
        dispatch_id: str | None,
    ) -> Any:
        self.seen.append((task_id, correlation, request))
        index = min(len(self.seen) - 1, len(self._completions) - 1)
        return self._completions[index]

    def reopen(self, task_id: str, episode: str) -> None:
        pass

    def close(self, task_id: str, episode: str) -> None:
        pass

    def refuse(self, task_id: str) -> None:
        pass

    def release(self, task_id: str) -> None:
        pass


class _RecordingSandbox(LocalSandboxExecutor):
    def __init__(self, denied: bool = False, unavailable: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._denied = denied
        self._unavailable = unavailable

    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
        if self._unavailable:
            raise SandboxUnavailable("the sandbox could not start a command")
        if self._denied:
            raise SandboxDenied("the capability is not this dispatch's authority")
        self.commands.append(command.argv)
        return SandboxCommandResult(
            exit_code=0, stdout=f"out:{command.argv[-1]}", stderr=""
        )


def _call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ModelToolCall:
    return ModelToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments))


def _facade(
    completions: list[ModelCompletion], sandbox: LocalSandboxExecutor | None
) -> tuple[ResponsesFacade, _ScriptedEgress, PendingEgressRequestStore, str]:
    egress = _ScriptedEgress(completions)
    pending = PendingEgressRequestStore()
    facade = ResponsesFacade(held_egress=cast(Any, egress), pending=pending)
    token = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND, _SEARCH], "dsp-1", sandbox
    )
    return facade, egress, pending, token


def test_a_command_is_run_locally_and_answered_in_the_same_turn() -> None:
    sandbox = _RecordingSandbox()
    facade, egress, pending, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "one"]}),),
            ),
            ModelCompletion(content="done"),
        ],
        sandbox,
    )

    output = facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == [("echo", "one")]
    assert output[0]["content"][0]["text"] == "done"
    # Nothing about the command reached control: no group, no stashed request.
    assert facade.take_captured_group(_TASK) is None
    assert pending.peek(_TASK, "c1") is None
    # The command's result was fed back into this turn's own message list.
    second_request = egress.seen[1][2]
    roles = [m["role"] for m in second_request.body["messages"]]
    assert roles[-2:] == ["assistant", "tool"]
    assert "out:one" in second_request.body["messages"][-1]["content"]


def test_many_commands_run_in_one_turn() -> None:
    sandbox = _RecordingSandbox()
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "a"]}, "c1"),),
            ),
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "b"]}, "c2"),),
            ),
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "c"]}, "c3"),),
            ),
            ModelCompletion(content="all three ran"),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == [("echo", "a"), ("echo", "b"), ("echo", "c")]
    # One held model egress per model round, and nothing extra for the commands.
    assert len(egress.seen) == 4
    assert facade.take_captured_group(_TASK) is None


def test_a_mediated_call_still_captures_after_the_commands() -> None:
    sandbox = _RecordingSandbox()
    facade, _, pending, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["ls"]}, "c1"),),
            ),
            ModelCompletion(
                content="",
                tool_calls=(_call("web_search", {"query": "flowmesh"}, "c2"),),
            ),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == [("ls",)]
    group = facade.take_captured_group(_TASK)
    assert group is not None and len(group.members) == 1
    assert group.members[0].kind is BoundaryEventKind.INVOCATION


def test_a_denied_command_is_reported_to_the_model_not_escalated() -> None:
    sandbox = _RecordingSandbox(denied=True)
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "x"]}),),
            ),
            ModelCompletion(content="understood"),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    result = egress.seen[1][2].body["messages"][-1]["content"]
    assert result.startswith("denied:")
    assert facade.take_captured_group(_TASK) is None


def test_an_agent_without_a_sandbox_is_told_it_has_none() -> None:
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["echo", "x"]}),),
            ),
            ModelCompletion(content="ok"),
        ],
        None,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    assert "denied" in egress.seen[1][2].body["messages"][-1]["content"]


def test_a_malformed_command_is_denied_without_running() -> None:
    sandbox = _RecordingSandbox()
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="", tool_calls=(_call("run_command", {"command": []}),)
            ),
            ModelCompletion(content="ok"),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == []
    assert "denied" in egress.seen[1][2].body["messages"][-1]["content"]


def test_an_unavailable_runtime_denies_rather_than_failing_the_turn() -> None:
    """A runtime that cannot start a command settles the action, not the request."""
    sandbox = _RecordingSandbox(unavailable=True)
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="", tool_calls=(_call("run_command", {"command": ["echo"]}),)
            ),
            ModelCompletion(content="understood"),
        ],
        sandbox,
    )

    output = facade.handle_turn(_TASK, token, {"input": "go"})

    assert output[0]["content"][0]["text"] == "understood"
    assert egress.seen[1][2].body["messages"][-1]["content"].startswith("denied:")


def test_a_turn_runs_no_more_than_the_command_cap() -> None:
    """The cap bounds commands, not rounds: one completion runs no unbounded batch."""
    sandbox = _RecordingSandbox()
    over_cap = tuple(
        _call("run_command", {"command": ["echo", str(i)]}, f"c{i}")
        for i in range(_MAX_TURN_COMMANDS + 5)
    )
    facade, _, _, token = _facade(
        [
            ModelCompletion(content="", tool_calls=over_cap),
            ModelCompletion(content="stopped"),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    assert len(sandbox.commands) == _MAX_TURN_COMMANDS


def test_a_native_tool_co_emitted_with_a_command_still_gets_a_result() -> None:
    """Every call the model emitted is answered, so the next request is well-formed."""
    sandbox = _RecordingSandbox()
    facade, egress, _, token = _facade(
        [
            ModelCompletion(
                content="",
                tool_calls=(
                    _call("run_command", {"command": ["ls"]}, "c1"),
                    _call("apply_patch", {"patch": "..."}, "c2"),
                ),
            ),
            ModelCompletion(content="done"),
        ],
        sandbox,
    )

    facade.handle_turn(_TASK, token, {"input": "go"})

    messages = egress.seen[1][2].body["messages"]
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert answered == {"c1", "c2"}


def test_a_turn_cancelled_during_a_command_proposes_no_later_round() -> None:
    rendezvous = ModelTurnRendezvous()
    pending = PendingEgressRequestStore()
    proposed: list[str] = []
    egressed: list[str] = []

    class _Sidecar:
        def egress_now(self, permit: MediatedOperationPermit) -> ModelCompletion:
            egressed.append(permit.call_correlation)
            return ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["make"]}),),
            )

    def propose(proposal: AgentModelTurnProposal) -> None:
        proposed.append(proposal.call_correlation)
        rendezvous.deliver_permit(
            _permit(proposal.agent_task_id, proposal.call_correlation)
        )

    held = HeldModelEgress(
        rendezvous=rendezvous,
        pending=pending,
        propose=propose,
        sidecar=cast(Any, _Sidecar()),
        timeout_sec=5.0,
    )
    facade = ResponsesFacade(held_egress=held, pending=pending)

    class _CancelledMidCommand(_RecordingSandbox):
        def execute(self, command: SandboxCommand) -> SandboxCommandResult:
            facade.refuse_episode(_TASK)
            facade.release_episode(_TASK)
            return super().execute(command)

    sandbox = _CancelledMidCommand()
    token = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND], "dsp-1", sandbox
    )

    with pytest.raises(FacadeTurnError, match="cancelled"):
        facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == [("make",)]
    assert len(proposed) == 1
    assert egressed == proposed
    assert pending.occurrences() == []
    assert not rendezvous.has_waiter(_TASK, proposed[0])


def test_a_round_during_the_harness_close_waits_for_the_release_unproposed() -> None:
    rendezvous = ModelTurnRendezvous()
    pending = PendingEgressRequestStore()
    proposed: list[str] = []
    order: list[str] = []

    class _Sidecar:
        def egress_now(self, permit: MediatedOperationPermit) -> ModelCompletion:
            return ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["make"]}),),
            )

    def propose(proposal: AgentModelTurnProposal) -> None:
        proposed.append(proposal.call_correlation)
        rendezvous.deliver_permit(
            _permit(proposal.agent_task_id, proposal.call_correlation)
        )

    held = HeldModelEgress(
        rendezvous=rendezvous,
        pending=pending,
        propose=propose,
        sidecar=cast(Any, _Sidecar()),
        timeout_sec=5.0,
    )
    facade = ResponsesFacade(held_egress=held, pending=pending)

    def close_the_harness_then_release() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            task_id == _TASK for task_id, _ in rendezvous._waiters
        ):
            time.sleep(0.005)
        order.append("harness exited")
        facade.release_episode(_TASK)

    class _GivenUpMidCommand(_RecordingSandbox):
        def execute(self, command: SandboxCommand) -> SandboxCommandResult:
            facade.refuse_episode(_TASK)
            threading.Thread(target=close_the_harness_then_release, daemon=True).start()
            return super().execute(command)

    sandbox = _GivenUpMidCommand()
    token = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND], "dsp-1", sandbox
    )

    with pytest.raises(FacadeTurnError, match="cancelled"):
        facade.handle_turn(_TASK, token, {"input": "go"})
    order.append("turn answered")

    assert order == ["harness exited", "turn answered"]
    assert len(proposed) == 1
    assert pending.occurrences() == []


class _Control:
    """Answers each proposed model turn with a permit over its digest, and egresses
    it as the sidecar does: fenced against the request the worker stashed."""

    def __init__(self, replies: Any) -> None:
        self.rendezvous = ModelTurnRendezvous()
        self.pending = PendingEgressRequestStore()
        self.proposals: list[AgentModelTurnProposal] = []
        self.answer_at_once = True
        self._replies = replies
        self._proposed = threading.Condition()
        self.facade = ResponsesFacade(
            held_egress=HeldModelEgress(
                rendezvous=self.rendezvous,
                pending=self.pending,
                propose=self._propose,
                sidecar=cast(Any, self),
                timeout_sec=5.0,
            ),
            pending=self.pending,
        )

    def _propose(self, proposal: AgentModelTurnProposal) -> None:
        with self._proposed:
            self.proposals.append(proposal)
            self._proposed.notify_all()
        if self.answer_at_once:
            self.answer(proposal)

    def answer(self, proposal: AgentModelTurnProposal) -> None:
        self.rendezvous.deliver_permit(
            _permit(proposal.agent_task_id, proposal.call_correlation).model_copy(
                update={"request_digest": proposal.request_digest}
            )
        )

    def await_proposals(self, count: int) -> None:
        with self._proposed:
            assert self._proposed.wait_for(lambda: len(self.proposals) >= count, 5)

    def egress_now(self, permit: MediatedOperationPermit) -> Any:
        request = self.pending.peek(permit.agent_task_id, permit.call_correlation)
        assert isinstance(request, ModelRequest)
        digest = model_request_digest(request.interface, request.url, request.body)
        if digest != permit.request_digest:
            return HeldEgressReject(reason="permit fence rejected: digest")
        return self._replies(permit.call_correlation, request)


@pytest.mark.parametrize("given_up", ["refused", "unregistered"])
def test_a_turn_given_up_mid_batch_runs_no_more_of_its_commands(given_up: str) -> None:
    batch = ModelCompletion(
        content="",
        tool_calls=tuple(
            _call("run_command", {"command": ["step", str(i)]}, call_id=f"c{i}")
            for i in range(4)
        ),
    )
    control = _Control(lambda *_: batch)
    facade = control.facade

    def release_once_the_next_round_waits() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not control.rendezvous.has_waiter(
            _TASK, "model:0:1"
        ):
            time.sleep(0.005)
        facade.release_episode(_TASK)

    class _GivenUpOnFirstCommand(_RecordingSandbox):
        def execute(self, command: SandboxCommand) -> SandboxCommandResult:
            facade.refuse_episode(_TASK)
            if given_up == "refused":
                threading.Thread(
                    target=release_once_the_next_round_waits, daemon=True
                ).start()
            else:
                facade.release_episode(_TASK)
                facade.unregister_episode(_TASK)
            return super().execute(command)

    sandbox = _GivenUpOnFirstCommand()
    token = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND], "dsp-1", sandbox
    )

    with pytest.raises(FacadeTurnError, match="cancelled"):
        facade.handle_turn(_TASK, token, {"input": "go"})

    assert sandbox.commands == [("step", "0")]
    assert [p.call_correlation for p in control.proposals] == ["model:0"]
    assert control.pending.occurrences() == []


def test_a_given_up_turn_egresses_nothing_once_its_task_registers_again() -> None:
    def reply(correlation: str, request: Any) -> ModelCompletion:
        if correlation != "model:0":
            return ModelCompletion(content="done")
        tag = "stale" if "stale" in json.dumps(request.body) else "retry"
        return ModelCompletion(
            content="", tool_calls=(_call("run_command", {"command": [tag]}),)
        )

    control = _Control(reply)
    control.answer_at_once = False
    facade = control.facade
    in_command = threading.Event()
    resume = threading.Event()

    class _Sandbox(_RecordingSandbox):
        def execute(self, command: SandboxCommand) -> SandboxCommandResult:
            if command.argv == ("stale",):
                in_command.set()
                assert resume.wait(5)
            return super().execute(command)

    turns: dict[str, Any] = {}

    def turn(name: str, token: str) -> threading.Thread:
        def run() -> None:
            try:
                turns[name] = facade.handle_turn(_TASK, token, {"input": name})
            except FacadeTurnError as exc:
                turns[name] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    first = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND], "dsp-1", _Sandbox()
    )
    stale = turn("stale", first)
    control.await_proposals(1)
    control.answer(control.proposals[0])
    assert in_command.wait(5)
    # The step raised: its turn is given up, and its retry lands on this worker.
    facade.refuse_episode(_TASK)
    facade.release_episode(_TASK)
    facade.unregister_episode(_TASK)
    second = facade.register_episode(
        _TASK, "http://up/v1", "m", [_RUN_COMMAND], "dsp-1", _Sandbox()
    )
    retry = turn("retry", second)
    control.await_proposals(2)
    control.answer(control.proposals[1])
    control.await_proposals(3)

    # The given-up turn's command returns while the retry waits on the same call.
    resume.set()
    stale.join(5)
    control.answer(control.proposals[2])
    retry.join(5)

    assert [p.call_correlation for p in control.proposals] == [
        "model:0",
        "model:0",
        "model:0:1",
    ]
    assert isinstance(turns["stale"], FacadeTurnError)
    assert turns["retry"][0]["content"][0]["text"] == "done"


def test_a_step_that_raises_gives_up_the_turn_still_running(tmp_path: Path) -> None:
    rendezvous = ModelTurnRendezvous()
    pending = PendingEgressRequestStore()
    step_ended = threading.Event()
    after_the_step: list[str] = []

    class _Sidecar:
        def egress_now(self, permit: MediatedOperationPermit) -> ModelCompletion:
            return ModelCompletion(
                content="",
                tool_calls=(_call("run_command", {"command": ["make"]}),),
            )

    def propose(proposal: AgentModelTurnProposal) -> None:
        if step_ended.is_set():
            after_the_step.append(proposal.call_correlation)
        rendezvous.deliver_permit(
            _permit(proposal.agent_task_id, proposal.call_correlation)
        )

    held = HeldModelEgress(
        rendezvous=rendezvous,
        pending=pending,
        propose=propose,
        sidecar=cast(Any, _Sidecar()),
        timeout_sec=5.0,
    )
    facade = ResponsesFacade(held_egress=held, pending=pending)
    commanding = threading.Event()

    class _SlowSandbox(_RecordingSandbox):
        def execute(self, command: SandboxCommand) -> SandboxCommandResult:
            commanding.set()
            step_ended.wait(5)
            return super().execute(command)

    turn_errors: list[BaseException] = []

    class _ReaderDied(_FakeAdapter):
        def __init__(self) -> None:
            super().__init__(HarnessResult(kind=HarnessResultKind.COMPLETION))
            self.turn: threading.Thread | None = None

        def start(self, activation_id, *, capsule, outcomes) -> HarnessResult:
            token = facade.register_episode(
                activation_id,
                "http://up/v1",
                "m",
                [_RUN_COMMAND],
                "dsp-1",
                _SlowSandbox(),
            )

            def turn() -> None:
                try:
                    facade.handle_turn(activation_id, token, {"input": "go"})
                except FacadeTurnError as exc:
                    turn_errors.append(exc)

            self.turn = threading.Thread(target=turn, daemon=True)
            self.turn.start()
            assert commanding.wait(5)
            raise RuntimeError("the Codex app-server closed its stdout")

    adapter = _ReaderDied()
    register_adapter("fake", lambda *_: adapter)
    lifecycle = MagicMock()
    lifecycle.responses_facade = facade
    msg = _dispatch_msg()
    executor = AgentEpisodeExecutor(make_worker_config(), lifecycle=lifecycle)

    with pytest.raises(RuntimeError, match="closed its stdout"):
        executor.run(msg, tmp_path)
    step_ended.set()
    assert adapter.turn is not None
    adapter.turn.join(5)

    assert adapter.cancelled == [msg.task_id]
    assert after_the_step == []
    assert len(turn_errors) == 1
    assert pending.occurrences() == []


def _permit(task_id: str, call_correlation: str) -> MediatedOperationPermit:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=task_id,
        call_correlation=call_correlation,
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
    )
