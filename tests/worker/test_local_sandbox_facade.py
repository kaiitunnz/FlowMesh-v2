"""The held turn answers an agent's commands locally, without a control round trip.

A ``run_command`` call the model emits is resolved by the worker inside the turn: it is
never captured as a turn-group member, never stashed for the egress path, and never
settles through control. Only the model calls between commands are mediated, as they
already were.
"""

import json
from typing import Any, cast

from shared.harness import BoundaryEventKind
from shared.sandbox import (
    SANDBOX_EXECUTE_INTERFACE,
    LocalSandboxExecutor,
    SandboxCommand,
    SandboxCommandResult,
    SandboxDenied,
)
from shared.tools.facade import FacadeDescriptor, FacadeResolution
from shared.tools.model.schema import ModelCompletion, ModelToolCall
from shared.tools.search.schema import SEARCH_INTERFACE
from worker.egress import PendingEgressRequestStore
from worker.model_turn import ResponsesFacade

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

    def run(self, task_id: str, correlation: str, request: Any) -> Any:
        self.seen.append((task_id, correlation, request))
        index = min(len(self.seen) - 1, len(self._completions) - 1)
        return self._completions[index]


class _RecordingSandbox(LocalSandboxExecutor):
    def __init__(self, denied: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._denied = denied

    def execute(self, command: SandboxCommand) -> SandboxCommandResult:
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
        _TASK, "http://up/v1", "m", [_RUN_COMMAND, _SEARCH], sandbox
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
