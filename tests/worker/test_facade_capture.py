"""Capturing a held model turn's facade calls into a control-recordable group."""

from shared.harness import BoundaryEventKind
from shared.tools.facade import FacadeCompletionMode, FacadeDescriptor
from shared.tools.model.schema import ModelToolCall
from shared.tools.search.schema import SEARCH_INTERFACE, tool_request_digest
from worker.facade_capture import (
    build_facade_capture,
    partition_facade_calls,
    turn_base,
)

_SEARCH = FacadeDescriptor(
    name="web_search",
    kind=BoundaryEventKind.INVOCATION,
    interface=SEARCH_INTERFACE,
    tool_schema="{}",
)
_SPAWN = FacadeDescriptor(
    name="spawn_agent", kind=BoundaryEventKind.SPAWN, tool_schema="{}"
)
_DESCRIPTORS = [_SEARCH, _SPAWN]
_TASK = "tsk-agent"


def _call(name: str, arguments: str, call_id: str = "c0") -> ModelToolCall:
    return ModelToolCall(call_id=call_id, name=name, arguments=arguments)


def test_partition_separates_facade_from_native_calls() -> None:
    calls = (
        _call("web_search", '{"query": "x"}'),
        _call("native_fn", "{}"),
        _call("spawn_agent", '{"region": "worker"}'),
    )
    facade, other = partition_facade_calls(calls, _DESCRIPTORS)
    assert [c.name for c in facade] == ["web_search", "spawn_agent"]
    assert [c.name for c in other] == ["native_fn"]


def test_search_member_carries_its_digest_and_stashes_the_request() -> None:
    facade = [_call("web_search", '{"query": "weather", "max_results": 5}', "call0")]
    capture = build_facade_capture(_TASK, facade, _DESCRIPTORS, turn_base=0)

    assert capture.group.group_id == f"{_TASK}:0"
    (member,) = capture.group.members
    assert member.kind is BoundaryEventKind.INVOCATION
    assert member.completion_mode is FacadeCompletionMode.AWAIT_OUTCOME
    assert member.call_correlation == f"{_TASK}:0:0"
    assert member.harness_call_id == "call0"
    assert member.interface_or_region == SEARCH_INTERFACE
    # The digest matches the worker egress's own digest of the stashed request, so the
    # fence agrees; the raw request is kept in custody, never on the member.
    assert member.request_digest == tool_request_digest(SEARCH_INTERFACE, "weather", 5)
    assert member.request_payload is None
    ((correlation, request),) = capture.stashes
    assert correlation == f"{_TASK}:0:0"
    assert request.query == "weather" and request.max_results == 5


def test_spawn_member_carries_args_and_region_without_a_digest() -> None:
    facade = [_call("spawn_agent", '{"region": "researcher", "args": {}}', "s0")]
    capture = build_facade_capture(_TASK, facade, _DESCRIPTORS, turn_base=1)
    (member,) = capture.group.members
    assert member.kind is BoundaryEventKind.SPAWN
    assert member.completion_mode is FacadeCompletionMode.ADMIT_AND_CLOSE
    assert member.interface_or_region == "researcher"
    assert member.request_payload == '{"region": "researcher", "args": {}}'
    assert member.request_digest is None
    assert capture.stashes == ()  # a spawn is control-admitted, never worker egress


def test_mixed_group_orders_members_and_stashes_only_searches() -> None:
    facade = [
        _call("web_search", '{"query": "a"}', "c1"),
        _call("spawn_agent", '{"region": "worker"}', "c2"),
        _call("web_search", '{"query": "b"}', "c3"),
    ]
    capture = build_facade_capture(_TASK, facade, _DESCRIPTORS, turn_base=2)
    kinds = [m.kind for m in capture.group.members]
    assert kinds == [
        BoundaryEventKind.INVOCATION,
        BoundaryEventKind.SPAWN,
        BoundaryEventKind.INVOCATION,
    ]
    assert [c for c, _ in capture.stashes] == [f"{_TASK}:2:0", f"{_TASK}:2:2"]


def test_turn_base_counts_settled_outputs() -> None:
    history = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "c1", "name": "web_search"},
        {"type": "function_call_output", "call_id": "c1", "output": "r"},
        {"type": "function_call_output", "call_id": "c2", "output": "r2"},
    ]
    assert turn_base(history) == 2
    assert turn_base("not a list") == 0
