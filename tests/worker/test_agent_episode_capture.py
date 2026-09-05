"""Worker-side capture of a worker-originated tool request off the returned boundary."""

from shared.harness import BoundaryEventKind, BoundaryRequest, HarnessResult
from shared.harness.adapter import HarnessResultKind
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    parse_search_request,
    tool_request_digest,
)
from worker.executors import pending_tool_request
from worker.executors.agent_episode_executor import AgentEpisodeExecutor

_TASK = "tsk-agent"


def _boundary(
    payload: str | None, *, interface: str = SEARCH_INTERFACE
) -> HarnessResult:
    return HarnessResult(
        kind=HarnessResultKind.BOUNDARY,
        request=BoundaryRequest(
            kind=BoundaryEventKind.INVOCATION,
            call_correlation="m0",
            interface=interface,
            request_payload=payload,
        ),
    )


def test_search_boundary_is_stripped_and_stored() -> None:
    result = AgentEpisodeExecutor._capture_local_request(
        _TASK, _boundary('{"query": "weather", "max_results": 3}')
    )
    req = result.request
    assert req is not None
    assert req.request_payload is None
    assert req.requested_max_results == 3
    assert req.request_digest == tool_request_digest(SEARCH_INTERFACE, "weather", 3)
    stored = pending_tool_request.take(_TASK, "m0")
    assert stored is not None and stored.query == "weather" and stored.max_results == 3


def test_non_search_boundary_passes_through() -> None:
    original = _boundary("do something", interface="model")
    result = AgentEpisodeExecutor._capture_local_request(_TASK, original)
    assert result.request is not None
    assert result.request.request_payload == "do something"
    assert result.request.request_digest is None
    assert pending_tool_request.peek(_TASK, "m0") is None


def test_boundary_without_payload_passes_through() -> None:
    original = _boundary(None)
    result = AgentEpisodeExecutor._capture_local_request(_TASK, original)
    assert result.request is not None
    assert result.request.request_digest is None
    assert pending_tool_request.peek(_TASK, "m0") is None


def test_parse_search_request_accepts_object_and_bare_string() -> None:
    assert parse_search_request('{"query": "x", "max_results": 2}').max_results == 2
    bare = parse_search_request("just a query")
    assert bare.query == "just a query" and bare.max_results >= 1
