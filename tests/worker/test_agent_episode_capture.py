"""Worker-side capture of a worker-originated egress request off a returned boundary."""

from shared.harness import BoundaryEventKind, BoundaryRequest, HarnessResult
from shared.harness.adapter import EpisodeModelBinding, HarnessResultKind
from shared.tasks.specs.misc import ModelBindingMode
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    ModelRequest,
    model_request_digest,
)
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    parse_search_request,
    tool_request_digest,
)
from worker.executors.agent_episode_executor import AgentEpisodeExecutor
from worker.lifecycle import PendingEgressRequestStore

_TASK = "tsk-agent"
_OPENAI = EpisodeModelBinding(
    mode=ModelBindingMode.OPENAI, url="http://up/v1", model="m"
)


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
    store = PendingEgressRequestStore()
    result = AgentEpisodeExecutor._capture_local_request(
        store, _TASK, _boundary('{"query": "weather", "max_results": 3}'), None
    )
    req = result.request
    assert req is not None
    assert req.request_payload is None
    assert req.request_digest == tool_request_digest(SEARCH_INTERFACE, "weather", 3)
    stored = store.peek(_TASK, "m0")
    assert isinstance(stored, ToolRequest)
    assert stored.query == "weather" and stored.max_results == 3


def test_model_boundary_is_stripped_and_stored_for_openai_binding() -> None:
    store = PendingEgressRequestStore()
    result = AgentEpisodeExecutor._capture_local_request(
        store, _TASK, _boundary("summarize this", interface=MODEL_INTERFACE), _OPENAI
    )
    req = result.request
    assert req is not None
    assert req.request_payload is None
    assert req.request_digest == model_request_digest(
        MODEL_INTERFACE, "http://up/v1", "m", "summarize this"
    )
    stored = store.peek(_TASK, "m0")
    assert isinstance(stored, ModelRequest)
    assert stored.prompt == "summarize this"


def test_model_boundary_passes_through_without_external_binding() -> None:
    store = PendingEgressRequestStore()
    canned = EpisodeModelBinding(mode=ModelBindingMode.CANNED)
    original = _boundary("do something", interface=MODEL_INTERFACE)
    result = AgentEpisodeExecutor._capture_local_request(store, _TASK, original, canned)
    assert result.request is not None
    assert result.request.request_payload == "do something"
    assert result.request.request_digest is None
    assert store.peek(_TASK, "m0") is None


def test_boundary_without_payload_passes_through() -> None:
    store = PendingEgressRequestStore()
    original = _boundary(None)
    result = AgentEpisodeExecutor._capture_local_request(store, _TASK, original, None)
    assert result.request is not None
    assert result.request.request_digest is None
    assert store.peek(_TASK, "m0") is None


def test_parse_search_request_accepts_object_and_bare_string() -> None:
    assert parse_search_request('{"query": "x", "max_results": 2}').max_results == 2
    bare = parse_search_request("just a query")
    assert bare.query == "just a query" and bare.max_results >= 1
