"""The fabric tool broker terminalizes a server-captured tool boundary.

The broker holds no provider client and performs no egress: external-tool egress runs
only in a worker's mediated-egress sidecar. A boundary reaching the broker (a
server-captured, no-worker-origin facade boundary) has no in-server egress, so the
broker settles it off the agent's lane as a typed unavailable outcome. This file also
covers the retained search provider backends the sidecar egresses through.
"""

from dataclasses import dataclass
from typing import Any

from server.config import WebSearchConfig
from server.orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    GrantSnapshot,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from server.tools.fabric_tool_broker import FabricToolBroker
from shared.harness import BoundaryEventKind
from shared.tools.search.providers import (
    SearchQuotaExceeded,
    SearchTimeout,
)


def _env(
    payload: str | None, *, task: str = "tsk-1", interface: str = SEARCH_INTERFACE
):
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface=interface,
        invocation_id="inv-1",
        task_id=task,
        activation_id="act-1",
        call_correlation="c0",
        request_payload=payload,
        grant_snapshot=GrantSnapshot(grant_id="agr-1"),
    )


def _broker() -> tuple[FabricToolBroker, list[tuple[str, str, Any]]]:
    settled: list[tuple[str, str, Any]] = []
    broker = FabricToolBroker.build(
        WebSearchConfig(),
        lambda t, c, v: settled.append((t, c, v)),
    )
    return broker, settled


@dataclass(frozen=True)
class _ProviderCfg:
    """A minimal provider-config double (the worker builds the real binding)."""

    provider: str
    api_key: str | None = None


def _carrier_outcome(carrier: Any) -> ToolOutcome:
    """The typed outcome inside an inline settle carrier."""
    return ToolOutcome.model_validate_json(carrier.value)


def test_captured_boundary_has_no_in_server_egress() -> None:
    broker, settled = _broker()
    broker._run(_env('{"query": "latest openai model 2026", "max_results": 2}'))
    assert len(settled) == 1
    task_id, call, carrier = settled[0]
    assert (task_id, call) == ("tsk-1", "c0")
    outcome = _carrier_outcome(carrier)
    assert outcome.status is ToolOutcomeStatus.UNAVAILABLE
    assert "no in-server egress" in outcome.value


def test_build_holds_no_provider_client() -> None:
    broker, _ = _broker()
    # The control-only broker constructs with no provider or carriage; egress is the
    # worker's mediated-egress sidecar.
    assert not hasattr(broker, "_policy")
    assert not hasattr(broker, "_provider")


_DDG_HTML = """
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fsol">
GPT-5.6 <b>Sol</b></a>
<a class="result__snippet" href="x">OpenAI's newest <b>model</b>.</a>
"""


def test_duckduckgo_provider_parses_and_unwraps(monkeypatch: Any) -> None:
    import shared.tools.search.providers as mod

    class _Resp:
        status_code = 200
        text = _DDG_HTML

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _Resp())
    results = mod.DuckDuckGoProvider().search("q", max_results=5, timeout_sec=1.0)
    assert len(results) == 1
    assert results[0].title == "GPT-5.6 Sol"
    assert results[0].url == "https://openai.com/sol"
    assert results[0].snippet == "OpenAI's newest model."


def test_duckduckgo_provider_maps_http_faults(monkeypatch: Any) -> None:
    import shared.tools.search.providers as mod

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.text = ""

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _Resp(429))
    try:
        mod.DuckDuckGoProvider().search("q", max_results=5, timeout_sec=1.0)
        raise AssertionError("expected a quota fault")
    except SearchQuotaExceeded:
        pass

    def _timeout(*a: Any, **k: Any):
        raise mod.requests.Timeout("slow")

    monkeypatch.setattr(mod.requests, "post", _timeout)
    try:
        mod.DuckDuckGoProvider().search("q", max_results=5, timeout_sec=1.0)
        raise AssertionError("expected a timeout fault")
    except SearchTimeout:
        pass


def test_serper_provider_parses_organic_and_maps_faults(monkeypatch: Any) -> None:
    import shared.tools.search.providers as mod

    class _Resp:
        def __init__(self, code: int, payload: dict[str, Any]) -> None:
            self.status_code = code
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    seen: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["key"] = kwargs["headers"].get("X-API-KEY")
        return _Resp(
            200,
            {"organic": [{"title": "T", "link": "https://x/y", "snippet": "S"}]},
        )

    monkeypatch.setattr(mod.requests, "post", _post)
    results = mod.SerperProvider("k-123").search("q", max_results=3, timeout_sec=1.0)
    assert seen["url"] == "https://google.serper.dev/search" and seen["key"] == "k-123"
    assert results == [mod.SearchResult(title="T", url="https://x/y", snippet="S")]

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _Resp(429, {}))
    try:
        mod.SerperProvider("k").search("q", max_results=3, timeout_sec=1.0)
        raise AssertionError("expected a quota fault")
    except SearchQuotaExceeded:
        pass


def test_build_search_provider_selects_by_config() -> None:
    from shared.tools.search.providers import (
        DuckDuckGoProvider,
        SerperProvider,
        build_search_provider,
    )

    assert isinstance(
        build_search_provider(_ProviderCfg("duckduckgo")),
        DuckDuckGoProvider,
    )
    keyed = build_search_provider(_ProviderCfg("serper", "k-abc"))
    assert isinstance(keyed, SerperProvider)
    for bad in (
        _ProviderCfg("serper"),  # keyed provider with no key
        _ProviderCfg("nope"),  # unknown provider
    ):
        try:
            build_search_provider(bad)
            raise AssertionError("expected a config error")
        except ValueError:
            pass


def test_lazy_provider_defers_a_missing_key_to_first_search() -> None:
    from shared.tools.search.providers import LazySearchProvider

    # A keyed provider with no key must not fail at construction — only on egress,
    # so a deployment that egresses only off-server never builds it on the server.
    lazy = LazySearchProvider(_ProviderCfg("serper", None))
    try:
        lazy.search("q", max_results=1, timeout_sec=1.0)
        raise AssertionError("expected the missing key to raise on first search")
    except ValueError:
        pass
