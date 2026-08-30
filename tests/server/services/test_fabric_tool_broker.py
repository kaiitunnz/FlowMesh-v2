"""The fabric tool broker executes a search boundary and normalizes a typed outcome.

The broker runs off the agent's lane, maps a provider fault to a typed ``ToolOutcome``,
bounds an episode's search budget, and settles the durable envelope through the runtime
callback — never an empty success and never the model settler.
"""

from typing import Any

from server.config import WebSearchConfig
from server.orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    GrantSnapshot,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from server.services.fabric_tool_broker import FabricToolBroker
from server.services.search_providers import (
    SearchQuotaExceeded,
    SearchResult,
    SearchTimeout,
    SearchUnavailable,
)
from shared.harness import BoundaryEventKind


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


class _StubProvider:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int, timeout_sec: float):
        self.calls.append((query, max_results))
        if self._error is not None:
            raise self._error
        return self._results


def _broker(provider: Any, **cfg: Any) -> tuple[FabricToolBroker, list[tuple]]:
    settled: list[tuple] = []
    broker = FabricToolBroker(
        WebSearchConfig(**cfg), lambda t, c, v: settled.append((t, c, v)), provider
    )
    return broker, settled


def _outcome(settled: list[tuple]) -> ToolOutcome:
    assert len(settled) == 1
    return ToolOutcome.model_validate_json(settled[0][2])


def test_success_normalizes_results_with_provenance() -> None:
    provider = _StubProvider(
        [
            SearchResult(
                title="GPT-5.6 Sol", url="https://openai.com/x", snippet="new"
            ),
            SearchResult(title="Other", url="https://e.com/y", snippet="more"),
        ]
    )
    broker, settled = _broker(provider)
    broker._run(_env('{"query": "latest openai model 2026", "max_results": 2}'))
    outcome = _outcome(settled)
    assert outcome.status is ToolOutcomeStatus.SUCCESS
    assert "GPT-5.6 Sol" in outcome.value and "https://openai.com/x" in outcome.value
    assert outcome.provenance[0] == {
        "title": "GPT-5.6 Sol",
        "url": "https://openai.com/x",
    }
    assert provider.calls == [("latest openai model 2026", 2)]


def test_empty_results_is_a_clear_success_not_a_hallucination() -> None:
    broker, settled = _broker(_StubProvider([]))
    broker._run(_env('{"query": "obscure"}'))
    outcome = _outcome(settled)
    assert outcome.status is ToolOutcomeStatus.SUCCESS and "No results" in outcome.value


def test_provider_faults_map_to_typed_outcomes() -> None:
    for error, status in (
        (SearchTimeout("t"), ToolOutcomeStatus.TIMEOUT),
        (SearchQuotaExceeded("q"), ToolOutcomeStatus.QUOTA),
        (SearchUnavailable("u"), ToolOutcomeStatus.UNAVAILABLE),
    ):
        broker, settled = _broker(_StubProvider(error=error))
        broker._run(_env('{"query": "x"}'))
        assert _outcome(settled).status is status


def test_per_episode_budget_exhausts_to_quota() -> None:
    broker, settled = _broker(_StubProvider([]), max_calls=2)
    for _ in range(3):
        settled.clear()
        broker._run(_env('{"query": "x"}'))
    assert _outcome(settled).status is ToolOutcomeStatus.QUOTA


def test_an_unknown_interface_is_unavailable_never_executed() -> None:
    provider = _StubProvider([SearchResult(title="a", url="u", snippet="s")])
    broker, settled = _broker(provider)
    broker._run(_env('{"query": "x"}', interface="mystery/v1"))
    assert _outcome(settled).status is ToolOutcomeStatus.UNAVAILABLE
    assert provider.calls == []


def test_max_results_is_capped_to_config() -> None:
    provider = _StubProvider([])
    broker, _ = _broker(provider, max_results=3)
    broker._run(_env('{"query": "x", "max_results": 99}'))
    assert provider.calls == [("x", 3)]


_DDG_HTML = """
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fsol">
GPT-5.6 <b>Sol</b></a>
<a class="result__snippet" href="x">OpenAI's newest <b>model</b>.</a>
"""


def test_duckduckgo_provider_parses_and_unwraps(monkeypatch: Any) -> None:
    import server.services.search_providers as mod

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
    import server.services.search_providers as mod

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
