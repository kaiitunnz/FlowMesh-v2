"""Execution-locality split for fabric-served external tools.

The sidecar surface enforces the server-issued operation envelope — refusing an
interface it does not serve or a request beyond its issued budget without egressing —
the two localities yield the identical outcome for one request, and the deployment
policy keeps a keyed provider on server relay unless it is explicitly opted in.
"""

from typing import Any

from server.config import WebSearchConfig
from server.orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    ToolOutcome,
    ToolOutcomeStatus,
)
from server.services.tool_egress import (
    ColocatedSidecarCarriage,
    EgressLocality,
    EgressLocalityPolicy,
    ExternalToolSidecar,
    ServerRelayAdapter,
    ToolOperationEnvelope,
    ToolRequest,
    WorkerSidecarAdapter,
)
from shared.tools.providers import SearchResult


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


def _envelope(interface: str = SEARCH_INTERFACE, max_results: int = 5):
    return ToolOperationEnvelope(
        interface=interface,
        idempotency_key="idm-1",
        max_results=max_results,
        timeout_sec=1.0,
        result_char_cap=6000,
    )


def _request(interface: str = SEARCH_INTERFACE, query: str = "q", max_results: int = 5):
    return ToolRequest(interface=interface, query=query, max_results=max_results)


def test_sidecar_refuses_an_interface_it_does_not_serve_without_egress() -> None:
    provider = _StubProvider([SearchResult(title="a", url="u", snippet="s")])
    outcome = ExternalToolSidecar(provider).execute(
        _envelope(interface="filesystem/v1"), _request(interface="filesystem/v1")
    )
    assert outcome.status is ToolOutcomeStatus.UNAVAILABLE
    assert provider.calls == []


def test_sidecar_refuses_a_request_interface_outside_its_envelope() -> None:
    provider = _StubProvider([SearchResult(title="a", url="u", snippet="s")])
    outcome = ExternalToolSidecar(provider).execute(
        _envelope(), _request(interface="other/v1")
    )
    assert outcome.status is ToolOutcomeStatus.UNAVAILABLE
    assert provider.calls == []


def test_sidecar_refuses_an_over_budget_request_without_egress() -> None:
    provider = _StubProvider([SearchResult(title="a", url="u", snippet="s")])
    outcome = ExternalToolSidecar(provider).execute(
        _envelope(max_results=2), _request(max_results=10)
    )
    assert outcome.status is ToolOutcomeStatus.QUOTA
    assert provider.calls == []


def test_sidecar_normalizes_a_successful_search() -> None:
    provider = _StubProvider(
        [SearchResult(title="GPT", url="https://openai.com/x", snippet="new")]
    )
    outcome = ExternalToolSidecar(provider).execute(
        _envelope(), _request(query="latest model", max_results=1)
    )
    assert outcome.status is ToolOutcomeStatus.SUCCESS
    assert "GPT" in outcome.value and "https://openai.com/x" in outcome.value
    assert outcome.provenance[0] == {"title": "GPT", "url": "https://openai.com/x"}
    assert provider.calls == [("latest model", 1)]


def test_server_relay_and_worker_sidecar_yield_the_identical_outcome() -> None:
    results = [SearchResult(title="T", url="https://x/y", snippet="S")]
    server = ServerRelayAdapter(ExternalToolSidecar(_StubProvider(results)))
    worker = WorkerSidecarAdapter(
        ColocatedSidecarCarriage(ExternalToolSidecar(_StubProvider(results)))
    )
    envelope, request = _envelope(), _request(query="same query", max_results=3)
    assert server.locality is EgressLocality.SERVER_RELAY
    assert worker.locality is EgressLocality.WORKER_SIDECAR
    assert server.execute(envelope, request) == worker.execute(envelope, request)


def _policy(**cfg: Any) -> tuple[EgressLocalityPolicy, object, object]:
    server, worker = object(), object()
    return EgressLocalityPolicy(WebSearchConfig(**cfg), server, worker), server, worker  # type: ignore[arg-type]


def test_policy_defaults_to_server_relay() -> None:
    policy, server, _ = _policy(provider="duckduckgo")
    assert policy.select() is server


def test_policy_routes_any_provider_to_the_worker_sidecar_when_configured() -> None:
    keyless, _, worker = _policy(
        provider="duckduckgo", egress_locality="worker_sidecar"
    )
    assert keyless.select() is worker
    keyed, _, worker = _policy(
        provider="serper", api_key="k", egress_locality="worker_sidecar"
    )
    assert keyed.select() is worker


def test_a_worker_carriage_delivers_the_surfaces_enforced_outcome() -> None:
    provider = _StubProvider([SearchResult(title="a", url="u", snippet="s")])
    carriage = ColocatedSidecarCarriage(ExternalToolSidecar(provider))
    over_budget = carriage(_envelope(max_results=1), _request(max_results=9))
    assert isinstance(over_budget, ToolOutcome)
    assert over_budget.status is ToolOutcomeStatus.QUOTA
    assert provider.calls == []
