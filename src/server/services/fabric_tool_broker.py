"""The FabricToolBroker — a dedicated actor that executes fabric-served tool boundaries.

The broker is the execution home for mediated tool invocations (today ``search/v1``). It
is not a semantic authority and holds no durable store: the engine validates and records
the invocation, the runtime submits the durable envelope, and the broker dispatches it
off the agent's lane, normalizes the provider's answer into a typed ``ToolOutcome``, and
hands that back to be terminalized durably before the agent resumes. Its per-episode
budget and in-flight set are a rebuildable projection of ledger facts, restored by
re-submission on restart, never persisted here.
"""

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ..config import WebSearchConfig
from ..orchestration.tool_dispatch import (
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from .search_providers import (
    SearchProvider,
    SearchQuotaExceeded,
    SearchResult,
    SearchTimeout,
    SearchUnavailable,
    build_search_provider,
)

SEARCH_INTERFACE = "search/v1"

# (task_id, call_correlation, serialized ToolOutcome) — the runtime settles it durably.
SettleCallback = Callable[[str, str, str], None]


class FabricToolBroker:
    """Execute a fabric-served tool boundary off-lane and return a typed outcome."""

    def __init__(
        self,
        config: WebSearchConfig,
        settle: SettleCallback,
        provider: SearchProvider | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = config
        self._settle = settle
        self._provider = provider or build_search_provider(config.provider)
        self._log = logger or logging.getLogger("fabric-tool-broker")
        self._pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="fabric-tool-broker"
        )
        self._lock = threading.Lock()
        self._calls: dict[str, int] = {}  # per-episode call budget, rebuildable

    def submit(self, env: ToolInvocationEnvelope) -> None:
        """Accept a recorded tool boundary and dispatch it off the agent's lane."""
        self._pool.submit(self._run, env)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    def _run(self, env: ToolInvocationEnvelope) -> None:
        try:
            outcome = self._dispatch(env)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a broker fault is a typed tool outcome
            self._log.warning("fabric tool dispatch failed: %s", exc)
            outcome = ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value=f"the {env.interface} tool is unavailable",
            )
        self._settle(env.task_id, env.call_correlation, outcome.model_dump_json())

    def _dispatch(self, env: ToolInvocationEnvelope) -> ToolOutcome:
        if env.interface != SEARCH_INTERFACE:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value=f"no handler for interface {env.interface}",
            )
        if not self._charge(env.task_id):
            return ToolOutcome(
                status=ToolOutcomeStatus.QUOTA,
                value=f"search budget exhausted ({self._cfg.max_calls} calls/episode)",
            )
        query, max_results = self._parse_request(env.request_payload)
        if not query:
            return ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS, value="No query supplied."
            )
        self._log.info(
            "fabric search interface=%s inv=%s query=%r",
            env.interface,
            env.invocation_id,
            query,
        )
        try:
            results = self._provider.search(
                query, max_results=max_results, timeout_sec=self._cfg.timeout_sec
            )
        except SearchTimeout:
            return ToolOutcome(
                status=ToolOutcomeStatus.TIMEOUT, value="the web search timed out"
            )
        except SearchQuotaExceeded:
            return ToolOutcome(
                status=ToolOutcomeStatus.QUOTA, value="the search provider rate-limited"
            )
        except SearchUnavailable:
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the search provider was unreachable",
            )
        return self._normalize(query, results)

    def _charge(self, task_id: str) -> bool:
        with self._lock:
            used = self._calls.get(task_id, 0)
            if used >= self._cfg.max_calls:
                return False
            self._calls[task_id] = used + 1
            return True

    def _parse_request(self, payload: str | None) -> tuple[str, int]:
        cap = self._cfg.max_results
        if not payload:
            return "", cap
        try:
            args = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return payload.strip(), cap
        if not isinstance(args, dict):
            return "", cap
        query = args.get("query")
        requested = args.get("max_results")
        n = min(int(requested), cap) if isinstance(requested, int) else cap
        return (query.strip() if isinstance(query, str) else ""), max(1, n)

    def _normalize(self, query: str, results: list[SearchResult]) -> ToolOutcome:
        if not results:
            return ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS, value=f"No results for {query!r}."
            )
        blocks: list[str] = []
        provenance: list[dict[str, str]] = []
        for i, r in enumerate(results, 1):
            blocks.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
            provenance.append({"title": r.title, "url": r.url})
        value = "\n\n".join(blocks)[: self._cfg.result_char_cap]
        return ToolOutcome(
            status=ToolOutcomeStatus.SUCCESS,
            value=value,
            provenance=tuple(provenance),
        )
