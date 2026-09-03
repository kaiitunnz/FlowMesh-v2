"""The FabricToolBroker — the authoritative control path for fabric-served tools.

The broker is the control home for mediated tool invocations (today ``search/v1``). It
is not a semantic authority over the ledger and holds no durable store: the engine
validates and records the invocation, the runtime submits the durable envelope, and the
broker draws down the episode budget, issues a bounded operation envelope, selects an
execution locality under deployment policy, and hands the operation to that adapter off
the agent's lane. It hands the adapter's typed ``ToolOutcome`` back to be terminalized
durably before the agent resumes. Its per-episode budget and in-flight set are a
rebuildable projection of ledger facts, restored by re-submission on restart, never
persisted here.
"""

import json
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ..config import WebSearchConfig
from ..orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from .search_providers import SearchProvider, build_search_provider
from .tool_egress import (
    ColocatedSidecarCarriage,
    EgressLocalityPolicy,
    ExternalToolSidecar,
    ServerRelayAdapter,
    ToolOperationEnvelope,
    ToolRequest,
    WorkerSidecarAdapter,
)

# (task_id, call_correlation, serialized ToolOutcome) — the runtime settles it durably.
SettleCallback = Callable[[str, str, str], None]


class FabricToolBroker:
    """Draw down budget, authorize a bounded operation, and route it to a locality."""

    def __init__(
        self,
        config: WebSearchConfig,
        settle: SettleCallback,
        policy: EgressLocalityPolicy,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = config
        self._settle = settle
        self._policy = policy
        self._log = logger or logging.getLogger("fabric-tool-broker")
        self._pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="fabric-tool-broker"
        )
        self._lock = threading.Lock()
        # Per-episode call budget. Best-effort: it is an in-memory count reset on
        # restart, so an episode may exceed the budget across a crash — never
        # exactly-once accounting, by design for this demo.
        self._calls: dict[str, int] = {}

    @classmethod
    def build(
        cls,
        config: WebSearchConfig,
        settle: SettleCallback,
        provider: SearchProvider | None = None,
        logger: logging.Logger | None = None,
    ) -> "FabricToolBroker":
        """Wire the default localities around the configured provider.

        The co-located worker sidecar shares the configured provider, so no credential
        crosses a wire in-process; the policy gate governs whether a keyed provider is
        eligible for worker-sidecar egress at all.
        """
        log = logger or logging.getLogger("fabric-tool-broker")
        prov = provider or build_search_provider(config)
        server = ServerRelayAdapter(ExternalToolSidecar(prov, log))
        worker = WorkerSidecarAdapter(
            ColocatedSidecarCarriage(ExternalToolSidecar(prov, log))
        )
        return cls(config, settle, EgressLocalityPolicy(config, server, worker), log)

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
        request = ToolRequest(
            interface=env.interface, query=query, max_results=max_results
        )
        envelope = ToolOperationEnvelope(
            interface=env.interface,
            idempotency_key=env.idempotency_key,
            max_results=max_results,
            timeout_sec=self._cfg.timeout_sec,
            result_char_cap=self._cfg.result_char_cap,
        )
        adapter = self._policy.select()
        self._log.info(
            "fabric search interface=%s inv=%s locality=%s query=%r",
            env.interface,
            env.invocation_id,
            adapter.locality.value,
            query,
        )
        return adapter.execute(envelope, request)

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
