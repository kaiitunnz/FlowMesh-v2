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
from dataclasses import dataclass

from shared.tools.search.providers import LazySearchProvider, SearchProvider

from ..config import WebSearchConfig
from ..orchestration.tool_dispatch import (
    SEARCH_INTERFACE,
    ToolInvocationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from .tool_egress import (
    AmbiguousDelivery,
    CarriageResult,
    ColocatedSidecarCarriage,
    EgressLocalityPolicy,
    ExecutionTransport,
    ExternalToolSidecar,
    OutcomeCarrier,
    ServerRelayAdapter,
    ToolOperationEnvelope,
    ToolRequest,
    WorkerSidecarAdapter,
    inline_outcome,
)

# (task_id, call_correlation, carrier) — an inline control datum or a reference-backed
# manifest the runtime settles durably; the broker never assembles a result body.
SettleCallback = Callable[[str, str, OutcomeCarrier], None]
# (task_id, call_correlation) -> re-dispatched; holds the boundary pending to re-drive.
RedispatchCallback = Callable[[str, str], bool]

# Bounded retry policy for an ambiguous delivery: a lost reply re-drives the same
# logical operation under its idm-* at most this many physical attempts before it
# terminalizes with an ambiguity audit outcome. Each attempt is itself deadline-bounded.
_MAX_DELIVERY_ATTEMPTS = 3


@dataclass
class _Recovery:
    """Per logical operation recovery state across physical delivery attempts."""

    attempts: int = 0
    charged: bool = False


class FabricToolBroker:
    """Draw down budget, authorize a bounded operation, and route it to a locality."""

    def __init__(
        self,
        config: WebSearchConfig,
        settle: SettleCallback,
        policy: EgressLocalityPolicy,
        logger: logging.Logger | None = None,
        redispatch: RedispatchCallback | None = None,
    ) -> None:
        self._cfg = config
        self._settle = settle
        self._redispatch = redispatch
        self._policy = policy
        self._log = logger or logging.getLogger("fabric-tool-broker")
        self._pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="fabric-tool-broker"
        )
        self._lock = threading.Lock()
        # Per-episode call budget. Best-effort: it is an in-memory count reset on
        # restart, so an episode may exceed the budget across a crash — never
        # exactly-once accounting.
        self._calls: dict[str, int] = {}
        # Per logical operation recovery state, keyed by (task_id, call_correlation): a
        # re-drive reuses the same authority/quota accounting and bounds its attempts.
        self._recovery: dict[tuple[str, str], _Recovery] = {}

    @classmethod
    def build(
        cls,
        config: WebSearchConfig,
        settle: SettleCallback,
        provider: SearchProvider | None = None,
        logger: logging.Logger | None = None,
        worker_carriage: ExecutionTransport | None = None,
        redispatch: RedispatchCallback | None = None,
    ) -> "FabricToolBroker":
        """Wire the localities around the configured provider.

        The server relay egresses in-server. The worker-sidecar locality uses
        ``worker_carriage`` when a remote carriage is configured — real off-server
        egress to a worker sidecar — otherwise a co-located sidecar sharing the server's
        provider, so no credential crosses a wire in-process. The provider is built
        lazily, so a deployment that only egresses off-server never constructs it and
        need not carry its credential on the server.
        """
        log = logger or logging.getLogger("fabric-tool-broker")
        prov: SearchProvider = provider or LazySearchProvider(config)
        server = ServerRelayAdapter(ExternalToolSidecar(prov, log))
        carriage = worker_carriage or ColocatedSidecarCarriage(
            ExternalToolSidecar(prov, log)
        )
        worker = WorkerSidecarAdapter(carriage)
        return cls(
            config,
            settle,
            EgressLocalityPolicy(config, server, worker),
            log,
            redispatch=redispatch,
        )

    def submit(self, env: ToolInvocationEnvelope) -> None:
        """Accept a recorded tool boundary and dispatch it off the agent's lane."""
        self._pool.submit(self._run, env)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

    def _run(self, env: ToolInvocationEnvelope) -> None:
        key = (env.task_id, env.call_correlation)
        try:
            result = self._dispatch(env, key)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a broker fault is a typed control datum
            self._log.warning("fabric tool dispatch failed: %s", exc)
            result = inline_outcome(
                ToolOutcome(
                    status=ToolOutcomeStatus.UNAVAILABLE,
                    value=f"the {env.interface} tool is unavailable",
                )
            )
        if isinstance(result, AmbiguousDelivery):
            self._on_ambiguous(env, key, result)
            return
        self._finish(key)
        self._settle(env.task_id, env.call_correlation, result)

    def _dispatch(
        self, env: ToolInvocationEnvelope, key: tuple[str, str]
    ) -> CarriageResult:
        if env.interface != SEARCH_INTERFACE:
            return inline_outcome(
                ToolOutcome(
                    status=ToolOutcomeStatus.UNAVAILABLE,
                    value=f"no handler for interface {env.interface}",
                )
            )
        recovery = self._touch(key)
        if not recovery.charged:
            # Draw down the episode budget once per logical operation; a same-idm-*
            # re-drive of a lost delivery reuses it rather than re-charging.
            if not self._charge(env.task_id):
                return inline_outcome(
                    ToolOutcome(
                        status=ToolOutcomeStatus.QUOTA,
                        value=(
                            f"search budget exhausted "
                            f"({self._cfg.max_calls} calls/episode)"
                        ),
                    )
                )
            recovery.charged = True
        query, max_results = self._parse_request(env.request_payload)
        if not query:
            return inline_outcome(
                ToolOutcome(
                    status=ToolOutcomeStatus.SUCCESS, value="No query supplied."
                )
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
            task_id=env.task_id,
        )
        adapter = self._policy.select()
        self._log.info(
            "fabric search interface=%s inv=%s locality=%s attempt=%s query=%r",
            env.interface,
            env.invocation_id,
            adapter.locality.value,
            recovery.attempts,
            query,
        )
        return adapter.execute(envelope, request)

    def _on_ambiguous(
        self,
        env: ToolInvocationEnvelope,
        key: tuple[str, str],
        result: AmbiguousDelivery,
    ) -> None:
        with self._lock:
            recovery = self._recovery.get(key)
            attempts = (
                recovery.attempts if recovery is not None else _MAX_DELIVERY_ATTEMPTS
            )
        if self._redispatch is not None and attempts < _MAX_DELIVERY_ATTEMPTS:
            if self._redispatch(env.task_id, env.call_correlation):
                # Held the durable boundary pending; re-drive the logical operation.
                self._log.info(
                    "remote tool op ambiguous; re-driving inv=%s attempt=%s reason=%s",
                    env.invocation_id,
                    attempts,
                    result.reason,
                )
                return
            # The re-drive was refused: the boundary already settled or was cancelled,
            # so its existing outcome stands. Recovery ends here — never manufacture a
            # terminal outcome over a boundary that is already terminal.
            self._finish(key)
            return
        # Genuine bounded-retry exhaustion (or no recovery path): terminalize with the
        # ambiguity audit outcome rather than treating a lost reply as a result.
        self._finish(key)
        self._settle(
            env.task_id,
            env.call_correlation,
            inline_outcome(
                ToolOutcome(
                    status=ToolOutcomeStatus.UNAVAILABLE,
                    value=(
                        f"the {env.interface} operation was lost and recovery was "
                        f"exhausted after {attempts} attempts"
                    ),
                )
            ),
        )

    def _touch(self, key: tuple[str, str]) -> _Recovery:
        with self._lock:
            recovery = self._recovery.get(key)
            if recovery is None:
                recovery = _Recovery()
                self._recovery[key] = recovery
            recovery.attempts += 1
            return recovery

    def _finish(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._recovery.pop(key, None)

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
