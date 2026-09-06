"""The worker-local mediated-egress sidecar: the enforced tool-fence egress lane.

The agent's own worker captured a fabric-tool request in worker-private custody and
proposed only its digest; central control returned a one-use permit over the worker's
authenticated attachment. This bounded lane validates the permit and digest against the
worker fence, egresses through the local provider, materializes a large result by
reference, and reports one permit-fenced terminal fact back over the attachment. It is a
worker-lifecycle execution lane, not a task, replica, endpoint, or authority: it holds
no admission credit, selects no worker, and discovers no peer.

Raw-request custody is retained non-destructively: the lane peeks the request, egresses,
reports the fenced outcome, and deletes the request only on the control plane's
committed-outcome reap. A crashed egress reports nothing, leaving the boundary pending
for a same-``idempotency_key`` re-drive; a fence failure is a declared terminal boundary
failure, never a retryable provider response.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.tools.contract import (
    MediatedOperationOutcome,
    MediatedOperationPermit,
    ToolOperationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.search.egress import ExternalToolSidecar
from shared.tools.search.providers import LazySearchProvider
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)

from .lifecycle import PendingToolRequestStore
from .tool_fence import ProviderBinding, fence_reason, materialize_tool_outcome

# Resolves this worker's id and incarnation once it is registered.
AudienceFn = Callable[[], tuple[str, int]]
# A sink the lane calls to report one fenced terminal fact over the attachment.
OutcomeSink = Callable[[MediatedOperationOutcome], None]

_BoundaryKey = tuple[str, str]


class MediatedEgressSidecar:
    """Validate a permit and egress one worker-originated fabric-tool operation."""

    def __init__(
        self,
        *,
        pending_requests: PendingToolRequestStore,
        audience: AudienceFn,
        provider: str,
        api_key: str | None,
        outcome_sink: OutcomeSink,
        content_store: FabricContentStore | None = None,
        policy_class: str = "default",
        interfaces: frozenset[str] = frozenset({SEARCH_INTERFACE}),
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        self._pending = pending_requests
        self._audience = audience
        self._provider = provider
        self._content_store = content_store
        self._policy_class = policy_class
        self._interfaces = interfaces
        self._sink = outcome_sink
        self._log = logger or logging.getLogger("worker-mediated-egress")
        self._sidecar = ExternalToolSidecar(
            LazySearchProvider(ProviderBinding(provider, api_key)), self._log
        )
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mediated-egress"
        )
        self._lock = threading.Lock()
        # Consumed one-use permit ids -> deadline. A re-drive re-mints a fresh id;
        # only an exact replay of one authorized permit is refused.
        self._consumed: dict[str, float] = {}
        # In-flight egress futures and cancelled boundaries, keyed by the boundary.
        self._inflight: dict[_BoundaryKey, Future[None]] = {}
        self._cancelled: set[_BoundaryKey] = set()

    def submit_permit(self, permit: MediatedOperationPermit) -> None:
        """Drive one authorized operation; ignore an exact permit replay."""
        key = (permit.agent_task_id, permit.call_correlation)
        with self._lock:
            if not self._consume_permit(permit.permit_id, permit.deadline_epoch):
                self._log.info("mediated permit replay ignored id=%s", permit.permit_id)
                return
            if key in self._inflight:
                return
            self._inflight[key] = self._pool.submit(self._drive, permit)

    def reap(self, agent_task_id: str, call_correlation: str) -> None:
        """Delete worker-private custody after a committed-outcome acknowledgement.

        A committed outcome and a cancellation both reap. If the egress has not started
        it is cancelled and dropped here, since ``_drive`` never runs to clear it; if it
        is already running its report is suppressed and ``_drive`` clears it on exit.
        """
        key = (agent_task_id, call_correlation)
        with self._lock:
            fut = self._inflight.get(key)
            if fut is not None:
                if fut.cancel():
                    self._inflight.pop(key, None)
                    self._cancelled.discard(key)
                else:
                    self._cancelled.add(key)
        self._pending.delete(agent_task_id, call_correlation)

    def stop(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _drive(self, permit: MediatedOperationPermit) -> None:
        key = (permit.agent_task_id, permit.call_correlation)
        try:
            report = self._produce(permit)
        except Exception as exc:  # noqa: BLE001 - a crashed egress leaves it ambiguous
            # No report: the control plane holds the boundary pending and re-drives
            # under the same idempotency key with a fresh permit.
            self._log.warning("mediated egress raised, leaving it ambiguous: %s", exc)
            report = None
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                cancelled = key in self._cancelled
                self._cancelled.discard(key)
        if report is not None and not cancelled:
            self._sink(report)

    def _produce(self, permit: MediatedOperationPermit) -> MediatedOperationOutcome:
        # Recover an already-materialized outcome first: a re-drive after a materialize
        # the control plane never recorded returns the prior result by its idempotency
        # key, before consuming the worker-private request.
        if (prior := self._prior_manifest(permit)) is not None:
            return self._report(permit, outcome_ref=prior)
        request = self._pending.peek(permit.agent_task_id, permit.call_correlation)
        if request is None:
            # No prior outcome and the capturing incarnation is gone: fail terminally
            # rather than egressing an unfenced request.
            return self._report(
                permit, error=f"no worker-private request for {permit.call_correlation}"
            )
        if (reason := self._fence_reject(permit, request)) is not None:
            return self._report(permit, error=f"tool permit fence rejected: {reason}")
        outcome = self._egress(permit, request)
        materialized = materialize_tool_outcome(
            outcome,
            idempotency_key=permit.idempotency_key,
            content_store=self._content_store,
        )
        if isinstance(materialized, OutcomeManifest):
            return self._report(permit, outcome_ref=materialized)
        return self._report(permit, outcome=materialized)

    def _egress(
        self, permit: MediatedOperationPermit, request: ToolRequest
    ) -> ToolOutcome:
        envelope = ToolOperationEnvelope(
            interface=permit.interface,
            idempotency_key=permit.idempotency_key,
            max_results=min(request.max_results, permit.max_results),
            timeout_sec=permit.timeout_sec,
            result_char_cap=permit.result_char_cap,
        )
        self._log.info(
            "mediated egress in worker=%s interface=%s",
            permit.target_id,
            permit.interface,
        )
        try:
            return self._sidecar.execute(envelope, request)
        except ValueError as exc:
            # A misprovisioned provider is a deterministic fault: a typed terminal
            # outcome rather than an ambiguous retry loop.
            self._log.warning("tool provider unavailable: %s", exc)
            return ToolOutcome(
                status=ToolOutcomeStatus.UNAVAILABLE,
                value="the external-tool provider is unavailable",
            )

    def _fence_reject(
        self, permit: MediatedOperationPermit, request: ToolRequest
    ) -> str | None:
        worker_id, generation = self._audience()
        return fence_reason(
            interface=permit.interface,
            target_id=permit.target_id,
            target_generation=permit.target_generation,
            policy_class=permit.policy_class,
            deadline_epoch=permit.deadline_epoch,
            request_digest=permit.request_digest,
            computed_digest=tool_request_digest(
                request.interface, request.query, request.max_results
            ),
            worker_id=worker_id,
            worker_generation=generation,
            allowed_interfaces=self._interfaces,
            expected_policy_class=None,
        )

    def _prior_manifest(
        self, permit: MediatedOperationPermit
    ) -> OutcomeManifest | None:
        if self._content_store is None or permit.idempotency_key is None:
            return None
        return self._content_store.find(permit.idempotency_key)

    def _consume_permit(self, permit_id: str, deadline_epoch: float) -> bool:
        """Atomically consume a one-use permit id; ``False`` on an exact replay."""
        now = time.time()
        for expired in [p for p, exp in self._consumed.items() if exp <= now]:
            self._consumed.pop(expired, None)
        if permit_id in self._consumed:
            return False
        self._consumed[permit_id] = deadline_epoch
        return True

    @staticmethod
    def _report(
        permit: MediatedOperationPermit,
        *,
        outcome: ToolOutcome | None = None,
        outcome_ref: OutcomeManifest | None = None,
        error: str | None = None,
    ) -> MediatedOperationOutcome:
        return MediatedOperationOutcome(
            permit_id=permit.permit_id,
            agent_task_id=permit.agent_task_id,
            call_correlation=permit.call_correlation,
            invocation_id=permit.invocation_id,
            idempotency_key=permit.idempotency_key,
            outcome=outcome,
            outcome_ref=outcome_ref,
            error=error,
        )


__all__ = ["AudienceFn", "MediatedEgressSidecar", "OutcomeSink"]
