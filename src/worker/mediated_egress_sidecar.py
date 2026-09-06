"""The worker-local mediated-egress sidecar: the enforced egress lane for a boundary.

The agent's own worker captures a fabric-tool or managed-model request in worker-private
custody and proposes only its digest; central control returns a one-use permit over the
worker's authenticated attachment. This bounded lane validates the permit and digest
against the worker fence, egresses through the local provider for the permit interface,
materializes a large result by reference, and reports one permit-fenced terminal fact
back over the attachment. It is a worker-lifecycle execution lane, not a task, replica,
endpoint, or authority.

Raw-request custody is retained non-destructively: the lane peeks the request, egresses,
reports the fenced outcome, and deletes the request only on the control plane's
committed-outcome reap. A crashed egress reports nothing, leaving the boundary pending
for a same-``idempotency_key`` re-drive; a fence failure is a declared terminal boundary
failure, never a retryable provider response.
"""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.tools.contract import (
    MediatedOperationOutcome,
    MediatedOperationPermit,
    ToolOperationEnvelope,
    ToolOutcome,
)
from shared.tools.model.egress import ModelEgressError
from shared.tools.model.schema import ModelCompletion

from .lifecycle import CapturedRequest, PendingEgressRequestStore
from .tool_fence import fence_reason, materialize_tool_outcome

# Resolves this worker's id and incarnation once it is registered.
AudienceFn = Callable[[], tuple[str, int]]
# A sink the lane calls to report one fenced terminal fact over the attachment.
OutcomeSink = Callable[[MediatedOperationOutcome], None]

_BoundaryKey = tuple[str, str]


class EgressInterface(Protocol):
    """One interface's egress: the request digest and the provider execution surface."""

    interface: str

    def digest(self, request: CapturedRequest) -> str: ...

    def execute(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        credential: str | None,
    ) -> ToolOutcome: ...


@runtime_checkable
class SyncModelEgress(Protocol):
    """A model egress that returns the whole message inline for a held turn."""

    interface: str

    def digest(self, request: CapturedRequest) -> str: ...

    def complete(
        self,
        envelope: ToolOperationEnvelope,
        request: CapturedRequest,
        credential: str | None,
    ) -> ModelCompletion: ...


@dataclass(frozen=True)
class HeldEgressReject:
    """A terminal rejection of a held model turn: a fence failure or provider fault."""

    reason: str


class MediatedEgressSidecar:
    """Validate a permit and egress one worker-originated mediated operation."""

    def __init__(
        self,
        *,
        pending_requests: PendingEgressRequestStore,
        audience: AudienceFn,
        egresses: Sequence[EgressInterface],
        outcome_sink: OutcomeSink,
        content_store: FabricContentStore | None = None,
        policy_class: str = "default",
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        self._pending = pending_requests
        self._audience = audience
        self._content_store = content_store
        self._policy_class = policy_class
        self._egresses = {egress.interface: egress for egress in egresses}
        self._interfaces = frozenset(self._egresses)
        self._sink = outcome_sink
        self._log = logger or logging.getLogger("worker-mediated-egress")
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
        """Delete worker-private custody after a committed outcome or a cancellation.

        An unstarted egress is cancelled and cleared here; a running egress has its
        report suppressed and is cleared when it finishes.
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
        egress = self._egresses.get(permit.interface)
        if egress is None:
            return self._report(
                permit, error=f"no egress for interface {permit.interface!r}"
            )
        request = self._pending.peek(permit.agent_task_id, permit.call_correlation)
        if request is None:
            # No prior outcome and the capturing incarnation is gone: fail terminally
            # rather than egressing an unfenced request.
            return self._report(
                permit, error=f"no worker-private request for {permit.call_correlation}"
            )
        if (reason := self._fence_reject(permit, egress.digest(request))) is not None:
            return self._report(permit, error=f"permit fence rejected: {reason}")
        outcome = self._egress(permit, request, egress)
        materialized = materialize_tool_outcome(
            outcome,
            idempotency_key=permit.idempotency_key,
            content_store=self._content_store,
        )
        if isinstance(materialized, OutcomeManifest):
            return self._report(permit, outcome_ref=materialized)
        return self._report(permit, outcome=materialized)

    @staticmethod
    def _envelope(permit: MediatedOperationPermit) -> ToolOperationEnvelope:
        return ToolOperationEnvelope(
            interface=permit.interface,
            idempotency_key=permit.idempotency_key,
            max_results=permit.max_results,
            timeout_sec=permit.timeout_sec,
            result_char_cap=permit.result_char_cap,
        )

    def _egress(
        self,
        permit: MediatedOperationPermit,
        request: CapturedRequest,
        egress: EgressInterface,
    ) -> ToolOutcome:
        self._log.info(
            "mediated egress in worker=%s interface=%s",
            permit.target_id,
            permit.interface,
        )
        return egress.execute(self._envelope(permit), request, permit.credential)

    def egress_now(
        self, permit: MediatedOperationPermit
    ) -> ModelCompletion | HeldEgressReject:
        """Egress a held model turn synchronously, returning its completion inline.

        Distinct from ``submit_permit``: a synchronous-turn-only facade consumes the
        permit in its own thread and needs the model's whole message back inline, not a
        control-plane settle. The permit is consumed one-use and the worker-private
        request is fenced before egress; a fence rejection or a provider fault is a
        terminal reject the facade fails the turn on. Custody is left for the facade to
        reap once the turn resolves.
        """
        with self._lock:
            if not self._consume_permit(permit.permit_id, permit.deadline_epoch):
                return HeldEgressReject(reason="permit replay")
        egress = self._egresses.get(permit.interface)
        if not isinstance(egress, SyncModelEgress):
            return HeldEgressReject(
                reason=f"no held egress for interface {permit.interface!r}"
            )
        request = self._pending.peek(permit.agent_task_id, permit.call_correlation)
        if request is None:
            return HeldEgressReject(
                reason=f"no worker-private request for {permit.call_correlation}"
            )
        if (reason := self._fence_reject(permit, egress.digest(request))) is not None:
            return HeldEgressReject(reason=f"permit fence rejected: {reason}")
        self._log.info(
            "held model egress in worker=%s interface=%s",
            permit.target_id,
            permit.interface,
        )
        try:
            return egress.complete(self._envelope(permit), request, permit.credential)
        except ModelEgressError as exc:
            return HeldEgressReject(reason=str(exc))

    def _fence_reject(
        self, permit: MediatedOperationPermit, computed_digest: str
    ) -> str | None:
        worker_id, generation = self._audience()
        return fence_reason(
            interface=permit.interface,
            target_id=permit.target_id,
            target_generation=permit.target_generation,
            policy_class=permit.policy_class,
            deadline_epoch=permit.deadline_epoch,
            request_digest=permit.request_digest,
            computed_digest=computed_digest,
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


__all__ = [
    "AudienceFn",
    "EgressInterface",
    "HeldEgressReject",
    "MediatedEgressSidecar",
    "OutcomeSink",
    "SyncModelEgress",
]
