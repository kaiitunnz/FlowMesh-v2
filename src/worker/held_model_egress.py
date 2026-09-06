"""The held model turn's authorize-before-egress round trip, worker-local.

A facade holding a Codex turn keeps the chat request in worker-private custody and runs
one authorize-before-egress round trip here: it arms a permit waiter before proposing
(so a fast permit cannot outrace the waiter), proposes only the request digest, blocks
on the rendezvous for the one-use permit, and egresses synchronously through the
sidecar, returning the model's whole message inline. A denial, a permit that never
arrives, or a fence rejection is a terminal turn failure. Custody drops on resolution;
recovery re-runs the whole turn under a fresh permit, so nothing durable is held here.
"""

import logging
from collections.abc import Callable

from shared.tools.contract import AgentModelTurnProposal
from shared.tools.model.schema import (
    ModelCompletion,
    ModelRequest,
    model_request_digest,
)

from .lifecycle import PendingEgressRequestStore
from .mediated_egress_sidecar import HeldEgressReject, MediatedEgressSidecar
from .model_turn_rendezvous import ModelTurnRendezvous, PermitDenied

ProposeFn = Callable[[AgentModelTurnProposal], None]


class HeldModelEgress:
    """Run one held model turn's propose-permit-egress round trip for its reply."""

    def __init__(
        self,
        *,
        rendezvous: ModelTurnRendezvous,
        pending: PendingEgressRequestStore,
        propose: ProposeFn,
        sidecar: MediatedEgressSidecar,
        timeout_sec: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self._rendezvous = rendezvous
        self._pending = pending
        self._propose = propose
        self._sidecar = sidecar
        self._timeout_sec = timeout_sec
        self._log = logger or logging.getLogger("held-model-egress")

    def run(
        self, task_id: str, call_correlation: str, request: ModelRequest
    ) -> ModelCompletion | HeldEgressReject:
        """Authorize and egress one held model turn, returning its whole reply."""
        digest = model_request_digest(request.interface, request.url, request.body)
        with self._rendezvous.register(task_id, call_correlation) as waiter:
            self._pending.put(task_id, call_correlation, request)
            try:
                self._propose(
                    AgentModelTurnProposal(
                        agent_task_id=task_id,
                        call_correlation=call_correlation,
                        request_digest=digest,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a propose fault fails the turn
                self._log.warning("held model propose failed: %s", exc)
                self._pending.delete(task_id, call_correlation)
                return HeldEgressReject(reason="could not propose the model turn")
            try:
                delivery = waiter.await_permit(self._timeout_sec)
                if delivery is None:
                    return HeldEgressReject(reason="the model turn permit never came")
                if isinstance(delivery, PermitDenied):
                    return HeldEgressReject(reason=delivery.reason)
                return self._sidecar.egress_now(delivery)
            finally:
                self._pending.delete(task_id, call_correlation)
