"""Worker-originated mediated operations whose permits control has relayed."""

from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from server.telemetry.tracing import ControlPlaneTracer, format_traceparent
from shared.schemas.command import MediatedOpMessage
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.tools.contract import (
    AgentModelTurnProposal,
    MediatedOperationOutcome,
    MediatedOperationPermit,
)

from ...config import WebSearchConfig
from ...orchestration import OrchestrationEngine
from ...orchestration.tool_dispatch import MODEL_INTERFACE
from ...registries.worker import WorkerRegistry
from ..models import TaskRecord
from .after_commit import Reap

# Releases an invocation's resident credit on its committed terminal, returning the
# consumption when it completes on the resident event loop.
type ResidentTerminalHook = Callable[[str, bool], Future[Any] | None]


@dataclass
class PendingOp:
    """A worker-originated tool operation whose permit was relayed to its origin
    worker, and when control re-drives it if no outcome has arrived."""

    agent_task_id: str
    call_correlation: str
    worker_id: str
    # The node the origin worker registered on; after a store wipe its id can name
    # another node's worker, which holds none of the operation's request.
    node_alias: str
    redrive_at: float
    redrives: int = 0


# A pending operation's boundary fails once it has been re-driven this many times.
_OP_REDRIVE_LIMIT = 5

# A generous bound on a materialized external-model completion; a larger response
# settles by reference under the reference-backed outcome contract.
_MODEL_PERMIT_RESULT_CHAR_CAP = 1_000_000


def deny_model_turn_payload(
    proposal: AgentModelTurnProposal, reason: str
) -> dict[str, Any]:
    """Build the payload of a deny frame that fails a held turn before its deadline."""
    return {
        "agent_task_id": proposal.agent_task_id,
        "call_correlation": proposal.call_correlation,
        "reason": reason,
    }


class MediatedOperations:
    """Holds each mediated operation whose permit was relayed to its origin worker,
    sizes and stamps permits, relays reaps and denials to workers, and releases a
    settled invocation's resident credit."""

    def __init__(
        self,
        tasks: dict[str, TaskRecord],
        engines: dict[str, OrchestrationEngine],
        worker_registry: WorkerRegistry,
        web_search: WebSearchConfig,
        model_egress_timeout_sec: float,
        content_scope_authority: Callable[[str, str], None] | None,
        control: ControlPlaneTracer,
    ) -> None:
        self._tasks = tasks
        self._engines = engines
        self._worker_registry = worker_registry
        self._web_search = web_search
        self._model_egress_timeout_sec = model_egress_timeout_sec
        self._content_scope_authority = content_scope_authority
        self._control = control
        # Worker-originated tool operations whose permit was relayed to the origin
        # worker's egress sidecar, keyed by permit id. Reaps custody on settle or
        # cancel. In-memory and rebuilt on restart from the pending boundary, never
        # durably persisted.
        self.pending_ops: dict[str, PendingOp] = {}
        self.resident_terminal_hook: ResidentTerminalHook | None = None

    def take_stale_ops(
        self, occurrence: tuple[str, str]
    ) -> list[tuple[str, PendingOp]]:
        """Drop and return the pending operations of one boundary occurrence."""
        stale = [
            (permit_id, op)
            for permit_id, op in self.pending_ops.items()
            if (op.agent_task_id, op.call_correlation) == occurrence
        ]
        for permit_id, _ in stale:
            del self.pending_ops[permit_id]
        return stale

    def record_issued_op(self, permit_id: str, op: PendingOp) -> None:
        self.pending_ops[permit_id] = op

    def take_settled(self, outcome: MediatedOperationOutcome) -> PendingOp | None:
        """Drop and return the pending operation an outcome settles."""
        pending = self.pending_ops.pop(outcome.permit_id, None)
        # A re-mint of the same operation is settled by this outcome too.
        occurrence = (outcome.agent_task_id, outcome.call_correlation)
        for permit_id, op in list(self.pending_ops.items()):
            if (op.agent_task_id, op.call_correlation) == occurrence:
                pending = pending or op
                del self.pending_ops[permit_id]
        return pending

    def take_overdue(
        self, worker_id: str, now: float
    ) -> tuple[list[PendingOp], list[tuple[str, PendingOp]]]:
        """Sort a worker's overdue operations into those out of re-drives, which are
        dropped, and those to re-drive, which are charged one."""
        exhausted: list[PendingOp] = []
        redrive: list[tuple[str, PendingOp]] = []
        for permit_id, op in list(self.pending_ops.items()):
            if op.worker_id != worker_id or op.redrive_at > now:
                continue
            if op.redrives >= _OP_REDRIVE_LIMIT:
                del self.pending_ops[permit_id]
                exhausted.append(op)
            else:
                op.redrives += 1
                redrive.append((permit_id, op))
        return exhausted, redrive

    def discard_op(self, permit_id: str, op: PendingOp) -> None:
        """Drop a pending operation unless a re-mint replaced it."""
        if self.pending_ops.get(permit_id) is op:
            del self.pending_ops[permit_id]

    def pending_for_worker(self, worker_id: str) -> set[tuple[str, str]]:
        """The (agent task, call) of each operation a worker originated."""
        return {
            (op.agent_task_id, op.call_correlation)
            for op in self.pending_ops.values()
            if op.worker_id == worker_id
        }

    def drop_worker_ops(self, worker_id: str) -> None:
        """Drop every pending operation a worker originated."""
        for permit_id, op in list(self.pending_ops.items()):
            if op.worker_id == worker_id:
                del self.pending_ops[permit_id]

    def op_permit_budget(self, interface: str) -> tuple[int, float, int]:
        """The (max_results, timeout, result_char_cap) budget a permit runs within."""
        if interface == MODEL_INTERFACE:
            return 1, self._model_egress_timeout_sec, _MODEL_PERMIT_RESULT_CHAR_CAP
        cfg = self._web_search
        return cfg.max_results, cfg.timeout_sec, cfg.result_char_cap

    def _assign_content_scope(
        self, permit: MediatedOperationPermit, agent: TaskRecord
    ) -> str:
        """Assign the scope this permit's outcome materializes in, and record it.

        The scope is the task's owner, so a result materializes in the tenant's
        namespace rather than the egressing worker's. Recording it against the permit's
        idempotency key is what lets the finalization this outcome later reports be
        checked against the scope control assigned, rather than one the reporting
        worker names for itself.
        """
        if self._content_scope_authority is not None and permit.idempotency_key:
            self._content_scope_authority(permit.idempotency_key, agent.org_id)
        return agent.org_id

    def stamped_permit_payload(
        self, permit: MediatedOperationPermit, agent: TaskRecord
    ) -> dict[str, Any]:
        """The permit's wire payload, carrying what only the dispatching record knows.

        The trace stamp is post-mint: the boundary span id derives from the permit's own
        ``invocation_id``, which does not exist as an object until minting returns.
        """
        stamp: dict[str, Any] = {
            "content_scope": self._assign_content_scope(permit, agent)
        }
        if self._control.enabled:
            stamp["traceparent"] = format_traceparent(
                workflow_to_trace_id_int(agent.workflow_id),
                derived_span_id(SpanIdKind.INVOCATION, permit.invocation_id),
            )
        return permit.model_copy(update=stamp).model_dump(mode="json")

    def reap_mediated_op(self, worker_id: str, agent_task_id: str, call: str) -> None:
        """Relay a best-effort reap so the origin worker drops the request custody."""
        worker = self._worker_registry.get_worker(worker_id)
        if worker is None:
            return
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="reap",
                payload={"agent_task_id": agent_task_id, "call_correlation": call},
            ),
        )

    def reap_captured_request_locked(
        self, worker_id: str | None, task_id: str, call: str, interface: str | None
    ) -> Reap | None:
        """Build the reap of a request the worker captured for a boundary that will
        never run."""
        if not worker_id:
            return None
        record = self._tasks.get(task_id)
        engine = self._engines.get(record.workflow_id) if record else None
        resident = (
            interface == MODEL_INTERFACE
            and engine is not None
            and engine.service_dependency(task_id) is not None
        )
        return Reap(worker_id, task_id, call, resident=resident)

    def relay_resident_reap(self, worker_id: str, task_id: str, call: str) -> None:
        """Relay a best-effort reap so the worker drops a captured resident request."""
        worker = self._worker_registry.get_worker(worker_id)
        if worker is None:
            return
        self._worker_registry.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind="resident_reap",
                payload={"task_id": task_id, "call_correlation": call},
            ),
        )

    def reap_ops_for_agents_locked(self, agent_task_ids: Sequence[str]) -> list[Reap]:
        """Take the agents' pending tool operations, returning the reap of each."""
        agents = set(agent_task_ids)
        reaps: list[Reap] = []
        for permit_id, op in list(self.pending_ops.items()):
            if op.agent_task_id in agents:
                del self.pending_ops[permit_id]
                reaps.append(Reap(op.worker_id, op.agent_task_id, op.call_correlation))
        return reaps

    def set_resident_terminal_hook(self, hook: ResidentTerminalHook) -> None:
        """Install the consumer that releases a resident admission credit on DS
        terminal."""
        self.resident_terminal_hook = hook

    def release_resident_credit(
        self, invocation_id: str, *, failed: bool
    ) -> Future[Any] | None:
        """Hand a committed terminal to the credit consumer; returns its consumption
        when the consumer reports one."""
        if self.resident_terminal_hook is None:
            return None
        return self.resident_terminal_hook(invocation_id, failed)

    def reap_captures_locked(
        self,
        worker_id: str | None,
        task_id: str,
        captures: list[tuple[str, str | None]],
    ) -> list[Reap]:
        """Build the reaps of the requests a step captured for boundaries control
        never runs."""
        return [
            reap
            for call, interface in captures
            if (
                reap := self.reap_captured_request_locked(
                    worker_id, task_id, call, interface
                )
            )
        ]
