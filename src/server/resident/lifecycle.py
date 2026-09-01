"""The Lifecycle & scale manager: the slower materialize/retain/drain loop.

It owns allocation leases and replica-directory lifecycle, performs policy-bounded,
demand-driven scale-from-zero for approved plan-derived families, and drains before an
idle teardown so accepted work reaches a terminal outcome. It never mints or releases an
admission credit: the capacity decision (join a warm replica, materialize one, or return
a typed denial) is a pure function of the directory, leases, and policy, and the actual
start and stop cross the flat worker plane through injected substrate hooks.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from shared.utils.ids import new_allocation_lease_id, new_replica_id

from ..utils.time import now_iso, parse_iso_ts
from .policy import ProvisioningDecision, ResidentPolicyLimits, decide_materialization
from .state import (
    SERVABLE_REPLICA_STATES,
    AllocationLease,
    ReplicaCapacityReport,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    SafeCapacityVector,
    ServiceFamily,
)
from .stores import ResidentStores

# Submits the family's serve substrate and returns the backing serve task id.
MaterializeFn = Callable[[ServiceFamily, ReplicaIncarnation], Awaitable[str]]
# Tears down a replica's backing serve task.
StopFn = Callable[[str], Awaitable[None]]

_ACTIVE_REPLICA_STATES: frozenset[ReplicaState] = frozenset(
    {
        ReplicaState.MATERIALIZING,
        ReplicaState.WARM,
        ReplicaState.BUSY,
        ReplicaState.DRAINING,
    }
)


@dataclass(frozen=True)
class CapacityPlan:
    """The decision for a family's demand: join a warm replica, materialize, or deny."""

    action: Literal["join", "materialize", "deny"]
    replica_id: str | None = None
    denial: ProvisioningDecision | None = None


class LifecycleScaleManager:
    """Materializes, retains, drains, and stops replicas for approved families."""

    def __init__(
        self,
        stores: ResidentStores,
        *,
        limits: ResidentPolicyLimits,
        admission_slots: int,
        idle_retain_sec: float = 0.0,
        persist: Callable[[], None] | None = None,
        materialize_fn: MaterializeFn | None = None,
        stop_fn: StopFn | None = None,
    ) -> None:
        self._stores = stores
        self._limits = limits
        self._admission_slots = max(1, admission_slots)
        self._idle_retain_sec = max(0.0, idle_retain_sec)
        self._persist = persist or (lambda: None)
        self._materialize_fn = materialize_fn
        self._stop_fn = stop_fn

    def _active_replicas(self, family: str) -> list[ReplicaIncarnation]:
        return [
            r
            for r in self._stores.directory.by_family(family)
            if r.state in _ACTIVE_REPLICA_STATES
        ]

    def plan_capacity(self, family: str, model_ref: str) -> CapacityPlan:
        """Decide, from the directory and policy, how to satisfy a family's demand."""
        active = self._active_replicas(family)
        joinable = next((r for r in active if r.state in SERVABLE_REPLICA_STATES), None)
        if joinable is not None:
            return CapacityPlan(action="join", replica_id=joinable.replica_id)
        if any(r.state is ReplicaState.MATERIALIZING for r in active):
            return CapacityPlan(action="materialize")
        decision = decide_materialization(
            model_ref=model_ref,
            limits=self._limits,
            active_replicas=len(active),
            materializing_replicas=sum(
                1 for r in active if r.state is ReplicaState.MATERIALIZING
            ),
        )
        if not decision.allowed:
            return CapacityPlan(action="deny", denial=decision)
        return CapacityPlan(action="materialize")

    async def materialize(self, family: ServiceFamily) -> ReplicaIncarnation:
        """Begin a bounded cold start: register the lease and replica, then start it."""
        if self._materialize_fn is None:
            raise RuntimeError("no materialize substrate is bound")
        replica = ReplicaIncarnation(
            replica_id=new_replica_id(),
            family=family.family,
            incarnation=1,
            state=ReplicaState.MATERIALIZING,
        )
        lease = AllocationLease(
            lease_id=new_allocation_lease_id(),
            family=family.family,
            replica_id=replica.replica_id,
            state=ReplicaState.MATERIALIZING,
        )
        replica.lease_id = lease.lease_id
        self._stores.leases.add(lease)
        self._stores.directory.add(replica)
        self._persist()
        try:
            serve_task_id = await self._materialize_fn(family, replica)
        except Exception:
            # A failed cold start must not wedge the family: invalidate the replica so a
            # later demand can materialize again, and let the caller settle the claim.
            self.on_preempt(replica.replica_id)
            raise
        replica.serve_task_id = serve_task_id
        replica.updated_at = now_iso()
        self._persist()
        return replica

    def on_replica_ready(self, replica_id: str, endpoint: ReplicaEndpoint) -> None:
        """Transition a materializing replica to warm with its reachable endpoint."""
        replica = self._stores.directory.get(replica_id)
        if replica is None or replica.state is not ReplicaState.MATERIALIZING:
            return
        replica.endpoint = endpoint
        replica.healthy = True
        replica.state = ReplicaState.WARM
        replica.updated_at = now_iso()
        self._promote_lease(replica_id, ReplicaState.WARM)
        self.refresh_report(replica_id)
        self._persist()

    def refresh_report(self, replica_id: str) -> None:
        """Ingest a conservative normalized capacity report for a replica's current
        state.
        """
        replica = self._stores.directory.get(replica_id)
        if replica is None:
            return
        replica.report_epoch += 1
        self._stores.reports.ingest(
            ReplicaCapacityReport(
                replica_id=replica_id,
                incarnation=replica.incarnation,
                report_epoch=replica.report_epoch,
                state=replica.state,
                healthy=replica.healthy and replica.state in SERVABLE_REPLICA_STATES,
                safe=SafeCapacityVector(admission_slots=self._admission_slots),
            )
        )

    def drain(self, replica_id: str) -> None:
        """Reject new claims on a replica while its admitted work reaches a safe
        outcome.
        """
        replica = self._stores.directory.get(replica_id)
        if replica is None or replica.state not in SERVABLE_REPLICA_STATES:
            return
        replica.state = ReplicaState.DRAINING
        replica.healthy = False
        replica.updated_at = now_iso()
        self._promote_lease(replica_id, ReplicaState.DRAINING)
        self.refresh_report(replica_id)
        self._persist()

    async def stop(self, replica_id: str) -> None:
        """Complete an idle teardown once a drained replica holds no admitted work."""
        replica = self._stores.directory.get(replica_id)
        if replica is None:
            return
        if self._stores.credit_ledger.held(replica_id) > 0:
            return
        replica.state = ReplicaState.STOPPED
        replica.healthy = False
        replica.updated_at = now_iso()
        self._promote_lease(replica_id, ReplicaState.STOPPED)
        self._persist()
        if self._stop_fn is not None and replica.serve_task_id is not None:
            await self._stop_fn(replica.serve_task_id)

    async def sweep_idle(self, *, now_ts: float | None = None) -> None:
        """Drain idle servable replicas past the retain window, then stop drained ones.

        A conservative scale-down: a servable replica holding no admission credit and
        idle past the retain window is drained; a drained replica still holding no
        credit is stopped, cancelling its serve task. A later eligible claim then
        materializes the family from zero again. A non-positive window disables it.
        """
        if self._idle_retain_sec <= 0:
            return
        reference = now_ts if now_ts is not None else parse_iso_ts(now_iso())
        for replica in self._stores.directory.all():
            held = self._stores.credit_ledger.held(replica.replica_id)
            if replica.state in SERVABLE_REPLICA_STATES:
                if held == 0 and self._idle_past_retain(replica, reference):
                    self.drain(replica.replica_id)
            elif replica.state is ReplicaState.DRAINING and held == 0:
                await self.stop(replica.replica_id)

    def _idle_past_retain(self, replica: ReplicaIncarnation, reference: float) -> bool:
        idle_for = reference - parse_iso_ts(replica.last_active_at)
        return idle_for >= self._idle_retain_sec

    def on_preempt(self, replica_id: str) -> None:
        """Invalidate a preempted or failed replica incarnation for reconciliation."""
        replica = self._stores.directory.get(replica_id)
        if replica is None:
            return
        replica.state = ReplicaState.PREEMPTED
        replica.healthy = False
        replica.incarnation += 1
        replica.updated_at = now_iso()
        self._promote_lease(replica_id, ReplicaState.PREEMPTED)
        self._persist()

    def _promote_lease(self, replica_id: str, state: ReplicaState) -> None:
        replica = self._stores.directory.get(replica_id)
        if replica is None or replica.lease_id is None:
            return
        lease = self._stores.leases.get(replica.lease_id)
        if lease is not None:
            lease.state = state
            lease.updated_at = now_iso()
