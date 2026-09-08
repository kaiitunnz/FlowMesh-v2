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
    AdmissionProfile,
    AllocationLease,
    ProvisioningDenialReason,
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
StopFn = Callable[[str], None]

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
        adapter_slots: int = 4,
        idle_retain_sec: float = 0.0,
        persist: Callable[[], None] | None = None,
        materialize_fn: MaterializeFn | None = None,
        stop_fn: StopFn | None = None,
    ) -> None:
        self._stores = stores
        self._limits = limits
        self._admission_slots = max(1, admission_slots)
        self._adapter_slots = max(1, adapter_slots)
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

    def plan_capacity(
        self, family: str, model_ref: str, profile: AdmissionProfile | None = None
    ) -> CapacityPlan:
        """Decide, from the directory and policy, how to satisfy a family's demand.

        A servable replica is joinable only if it can serve the claim's adapter; a base
        or held-adapter claim joins a warm replica, while a new distinct adapter joins
        only where a free adapter slot remains. When no servable replica can take the
        adapter and policy cannot materialize another, the demand is denied promptly and
        correctly rather than waiting out the cold-start deadline.
        """
        active = self._active_replicas(family)
        servable = [r for r in active if r.state in SERVABLE_REPLICA_STATES]
        joinable = next((r for r in servable if self._adapter_fits(r, profile)), None)
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
        if decision.allowed:
            return CapacityPlan(action="materialize")
        if servable and profile is not None and profile.adapter_ref is not None:
            # A warm replica exists but its adapter slots are full for this new adapter
            # and no replica can be added: an adapter-budget denial, not a cold start.
            # The co-occurring capacity reason is surfaced in the detail so the denial
            # names both why the adapter does not fit and why no replica can be added.
            reason = decision.reason.value if decision.reason is not None else "no room"
            return CapacityPlan(
                action="deny",
                denial=ProvisioningDecision.deny(
                    ProvisioningDenialReason.ADAPTER_SLOT_CAP,
                    f"no free adapter slot for {profile.adapter_ref!r} and the family "
                    f"cannot add a replica ({reason}: {decision.detail or ''})",
                ),
            )
        return CapacityPlan(action="deny", denial=decision)

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

    def adopt_standing_replica(
        self,
        family: ServiceFamily,
        *,
        serve_task_id: str,
        binding_generation: int,
        endpoint: ReplicaEndpoint,
    ) -> ReplicaIncarnation:
        """Register a public serve task's running replica as a standing allocation.

        Unlike a demand-managed replica, this is not materialized from zero: the serve
        task already runs the engine, so the replica is adopted directly into the
        directory as warm and task-lifetime pinned (``standing``), and is never
        idle-torn down while its serve task is live. It replaces any prior incarnation
        for the same serve task so a re-adoption supersedes the old fence.
        """
        for prior in self._stores.directory.by_family(family.family):
            if prior.serve_task_id == serve_task_id and prior.state in (
                _ACTIVE_REPLICA_STATES
            ):
                self.on_preempt(prior.replica_id)
        replica = ReplicaIncarnation(
            replica_id=new_replica_id(),
            family=family.family,
            incarnation=1,
            state=ReplicaState.WARM,
            endpoint=endpoint,
            healthy=True,
            serve_task_id=serve_task_id,
            binding_generation=binding_generation,
            standing=True,
        )
        lease = AllocationLease(
            lease_id=new_allocation_lease_id(),
            family=family.family,
            replica_id=replica.replica_id,
            state=ReplicaState.WARM,
        )
        replica.lease_id = lease.lease_id
        self._stores.leases.add(lease)
        self._stores.directory.add(replica)
        self.refresh_report(replica.replica_id)
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
        held = self._held_adapters(replica_id)
        self._stores.reports.ingest(
            ReplicaCapacityReport(
                replica_id=replica_id,
                incarnation=replica.incarnation,
                report_epoch=replica.report_epoch,
                state=replica.state,
                healthy=replica.healthy and replica.state in SERVABLE_REPLICA_STATES,
                safe=SafeCapacityVector(admission_slots=self._admission_slots),
                adapter_slots_free=max(0, self._adapter_slots - len(held)),
                held_adapters=held,
            )
        )

    def refresh_family_reports(self, family: str) -> None:
        """Re-report every servable replica of a family, so the adapter-slot gate reads
        the current held-adapter count before an admission decision.
        """
        for replica in self._stores.directory.by_family(family):
            if replica.state in SERVABLE_REPLICA_STATES:
                self.refresh_report(replica.replica_id)

    def _held_adapters(self, replica_id: str) -> tuple[str, ...]:
        """The distinct adapters a replica's credit-bearing claims currently hold.

        Multiple claims for the same adapter share one slot; a base (adapterless) claim
        holds none. A claim for a held adapter shares its slot rather than being denied.
        """
        held = {
            request.profile.adapter_ref
            for claim in self._stores.claims.credit_bearing_for_replica(replica_id)
            if (request := self._stores.invocations.get(claim.invocation_id))
            is not None
            and request.profile.adapter_ref is not None
        }
        return tuple(sorted(held))

    def replica_holds_adapter(self, replica_id: str, adapter_ref: str) -> bool:
        """Whether a credit-bearing claim on the replica still holds the adapter.

        Read on a claim release to decide whether the adapter's engine slot may be
        unloaded: it may be freed only once no remaining credit-bearing claim on the
        replica references it, so a peer's adapter is never unloaded out from under it.
        """
        return adapter_ref in self._held_adapters(replica_id)

    def _adapter_fits(
        self, replica: ReplicaIncarnation, profile: "AdmissionProfile | None"
    ) -> bool:
        """Whether the replica can serve the profile's adapter (or it needs none).

        A base claim always fits; an adapter already resident shares its slot; a new
        distinct adapter fits only while a free adapter slot remains.
        """
        if profile is None or profile.adapter_ref is None:
            return True
        held = self._held_adapters(replica.replica_id)
        return profile.adapter_ref in held or len(held) < self._adapter_slots

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

    def stop(self, replica_id: str) -> None:
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
        self._reap_serve_task(replica.serve_task_id)

    def sweep_idle(self, *, now_ts: float | None = None) -> None:
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
            if replica.standing:
                # A live standing serve allocation is pinned to its serve task and never
                # idle-torn-down; once drained by its task's stop it is stopped when its
                # admitted work has drained, so it does not linger DRAINING.
                if replica.state is ReplicaState.DRAINING and held == 0:
                    self.stop(replica.replica_id)
                continue
            if replica.state in SERVABLE_REPLICA_STATES:
                if held == 0 and self._idle_past_retain(replica, reference):
                    self.drain(replica.replica_id)
            elif replica.state is ReplicaState.DRAINING and held == 0:
                self.stop(replica.replica_id)

    def _idle_past_retain(self, replica: ReplicaIncarnation, reference: float) -> bool:
        idle_for = reference - parse_iso_ts(replica.last_active_at)
        return idle_for >= self._idle_retain_sec

    def on_preempt(self, replica_id: str) -> None:
        """Invalidate a preempted or failed replica incarnation for reconciliation.

        Reaps the invalidated incarnation's backing serve task so a replica the family
        will re-materialize from zero does not leave an orphaned serve workflow running.
        A standing serve allocation is exempt: it is the user's long-running task,
        drained
        only by its own lifecycle (stop/cancel/TTL/failure), and preempt-and-recreate
        cannot recover it — so a per-request failure never invalidates or reaps it.
        """
        replica = self._stores.directory.get(replica_id)
        if replica is None or replica.standing:
            return
        serve_task_id = replica.serve_task_id
        replica.state = ReplicaState.PREEMPTED
        replica.healthy = False
        replica.incarnation += 1
        replica.updated_at = now_iso()
        self._promote_lease(replica_id, ReplicaState.PREEMPTED)
        self._persist()
        self._reap_serve_task(serve_task_id)

    def _reap_serve_task(self, serve_task_id: str | None) -> None:
        """Cancel a replica's backing serve task; absent or terminal is a no-op."""
        if self._stop_fn is not None and serve_task_id is not None:
            self._stop_fn(serve_task_id)

    def _promote_lease(self, replica_id: str, state: ReplicaState) -> None:
        replica = self._stores.directory.get(replica_id)
        if replica is None or replica.lease_id is None:
            return
        lease = self._stores.leases.get(replica.lease_id)
        if lease is not None:
            lease.state = state
            lease.updated_at = now_iso()
