"""Resident-capacity control as the worker-originated resident boundary authority.

A resident model boundary is captured worker-private on the agent's own worker and
originated here as a control-only proposal. This service raises the durable
``ServiceClaim`` for the invocation, drives the Admission controller and Lifecycle &
scale manager, binds the selected replica's sidecar, resolves the network-plane origin
fence, and relays the claim-bound handoff to the origin worker — which carries the raw
request over the reverse-rendezvous relay and serves the engine stream on the replica
worker. It records ``ACCEPTED`` and mints the route authorization only on the origin
worker's engine enqueue ack, and settles the boundary by reference from the origin
worker's fenced terminal manifest. The credit releases only when the fenced DS terminal
returns through ``invocation_id``.

Every transition-gating report is safe under loss: a missing ack leaves the claim
credit-bearing, and a missing or uncertain outcome holds the credit and re-drives under
the same invocation identity rather than falling through to a wrong terminal.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from shared.resident.contracts import AdmissionHandoff, ReplicaEndpoint
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamStatus,
)
from shared.utils.ids import new_relay_session_id

from ..network.state import (
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteOrigin,
)
from ..orchestration.tool_dispatch import ToolInvocationEnvelope
from ..task.v2.representations.operators import ServiceDependency
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .policy import ResidentPolicyLimits
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ProvisioningDenialReason,
    ReplicaIncarnation,
    ReplicaState,
    ResidentSnapshot,
    ServiceClaim,
    ServiceFamily,
)
from .stores import ResidentStores

# Resolves a task's normalized resident dependency: (workflow_id, dependency) or None.
DependencyResolver = Callable[[str], tuple[str, ServiceDependency] | None]
# Settles a mediated boundary back at its originating call, or fails it with an error.
SettleCallback = Callable[..., bool]
# Re-drives a still-pending mediated boundary off-lane without settling it.
RedispatchCallback = Callable[[str, str], bool]
# Reads a serve substrate's reported endpoint once ready, else None.
EndpointProbe = Callable[[str], ReplicaEndpoint | None]
# Persists the authoritative CS snapshot.
PersistCallback = Callable[[], None]


class RouteResolver(Protocol):
    """The control-plane route resolution the origin fence binds.

    Satisfied structurally by the network plane; the resident path binds it to resolve
    the trusted origin whose id fences the handoff and authorization, rather than
    re-deriving endpoints, reachability, or the resolver.
    """

    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute] | None: ...


class ResidentSessionWriter(Protocol):
    """The per-session routing record the reverse-relay bridges route frames by."""

    async def update(self, session_id: str, **fields: str | int) -> None: ...

    async def delete(self, session_id: str) -> None: ...


# Relays one resident control frame to a worker over its attachment; True if delivered.
WorkerRelay = Callable[[str, str, dict[str, Any]], bool]
# Resolves the node a worker is attached to, or None when the worker is gone.
NodeOfWorker = Callable[[str | None], str | None]
# Resolves the origin worker assigned to an agent task, or None.
OriginWorkerOfTask = Callable[[str], str | None]
# Resolves the worker serving a replica incarnation, or None when it is gone.
ServeWorkerOf = Callable[[ReplicaIncarnation], str | None]


@dataclass
class ResidentWorkerDelivery:
    """What the service needs to carry a resident boundary over the worker-owned path.

    Present only when the network plane is enabled; resident-capacity control requires
    it. The resolvers map an agent task to its origin worker, a replica to its serving
    worker, and a worker to its node — in a single-node deployment every node is the
    root node. ``forward_api_key`` lets a keyless sidecar stand-in reach a keyed
    upstream.
    """

    relay: WorkerRelay
    origin_worker_of_task: OriginWorkerOfTask
    serve_worker_of: ServeWorkerOf
    node_of_worker: NodeOfWorker
    network: RouteResolver
    sessions: ResidentSessionWriter
    directly_routable: bool = False
    forward_api_key: str | None = None


class IngressDelivery(Protocol):
    """The per-request seam an ingress origination binds for its result and re-drive.

    An ingress invocation has no ``DS`` boundary to settle back to, so control routes an
    ingress attempt's outcome through this handle: the authorized response frames tee to
    the client, the fenced terminal records a durable ingress-terminal fact, and a lost
    attempt re-drives the origination. The handle carries no engine or payload
    semantics; it moves opaque frames the ingress relays to the client unparsed.
    """

    def tee(self, payload: str) -> None:
        """Relay one authorized response frame to the client, unparsed."""
        ...

    def record_terminal(self, reason: ClaimTerminalReason, detail: str | None) -> None:
        """Record the fenced ingress-terminal fact before the credit releases."""
        ...

    def complete(self) -> None:
        """Finalize the client response on a successful terminal."""
        ...

    def fail(self, detail: str) -> None:
        """Finalize the client response on a definite failure."""
        ...

    def redrive(self) -> None:
        """Re-drive the origination after an uncertain loss, under the held claim."""
        ...


@dataclass
class IngressOrigination:
    """One authenticated ingress request driven through resident admission.

    The ingress edge selects the designated origin worker (a placement decision, never a
    replica or credit) and holds the raw request; control admits the same claim and
    drives the same two-phase delivery as a workflow consumer. ``task_id`` and
    ``call_correlation`` are the worker-lane correlation the designated worker keys its
    private request custody by.
    """

    invocation_id: str
    idempotency_key: str
    task_id: str
    call_correlation: str
    subject: InvocationSubject
    dependency: ServiceDependency
    profile: AdmissionProfile
    origin_worker: str
    delivery: IngressDelivery


@dataclass
class _Origination:
    """The subject-neutral identity and routing an origination drives its claim."""

    task_id: str
    call_correlation: str
    invocation_id: str
    idempotency_key: str | None
    subject: InvocationSubject
    origin_worker: str | None
    ingress: IngressDelivery | None = None


@dataclass
class _Attempt:
    """The in-memory per-invocation state one origination binds for its later reports.

    It carries the fence subject (``origin_id``) and the relay wiring so the ack handler
    mints a matching authorization and the terminal reap relays a cancel and drops the
    session record. It is rebuilt by a re-drive after a restart, so a lost entry only
    ignores a stale report rather than releasing a credit. An ingress attempt also
    carries its delivery handle so the outcome tees, terminalizes, and re-drives through
    the edge.
    """

    task_id: str
    call_correlation: str
    invocation_id: str
    idempotency_key: str | None
    session_id: str
    origin_worker: str
    serve_worker: str
    origin_id: str
    deadline_at: str | None
    replica_id: str
    adapter_ref: str | None
    subject: InvocationSubject
    ingress: IngressDelivery | None = None


class ResidentCapacityControl:
    """The resident boundary origination seam and the two admission/lifecycle actors."""

    def __init__(
        self,
        *,
        stores: ResidentStores,
        admission: AdmissionController,
        lifecycle: LifecycleScaleManager,
        limits: ResidentPolicyLimits,
        dependency_resolver: DependencyResolver,
        settle_cb: SettleCallback,
        redispatch_cb: RedispatchCallback,
        endpoint_probe: EndpointProbe,
        delivery: ResidentWorkerDelivery | None = None,
        persist: PersistCallback | None = None,
        logger: logging.Logger | None = None,
        poll_interval_sec: float = 1.0,
        idle_sweep_interval_sec: float = 0.0,
        redrive_backoff_sec: float = 0.5,
        max_transient_redrives: int = 3,
    ) -> None:
        self._stores = stores
        self._admission = admission
        self._lifecycle = lifecycle
        self._limits = limits
        self._resolve_dependency = dependency_resolver
        self._settle = settle_cb
        self._redispatch = redispatch_cb
        self._probe_endpoint = endpoint_probe
        self._delivery = delivery
        self._persist = persist or (lambda: None)
        self._logger = logger or logging.getLogger("resident-capacity")
        self._poll_interval = poll_interval_sec
        self._idle_sweep_interval = idle_sweep_interval_sec
        self._redrive_backoff = redrive_backoff_sec
        self._max_transient_redrives = max_transient_redrives
        self._transient_failures: dict[str, int] = {}
        self._attempts: dict[str, _Attempt] = {}
        # Every deputy an ingress invocation injected its request into, so a re-drive
        # onto a fresh deputy reaps them all (the request key is stable per attempt).
        self._ingress_deputies: dict[str, set[str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._admit_lock = asyncio.Lock()
        self._sweep_task: asyncio.Task[None] | None = None

    def set_worker_delivery(self, delivery: ResidentWorkerDelivery) -> None:
        """Enable the worker-owned data path once the network plane is available."""
        self._delivery = delivery

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop the origination coroutines run on."""
        self._loop = loop

    def start(self) -> None:
        """Begin the background idle-teardown sweep when a sweep interval is set."""
        if self._loop is None or self._sweep_task is not None:
            return
        if self._idle_sweep_interval <= 0:
            return
        self._sweep_task = self._loop.create_task(self._maintenance_loop())

    def shutdown(self) -> None:
        """Cancel the background idle-teardown sweep."""
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            self._sweep_task = None

    async def _maintenance_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._idle_sweep_interval)
                try:
                    self._lifecycle.sweep_idle()
                except Exception:
                    self._logger.exception("resident idle sweep failed")
        except asyncio.CancelledError:
            return

    def originate(self, env: ToolInvocationEnvelope) -> None:
        """Originate a worker-captured resident boundary through resident admission."""
        if self._loop is None:
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error="resident-capacity control is not running",
            )
            return
        asyncio.run_coroutine_threadsafe(self._originate(env), self._loop)

    def originate_ingress(self, request: IngressOrigination) -> None:
        """Originate an authenticated ingress request through resident admission.

        The ingress edge has already authenticated the principal, resolved a published
        alias, and selected the designated origin worker; this drives the same admission
        gate and two-phase delivery as a workflow consumer.
        """
        if self._loop is None:
            request.delivery.fail("resident-capacity control is not running")
            return
        asyncio.run_coroutine_threadsafe(self._originate_ingress(request), self._loop)

    def inject_ingress_request(
        self, worker_id: str, task_id: str, call_correlation: str, request: str
    ) -> bool:
        """Place an ingress request into its designated deputy's worker-private custody.

        The raw request rides an opaque control frame to the deputy, where the origin
        driver peeks it like a worker that captured its own boundary; it never enters a
        durable fact. Returns whether the frame was delivered.
        """
        if self._delivery is None:
            return False
        return self._delivery.relay(
            worker_id,
            "resident_request_inject",
            {
                "task_id": task_id,
                "call_correlation": call_correlation,
                "request": request,
            },
        )

    async def _originate_ingress(self, request: IngressOrigination) -> None:
        try:
            await self._originate_ingress_inner(request)
        except Exception as exc:
            self._logger.exception(
                "resident ingress origination failed for invocation %s",
                request.invocation_id,
            )
            request.delivery.fail(f"resident ingress origination error: {exc}")

    def on_bootstrap_ack(self, ack: ResidentBootstrapAck) -> None:
        """Consume an origin worker's bootstrap-phase report off the calling lane."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._on_ack(ack), self._loop)

    def on_outcome(self, outcome: ResidentOpOutcome) -> None:
        """Consume an origin worker's fenced terminal report off the calling lane."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._on_outcome(outcome), self._loop)

    def on_invocation_terminal(self, invocation_id: str, failed: bool = False) -> None:
        """Release the admission credit from a fenced DS terminal outcome.

        Wired on both the success and the failure/cancel settlement of a resident
        boundary, so every fenced terminal — not only a completion — releases the
        credit. The settling thread is not the origination lane, so the admission-state
        mutation is marshaled onto the loop the origination coroutines run on, keeping
        every access to the claim store and attempt map single-threaded.
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._settle_terminal_local, invocation_id, failed
            )
        else:
            self._settle_terminal_local(invocation_id, failed)

    def _settle_terminal_local(self, invocation_id: str, failed: bool) -> None:
        reason = ClaimTerminalReason.FAILED if failed else ClaimTerminalReason.COMPLETED
        self._admission.settle_invocation_terminal(invocation_id, reason)
        self._transient_failures.pop(invocation_id, None)
        self._reap_attempt(invocation_id)

    def _reap_attempt(self, invocation_id: str) -> None:
        """Reap both ends of a resident invocation on its fenced terminal.

        The origin reap cancels the origin driver's lane and drops the worker-private
        raw request; the serve-worker reap tears down the replica's serve task and its
        engine request; then the durable session record is deleted. Every step is best
        effort — a gone worker simply has nothing to reap. An ingress re-drive may
        inject the request into several deputies under one invocation, so every deputy
        it touched is reaped, not only the last.
        """
        deputies = self._ingress_deputies.pop(invocation_id, set())
        attempt = self._attempts.pop(invocation_id, None)
        if attempt is None or self._delivery is None:
            return
        for worker in {attempt.origin_worker} | deputies:
            self._delivery.relay(
                worker,
                "resident_reap",
                {
                    "task_id": attempt.task_id,
                    "call_correlation": attempt.call_correlation,
                },
            )
        self._delivery.relay(
            attempt.serve_worker,
            "resident_sidecar_reap",
            {"invocation_id": attempt.invocation_id},
        )
        self._reclaim_adapter_slot(attempt)
        if self._loop is not None:
            self._loop.create_task(self._delivery.sessions.delete(attempt.session_id))

    def _reclaim_adapter_slot(self, attempt: _Attempt) -> None:
        """Unload the invocation's adapter iff its last credit-bearing claim released.

        The credit is released before the reap, so the held-adapter set no longer
        counts this invocation: if no remaining claim on the replica references the
        adapter, the engine slot may be freed. A concurrent same-adapter claim still
        holds the slot, so the adapter is never unloaded out from under a peer.
        """
        if self._delivery is None or attempt.adapter_ref is None:
            return
        if self._lifecycle.replica_holds_adapter(
            attempt.replica_id, attempt.adapter_ref
        ):
            return
        self._delivery.relay(
            attempt.serve_worker,
            "resident_adapter_unload",
            {
                "replica_id": attempt.replica_id,
                "adapter_name": attempt.adapter_ref,
            },
        )

    def list_service_families(self) -> list[ServiceFamily]:
        """The registered service families, for operator read access."""
        return self._stores.families.all()

    def list_replica_incarnations(self) -> list[ReplicaIncarnation]:
        """Every replica incarnation in the directory, inert ones included."""
        return self._stores.directory.all()

    def list_credit_bearing_claims(self) -> tuple[list[ServiceClaim], dict[str, int]]:
        """The credit-bearing claims and the per-replica held credit derived on read.

        The held count recomputes from the authoritative claims through the credit
        ledger; it is never a stored counter.
        """
        claims = [c for c in self._stores.claims.all() if c.holds_credit]
        held = {
            replica_id: self._stores.credit_ledger.held(replica_id)
            for replica_id in {c.replica_id for c in claims if c.replica_id is not None}
        }
        return claims, held

    def rehydrate(self, snapshot: ResidentSnapshot) -> None:
        """Rebuild the authoritative CS facts and reconcile in-flight claims after a
        restart.

        A credit-bearing claim whose data path did not survive the restart moves to
        ``UNCERTAIN`` rather than being re-admitted fresh, so its credit is not released
        until the linked invocation reaches a fenced terminal outcome.
        """
        self._stores.load_snapshot(snapshot)
        # Reports are not snapshotted and endpoint credentials are not persisted:
        # re-probe each servable replica to re-attach its endpoint and re-report
        # capacity so a warm replica is admittable again. A serve task that no longer
        # reports an endpoint is gone, so invalidate the incarnation to re-materialize.
        for replica in self._stores.directory.all():
            if (
                replica.state not in SERVABLE_REPLICA_STATES
                or replica.serve_task_id is None
            ):
                continue
            if (fresh := self._probe_endpoint(replica.serve_task_id)) is None:
                self._lifecycle.on_preempt(replica.replica_id)
                continue
            replica.endpoint = fresh
            self._lifecycle.refresh_report(replica.replica_id)
        for claim in self._stores.claims.all():
            if claim.state in (
                ClaimState.RESERVED,
                ClaimState.ACCEPTED,
                ClaimState.STREAMING,
            ):
                self._admission.on_route_loss(claim)

    async def _originate(self, env: ToolInvocationEnvelope) -> None:
        """Originate one resident boundary, settling an error at its call on any escape.

        The admit, sidecar-bind, and route-resolve paths settle or hold their own typed
        dispositions; this guard catches every other escape (binding resolution, claim
        raise/resume, endpoint probe) so the originating agent call always settles or
        holds instead of hanging.
        """
        try:
            await self._originate_inner(env)
        except Exception as exc:
            self._logger.exception(
                "resident origination failed for invocation %s", env.invocation_id
            )
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error=f"resident origination error: {exc}",
            )

    async def _originate_inner(self, env: ToolInvocationEnvelope) -> None:
        resolved = self._resolve_dependency(env.task_id)
        if resolved is None or not resolved[1].service_ref:
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error="resident model binding is unresolved",
            )
            return
        workflow_id, dependency = resolved
        origin_worker = (
            self._delivery.origin_worker_of_task(env.task_id)
            if self._delivery is not None
            else None
        )
        orig = _Origination(
            task_id=env.task_id,
            call_correlation=env.call_correlation,
            invocation_id=env.invocation_id,
            idempotency_key=env.idempotency_key,
            subject=InvocationSubject(
                kind=InvocationSubjectKind.WORKFLOW, id=workflow_id
            ),
            origin_worker=origin_worker,
        )
        profile = AdmissionProfile(
            engine_batch_key=dependency.engine_batch_key,
            adapter_ref=dependency.adapter,
            adapter_source=dependency.adapter_source,
        )
        await self._drive_claim(orig, dependency, profile)

    async def _drive_claim(
        self,
        orig: _Origination,
        dependency: ServiceDependency,
        profile: AdmissionProfile,
    ) -> None:
        """Admit the invocation's claim and relay its bootstrap, subject-neutrally.

        A workflow subject and an ingress subject funnel through the same admission
        gate, two-phase relay, and terminal fences; only the settle/re-drive routing
        differs,
        which the origination's delivery handle carries.
        """
        if self._delivery is None:
            self._settle_origination_error(
                orig, "resident-capacity control requires the network plane"
            )
            return
        model_ref = dependency.service_ref
        family = dependency.service_family
        existing = self._admission.active_claim(orig.invocation_id)
        if existing is not None and existing.holds_credit:
            # Resume a re-driven boundary on the in-flight claim: reissue to the same
            # fenced replica under the held credit, never re-admitting or releasing.
            claim = existing
            handoff = self._admission.rebuild_handoff(
                existing, idempotency_key=orig.idempotency_key
            )
            if handoff is None:
                self._admission.on_route_loss(existing)
                self._settle_origination_error(
                    orig, "resident replica is unavailable to resume the invocation"
                )
                return
        else:
            if existing is not None:
                claim = existing
            elif not self._ensure_family(dependency):
                self._fail(
                    orig,
                    ProvisioningDenialReason.MODEL_NOT_ALLOWED,
                    f"model {model_ref!r} is not in the allowed catalog",
                )
                return
            else:
                claim = self._admission.raise_claim(
                    invocation_id=orig.invocation_id,
                    subject=orig.subject,
                    family=family,
                    profile=profile,
                )
            handoff = await self._acquire_capacity(
                orig, family, model_ref, claim, profile
            )
            if handoff is None:
                return
        replica = self._stores.directory.get(handoff.replica_id)
        if replica is None or replica.endpoint is None:
            self._admission.on_route_loss(claim)
            self._settle_origination_error(
                orig, "resident replica endpoint is unavailable"
            )
            return
        await self._relay_bootstrap(orig, claim, profile, handoff, replica)

    async def _originate_ingress_inner(self, request: IngressOrigination) -> None:
        orig = _Origination(
            task_id=request.task_id,
            call_correlation=request.call_correlation,
            invocation_id=request.invocation_id,
            idempotency_key=request.idempotency_key,
            subject=request.subject,
            origin_worker=request.origin_worker,
            ingress=request.delivery,
        )
        await self._drive_claim(orig, request.dependency, request.profile)

    def _settle_origination_error(self, orig: _Origination, detail: str) -> None:
        """Route an origination-phase error to its subject's settle path."""
        if orig.ingress is not None:
            self._finalize_ingress(
                orig.invocation_id,
                orig.ingress,
                ClaimTerminalReason.FAILED,
                detail,
                success=False,
            )
        else:
            self._settle(orig.task_id, orig.call_correlation, None, error=detail)

    def _finalize_ingress(
        self,
        invocation_id: str,
        delivery: IngressDelivery,
        reason: ClaimTerminalReason,
        detail: str | None,
        *,
        success: bool,
    ) -> None:
        """Record the fenced ingress-terminal fact, release the credit, and finalize.

        The durable terminal fact is recorded before the credit releases, so a restart
        reconciles the claim against it. The release is idempotent and the reap
        tolerates an already-gone attempt, so a duplicate or late report is a no-op.
        """
        delivery.record_terminal(reason, detail)
        self._admission.settle_invocation_terminal(invocation_id, reason)
        self._transient_failures.pop(invocation_id, None)
        self._reap_attempt(invocation_id)
        if success:
            delivery.complete()
        else:
            delivery.fail(detail or "resident ingress failed")

    def fail_ingress(
        self, invocation_id: str, delivery: IngressDelivery, detail: str
    ) -> None:
        """Terminalize an ingress invocation whose re-drive gave up.

        A give-up (no deputy available, or the inject relay failed) leaves the claim
        UNCERTAIN holding credit; unlike a workflow, no DS terminal consumes it. This
        releases the credit through the fenced path — a FAILED ingress-terminal fact
        consumed by the same FSM — then fails the client, so the credit never strands.
        """
        self._finalize_ingress(
            invocation_id, delivery, ClaimTerminalReason.FAILED, detail, success=False
        )

    def reconcile_ingress_terminal(
        self, invocation_id: str, reason: ClaimTerminalReason
    ) -> None:
        """Settle a claim left credit-bearing by a crash between its terminal writes.

        On startup a recorded ingress-terminal fact replays through the same FSM so a
        claim rehydrated UNCERTAIN releases; there is no live client to finalize.
        """
        self._admission.settle_invocation_terminal(invocation_id, reason)
        self._reap_attempt(invocation_id)

    async def _relay_bootstrap(
        self,
        orig: _Origination,
        claim: ServiceClaim,
        profile: AdmissionProfile,
        handoff: AdmissionHandoff,
        replica: ReplicaIncarnation,
    ) -> None:
        """Bind the sidecar, resolve the origin fence, and relay the handoff.

        The origin worker carries the request and drives the engine ack over the
        reverse-relay; an unreachable origin, an unbindable sidecar, or an unresolved
        origin holds the credit uncertain and re-drives rather than terminalizing.
        """
        deps = self._delivery
        assert deps is not None
        origin_worker = orig.origin_worker
        origin_node = deps.node_of_worker(origin_worker)
        if origin_worker is None or origin_node is None:
            await self._hold_and_redrive(orig, claim, "no origin worker for boundary")
            return
        target_worker = deps.serve_worker_of(replica)
        target_node = deps.node_of_worker(target_worker)
        if target_worker is None or target_node is None:
            await self._hold_and_redrive(orig, claim, "resident replica worker is gone")
            return
        listener = await self._ensure_sidecar(replica, target_worker, target_node)
        if listener is None:
            await self._hold_and_redrive(orig, claim, "resident sidecar is unavailable")
            return
        resolved = await deps.network.resolve(origin_node, listener)
        if resolved is None:
            await self._hold_and_redrive(
                orig, claim, "no origin route for the boundary"
            )
            return
        origin, _route = resolved
        handoff = handoff.model_copy(
            update={
                "origin_id": origin.origin_id,
                "listener_generation": listener.listener_generation,
            }
        )
        # A fresh relay session per delivery attempt: a re-drive gets its own session,
        # so its bridge and per-direction sequence never collide with an old one.
        session_id = new_relay_session_id()
        await deps.sessions.update(
            session_id,
            origin_node=origin_node,
            target_node=target_node,
            origin_worker=origin_worker,
            target_worker=target_worker,
            invocation_id=orig.invocation_id,
            idm=orig.idempotency_key or "",
        )
        self._attempts[orig.invocation_id] = _Attempt(
            task_id=orig.task_id,
            call_correlation=orig.call_correlation,
            invocation_id=orig.invocation_id,
            idempotency_key=orig.idempotency_key,
            session_id=session_id,
            origin_worker=origin_worker,
            serve_worker=target_worker,
            origin_id=origin.origin_id,
            deadline_at=profile.deadline_at,
            replica_id=replica.replica_id,
            adapter_ref=profile.adapter_ref,
            subject=orig.subject,
            ingress=orig.ingress,
        )
        if orig.ingress is not None:
            self._ingress_deputies.setdefault(orig.invocation_id, set()).add(
                origin_worker
            )
        delivered = deps.relay(
            origin_worker,
            "resident_handoff",
            {
                "task_id": orig.task_id,
                "call_correlation": orig.call_correlation,
                "session_id": session_id,
                "handoff": handoff.model_dump(mode="json"),
                "tee": orig.ingress is not None,
            },
        )
        if not delivered:
            await self._hold_and_redrive(orig, claim, "origin worker relay failed")

    async def _on_ack(self, ack: ResidentBootstrapAck) -> None:
        """Record the engine enqueue ack and issue the route authorization, or hold.

        An ``ACKED`` report accepts the reserved claim (or reauthorizes a resumed one)
        and relays the immutable fence to the origin worker; a ``REJECTED`` report is a
        definite pre-acceptance refusal that releases the reservation; an ``UNCERTAIN``
        report holds the credit and re-drives. A report that does not match the live
        attempt (a stale or post-restart duplicate) is ignored, leaving the claim
        credit-bearing.
        """
        deps = self._delivery
        attempt = self._attempts.get(ack.invocation_id)
        claim = self._admission.active_claim(ack.invocation_id)
        if deps is None or attempt is None or claim is None:
            return
        if attempt.session_id != ack.session_id:
            return
        if ack.outcome is ResidentBootstrapOutcome.ACKED:
            if claim.state is ClaimState.RESERVED:
                auth = self._admission.accept_and_authorize(
                    claim,
                    idempotency_key=attempt.idempotency_key,
                    origin_id=attempt.origin_id,
                    deadline_at=attempt.deadline_at,
                )
                self._admission.on_stream_started(claim)
            else:
                # A resumed in-flight claim keeps its held credit; re-mint the fence.
                auth = self._admission.reauthorize(
                    claim,
                    idempotency_key=attempt.idempotency_key,
                    origin_id=attempt.origin_id,
                    deadline_at=attempt.deadline_at,
                )
            delivered = deps.relay(
                attempt.origin_worker,
                "resident_authorization",
                {
                    "call_correlation": attempt.call_correlation,
                    "auth": auth.model_dump(mode="json"),
                },
            )
            if not delivered:
                await self._hold_and_redrive_claim(
                    attempt, claim, "authorization relay failed"
                )
            return
        if ack.outcome is ResidentBootstrapOutcome.REJECTED:
            # A definite fence rejection releases the reservation and invalidates the
            # incarnation so the family re-materializes; it is never re-driven.
            self._release_definite(
                attempt,
                claim,
                f"resident bootstrap refused: {ack.rejection or 'unknown'}",
                pre_acceptance=True,
                preempt=True,
            )
            return
        await self._hold_and_redrive_claim(
            attempt, claim, "resident bootstrap delivery is uncertain"
        )

    async def _on_outcome(self, outcome: ResidentOpOutcome) -> None:
        """Settle the boundary from the origin worker's fenced terminal, or hold.

        A workflow ``SUCCESS`` settles by reference from the completed manifest and the
        fenced DS terminal releases the credit; an ingress ``SUCCESS`` records the
        fenced ingress-terminal fact, releases the credit, and finalizes the client
        response (whose frames already teed live). ``DEFINITE_FAILURE`` settles an error
        the same way per subject. ``UNCERTAIN`` holds the credit and re-drives. A report
        that does not match the live attempt is ignored, leaving the boundary pending
        for a same-invocation re-drive.
        """
        attempt = self._attempts.get(outcome.invocation_id)
        claim = self._admission.active_claim(outcome.invocation_id)
        if attempt is None or attempt.session_id != outcome.session_id:
            return
        if outcome.status is ResidentStreamStatus.SUCCESS:
            if outcome.manifest is None:
                return
            if attempt.ingress is not None:
                self._finalize_ingress(
                    attempt.invocation_id,
                    attempt.ingress,
                    ClaimTerminalReason.COMPLETED,
                    None,
                    success=True,
                )
            else:
                self._settle(
                    attempt.task_id, attempt.call_correlation, ref=outcome.manifest
                )
            return
        if outcome.status is ResidentStreamStatus.DEFINITE_FAILURE:
            detail = f"resident invocation failed: {outcome.error or 'unknown'}"
            if attempt.ingress is not None:
                self._finalize_ingress(
                    attempt.invocation_id,
                    attempt.ingress,
                    ClaimTerminalReason.FAILED,
                    detail,
                    success=False,
                )
            else:
                self._settle(
                    attempt.task_id, attempt.call_correlation, None, error=detail
                )
            return
        if claim is not None:
            await self._hold_and_redrive_claim(
                attempt, claim, outcome.error or "resident stream uncertain"
            )

    def on_stream_chunk(self, chunk: ResidentStreamChunk) -> None:
        """Tee one authorized response frame to a live ingress request's client."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._tee_chunk, chunk)

    def _tee_chunk(self, chunk: ResidentStreamChunk) -> None:
        attempt = self._attempts.get(chunk.invocation_id)
        if (
            attempt is None
            or attempt.ingress is None
            or attempt.session_id != chunk.session_id
        ):
            return
        attempt.ingress.tee(chunk.payload)

    async def _hold_and_redrive(
        self, orig: _Origination, claim: ServiceClaim, detail: str
    ) -> None:
        """Hold the credit uncertain and re-drive the boundary under its held claim."""
        await self._hold_locked(
            orig.invocation_id,
            orig.task_id,
            orig.call_correlation,
            claim,
            detail,
            orig.ingress,
        )

    async def _hold_and_redrive_claim(
        self, attempt: _Attempt, claim: ServiceClaim, detail: str
    ) -> None:
        await self._hold_locked(
            attempt.invocation_id,
            attempt.task_id,
            attempt.call_correlation,
            claim,
            detail,
            attempt.ingress,
        )

    async def _hold_locked(
        self,
        invocation_id: str,
        task_id: str,
        call_correlation: str,
        claim: ServiceClaim,
        detail: str,
        ingress: IngressDelivery | None,
    ) -> None:
        """Hold the credit uncertain and re-drive the boundary under its held claim.

        A transient or ambiguous loss neither completes nor releases: the claim stays
        uncertain and the boundary re-drives to resume on the same fenced replica. A
        path that keeps failing preempts the replica so the next resume's rebuild
        returns None and the fenced terminal releases — the hold is bounded by replica
        health, never a timer that could release while the engine still holds the slot.
        """
        if claim.state is ClaimState.TERMINAL:
            # A concurrent terminal (e.g. a cancel) already settled the boundary and
            # released the credit; there is nothing to hold or re-drive.
            return
        self._admission.on_route_loss(claim)
        count = self._transient_failures.get(invocation_id, 0) + 1
        self._transient_failures[invocation_id] = count
        if count >= self._max_transient_redrives and claim.replica_id is not None:
            self._lifecycle.on_preempt(claim.replica_id)
        self._logger.info(
            "resident delivery held uncertain (attempt %d): %s", count, detail
        )
        await asyncio.sleep(self._redrive_backoff)
        if ingress is not None:
            ingress.redrive()
        else:
            self._redispatch(task_id, call_correlation)

    def _release_definite(
        self,
        attempt: _Attempt,
        claim: ServiceClaim,
        detail: str,
        *,
        pre_acceptance: bool,
        preempt: bool,
    ) -> None:
        """Terminalize a definite failure so the fenced terminal releases the credit.

        A pre-acceptance refusal releases the still-reserved credit directly; the settle
        then terminalizes the boundary, and the fenced terminal is idempotent over the
        already-released claim.
        """
        if pre_acceptance and claim.state is ClaimState.RESERVED:
            self._admission.on_enqueue_failed(claim)
        if preempt and claim.replica_id is not None:
            self._lifecycle.on_preempt(claim.replica_id)
        if attempt.ingress is not None:
            self._finalize_ingress(
                attempt.invocation_id,
                attempt.ingress,
                ClaimTerminalReason.FAILED,
                detail,
                success=False,
            )
        else:
            self._settle(attempt.task_id, attempt.call_correlation, None, error=detail)

    async def _ensure_sidecar(
        self, replica: ReplicaIncarnation, worker_id: str, node_id: str
    ) -> ReplicaListenerAdvertisement | None:
        """Bind (or rebind) the replica's sidecar and stamp its listener advertisement.

        Reuses a listener already advertised for the current incarnation; a superseded
        incarnation forces a rebind under a fresh listener generation. The bind relays
        the replica's incarnation fence and its co-located engine endpoint to the
        serving worker as an ordinary control frame — never a dispatched task.
        """
        deps = self._delivery
        if deps is None or replica.endpoint is None:
            return None
        if (
            replica.listener is not None
            and replica.listener.incarnation == replica.incarnation
        ):
            return replica.listener
        generation = replica.listener_generation + 1
        # The sidecar reaches its co-located engine with the endpoint's own key, or the
        # deployment forward key so a keyless stand-in can still forward to a keyed
        # upstream.
        engine = replica.endpoint
        if engine.api_key is None and deps.forward_api_key is not None:
            engine = engine.model_copy(update={"api_key": deps.forward_api_key})
        family = self._stores.families.get(replica.family)
        interface = family.interface if family is not None else engine.interface
        delivered = deps.relay(
            worker_id,
            "resident_sidecar_bind",
            {
                "replica_id": replica.replica_id,
                "incarnation": replica.incarnation,
                "listener_generation": generation,
                "engine": {
                    "base_url": engine.base_url,
                    "model": engine.model,
                    "api_key": engine.api_key,
                    "interface": interface,
                },
            },
        )
        if not delivered:
            return None
        replica.listener_generation = generation
        replica.listener = ReplicaListenerAdvertisement(
            replica_id=replica.replica_id,
            family=replica.family,
            incarnation=replica.incarnation,
            listener_generation=generation,
            node_id=node_id,
            worker_id=worker_id,
            routes=(f"resident://{worker_id}",),
            protocols=("resident",),
            directly_routable=deps.directly_routable,
        )
        self._persist()
        return replica.listener

    def _ensure_family(self, dependency: ServiceDependency) -> bool:
        family = dependency.service_family
        if family in self._stores.families:
            return True
        model_ref = dependency.service_ref
        if self._limits.allowed_models and model_ref not in self._limits.allowed_models:
            return False
        self._stores.families.register(
            ServiceFamily(
                family=family,
                engine_batch_key=dependency.engine_batch_key,
                model_ref=model_ref,
                interface=dependency.interface.value,
                isolation=dependency.isolation,
                selection_strategy=self._limits.selection_strategy,
            )
        )
        return True

    async def _acquire_capacity(
        self,
        orig: _Origination,
        family: str,
        model_ref: str,
        claim: ServiceClaim,
        profile: AdmissionProfile,
    ) -> AdmissionHandoff | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.cold_start_deadline_sec
        while True:
            async with self._admit_lock:
                self._promote_ready_replicas(family)
                self._lifecycle.refresh_family_reports(family)
                handoff = self._admission.admit(
                    claim, profile, idempotency_key=orig.idempotency_key
                )
                if handoff is not None:
                    return handoff
                plan = self._lifecycle.plan_capacity(family, model_ref, profile)
                if plan.action == "deny" and plan.denial is not None:
                    self._admission.on_denied(claim)
                    self._fail(orig, plan.denial.reason, plan.denial.detail or "")
                    return None
                if plan.action == "materialize" and not self._has_materializing(family):
                    definition = self._stores.families.get(family)
                    if definition is not None:
                        try:
                            await self._lifecycle.materialize(definition)
                        except Exception as exc:  # cold start could not be started
                            self._admission.on_denied(claim)
                            self._fail(
                                orig,
                                ProvisioningDenialReason.RESOURCE_CAP,
                                f"resident materialization failed: {exc}",
                            )
                            return None
            if loop.time() >= deadline:
                self._admission.on_expired(claim)
                self._fail(
                    orig,
                    ProvisioningDenialReason.COLD_START_BUDGET,
                    "resident cold start did not become ready in time",
                )
                return None
            await asyncio.sleep(self._poll_interval)

    def _has_materializing(self, family: str) -> bool:
        return any(
            r.state is ReplicaState.MATERIALIZING
            for r in self._stores.directory.by_family(family)
        )

    def _promote_ready_replicas(self, family: str) -> None:
        for replica in self._stores.directory.by_family(family):
            if (
                replica.state is ReplicaState.MATERIALIZING
                and replica.serve_task_id is not None
                and (endpoint := self._probe_endpoint(replica.serve_task_id))
                is not None
            ):
                self._lifecycle.on_replica_ready(replica.replica_id, endpoint)

    def _fail(
        self,
        orig: _Origination,
        reason: ProvisioningDenialReason | None,
        detail: str,
    ) -> None:
        label = reason.value if reason is not None else "denied"
        self._logger.info("resident admission denied (%s): %s", label, detail)
        self._settle_origination_error(
            orig, f"resident admission denied ({label}): {detail}"
        )
