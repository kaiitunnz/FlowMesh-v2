"""Resident-capacity control wired to the mediated model-settle seam.

A resident model binding routes here instead of the external agent-model gateway. The
service raises a durable ServiceClaim for the invocation the engine already minted,
drives the Admission controller and Lifecycle & scale manager, delivers the request
through the inference adapter off the calling lane, and settles the outcome back at the
originating call. Execution defaults to the in-server relay; the credit releases only
when the fenced DS terminal returns through ``invocation_id``.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.tasks.specs import ModelBindingMode

from ..network.state import (
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteObservationOutcome,
    RouteOrigin,
    Transport,
)
from ..orchestration.tool_dispatch import ToolInvocationEnvelope
from ..task.v2.compiler.agent_binding import service_family_for_ref
from ..task.v2.representations.operators import AgentModelGatewayBinding
from .adapter import AdapterError, EngineInvocationAdapter
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .native import NativeTransport, NativeTransportError
from .policy import ResidentPolicyLimits
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionHandoff,
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    ProvisioningDenialReason,
    ReplicaEndpoint,
    ReplicaIncarnation,
    ReplicaState,
    ResidentSnapshot,
    ServiceClaim,
    ServiceFamily,
)
from .stores import ResidentStores

# Resolves a task's pinned model binding: (workflow_id, binding) or None.
BindingResolver = Callable[[str], tuple[str, AgentModelGatewayBinding] | None]
# Settles a mediated boundary back at its originating call, or fails it with an error.
SettleCallback = Callable[..., bool]
# Re-drives a still-pending mediated boundary off-lane without settling it.
RedispatchCallback = Callable[[str, str], bool]
# Reads a serve substrate's reported endpoint once ready, else None.
EndpointProbe = Callable[[str], ReplicaEndpoint | None]
# Persists the authoritative CS snapshot.
PersistCallback = Callable[[], None]


class RouteResolver(Protocol):
    """The control-plane route resolution the native delivery consumes.

    Satisfied structurally by the network plane; the resident path binds it rather than
    re-deriving endpoints, reachability, or the resolver.
    """

    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute] | None: ...

    def record_observations(
        self,
        origin: RouteOrigin,
        listener: ReplicaListenerAdvertisement,
        observations: list[tuple[Transport, RouteObservationOutcome]],
    ) -> None: ...


@dataclass
class NativeDeliveryDeps:
    """What the service needs to carry a resident invocation over the fabric path.

    Present only when the network plane is enabled; its absence selects the in-server
    compatibility path. The two node resolvers map a workflow task to its origin node
    and a replica to its host node — in a single-node deployment both are the root node.
    """

    network: RouteResolver
    transport: NativeTransport
    origin_node_of_task: Callable[[str], str | None]
    node_of_replica: Callable[[ReplicaIncarnation], str | None]
    sidecar_bind_host: str = "127.0.0.1"
    directly_routable: bool = False
    forward_api_key: str | None = None


class ResidentCapacityControl:
    """The resident model-settle entrypoint and the two admission/lifecycle actors."""

    def __init__(
        self,
        *,
        stores: ResidentStores,
        admission: AdmissionController,
        lifecycle: LifecycleScaleManager,
        adapter: EngineInvocationAdapter,
        limits: ResidentPolicyLimits,
        binding_resolver: BindingResolver,
        settle_cb: SettleCallback,
        redispatch_cb: RedispatchCallback,
        endpoint_probe: EndpointProbe,
        native_delivery: NativeDeliveryDeps | None = None,
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
        self._adapter = adapter
        self._limits = limits
        self._resolve_binding = binding_resolver
        self._settle = settle_cb
        self._redispatch = redispatch_cb
        self._probe_endpoint = endpoint_probe
        self._native = native_delivery
        self._persist = persist or (lambda: None)
        self._logger = logger or logging.getLogger("resident-capacity")
        self._poll_interval = poll_interval_sec
        self._idle_sweep_interval = idle_sweep_interval_sec
        self._redrive_backoff = redrive_backoff_sec
        self._max_transient_redrives = max_transient_redrives
        self._transient_failures: dict[str, int] = {}
        self._live_sessions: dict[str, tuple[str, str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._admit_lock = asyncio.Lock()
        self._sweep_task: asyncio.Task[None] | None = None

    def set_native_delivery(self, deps: NativeDeliveryDeps) -> None:
        """Enable the native fabric data path once the network plane is available."""
        self._native = deps

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop the off-lane invocation coroutines run on."""
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

    def is_resident(self, task_id: str) -> bool:
        """Whether a task's pinned model binding is served by resident capacity."""
        resolved = self._resolve_binding(task_id)
        return resolved is not None and resolved[1].mode is ModelBindingMode.RESIDENT

    def settle(self, env: ToolInvocationEnvelope) -> None:
        """Serve a resident model invocation off the calling lane."""
        if self._loop is None:
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error="resident-capacity control is not running",
            )
            return
        asyncio.run_coroutine_threadsafe(self._serve_invocation(env), self._loop)

    def on_invocation_terminal(self, invocation_id: str, failed: bool = False) -> None:
        """Release the admission credit from a fenced DS terminal outcome.

        Wired on both the success and the failure/cancel settlement of a resident
        boundary, so every fenced terminal — not only a completion — releases the
        credit.
        """
        reason = ClaimTerminalReason.FAILED if failed else ClaimTerminalReason.COMPLETED
        self._admission.on_ds_terminal(invocation_id, reason)
        self._transient_failures.pop(invocation_id, None)
        live = self._live_sessions.pop(invocation_id, None)
        if (
            failed
            and live is not None
            and self._native is not None
            and self._loop is not None
        ):
            # A fenced failure/cancel terminal reaps a still-held native session best
            # effort so the engine stream stops promptly; the deputy reaper is the
            # restart-safe backstop when this poke does not land.
            origin_node, session_id = live
            asyncio.run_coroutine_threadsafe(
                self._reap_native(origin_node, session_id), self._loop
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

        A credit-bearing claim whose adapter call did not survive the restart moves to
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

    async def _serve_invocation(self, env: ToolInvocationEnvelope) -> None:
        """Serve one resident invocation, settling an error at its call on any escape.

        The adapter and materialize paths settle their own typed outcomes; this guard
        catches every other escape (binding resolution, claim raise/resume, endpoint
        probe) so the originating agent call always settles instead of hanging.
        """
        try:
            await self._serve_invocation_inner(env)
        except Exception as exc:
            self._logger.exception(
                "resident serve failed for invocation %s", env.invocation_id
            )
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error=f"resident serve error: {exc}",
            )

    async def _serve_invocation_inner(self, env: ToolInvocationEnvelope) -> None:
        resolved = self._resolve_binding(env.task_id)
        if resolved is None or resolved[1].service_model_ref is None:
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error="resident model binding is unresolved",
            )
            return
        workflow_id, binding = resolved
        model_ref = binding.service_model_ref
        assert model_ref is not None
        family = service_family_for_ref(model_ref)

        profile = AdmissionProfile(engine_batch_key=family)
        existing = self._admission.active_claim(env.invocation_id)
        if existing is not None and existing.holds_credit:
            # Resume a re-driven boundary on the in-flight claim: reissue to the same
            # fenced replica under the held credit, never re-admitting or releasing.
            claim = existing
            handoff = self._admission.rebuild_handoff(
                existing, idempotency_key=env.idempotency_key
            )
            if handoff is None:
                self._admission.on_route_loss(existing)
                self._settle(
                    env.task_id,
                    env.call_correlation,
                    None,
                    error="resident replica is unavailable to resume the invocation",
                )
                return
        else:
            if existing is not None:
                claim = existing
            elif not self._ensure_family(family, model_ref):
                self._fail(
                    env,
                    ProvisioningDenialReason.MODEL_NOT_ALLOWED,
                    f"model {model_ref!r} is not in the allowed catalog",
                )
                return
            else:
                claim = self._admission.raise_claim(
                    invocation_id=env.invocation_id,
                    workflow_id=workflow_id,
                    family=family,
                    profile=profile,
                )
            handoff = await self._acquire_capacity(
                env, family, model_ref, claim, profile
            )
            if handoff is None:
                return
        replica = self._stores.directory.get(handoff.replica_id)
        if replica is None or replica.endpoint is None:
            self._admission.on_route_loss(claim)
            self._settle(
                env.task_id,
                env.call_correlation,
                None,
                error="resident replica endpoint is unavailable",
            )
            return
        origin_node = (
            self._native.origin_node_of_task(env.task_id)
            if self._native is not None
            else None
        )
        if self._native is not None and origin_node is not None:
            await self._deliver_native(
                env, claim, profile, handoff, replica, origin_node
            )
        else:
            await self._deliver_compat(env, claim, replica)

    async def _deliver_compat(
        self,
        env: ToolInvocationEnvelope,
        claim: ServiceClaim,
        replica: ReplicaIncarnation,
    ) -> None:
        """In-server claim-gated delivery for when the native fabric path is off."""
        assert replica.endpoint is not None
        try:
            completion = await self._adapter.issue(
                replica.endpoint, env.request_payload
            )
        except AdapterError as exc:
            if exc.connection_failure and claim.replica_id is not None:
                # The replica is unreachable: invalidate its incarnation so the next
                # admission re-materializes the family from zero rather than wedging on
                # a dead replica. A transient HTTP status leaves a live replica alone.
                self._lifecycle.on_preempt(claim.replica_id)
            if exc.pre_acceptance and claim.state is ClaimState.RESERVED:
                # A known pre-acceptance enqueue failure of a fresh reservation releases
                # its credit directly; no engine ever received it.
                self._admission.on_enqueue_failed(claim)
            else:
                # A post-acceptance loss marks the credit uncertain, but this
                # single-shot path then settles the boundary, so its fenced terminal
                # releases rather than holding and re-driving as the native path does.
                # That narrower ambiguous-loss window is a tracked follow-up to bring
                # onto the same split.
                self._admission.on_route_loss(claim)
            self._settle(
                env.task_id, env.call_correlation, None, error=f"resident issue: {exc}"
            )
            return
        if claim.state is ClaimState.RESERVED:
            self._admission.on_enqueue_ack(claim)
        self._settle(env.task_id, env.call_correlation, completion)

    async def _deliver_native(
        self,
        env: ToolInvocationEnvelope,
        claim: ServiceClaim,
        profile: AdmissionProfile,
        handoff: AdmissionHandoff,
        replica: ReplicaIncarnation,
        origin_node: str,
    ) -> None:
        """Carry the invocation two-phase over the data-direct fabric path.

        The bytes never cross the server: it resolves a route, pokes the origin deputy
        to deliver the bootstrap, records ``ACCEPTED`` and issues the fence on the ack,
        then pokes the authorized stream. A definite fence rejection releases the
        reservation and invalidates the incarnation; an unreachable path, an ambiguous
        bootstrap, or a seam failure holds the credit uncertain and re-drives.
        """
        deps = self._native
        assert deps is not None
        listener = await self._ensure_sidecar(replica)
        if listener is None:
            await self._hold_and_redrive(env, claim, "resident sidecar is unavailable")
            return
        resolved = await deps.network.resolve(origin_node, listener)
        if resolved is None:
            await self._hold_and_redrive(env, claim, "no route to the resident replica")
            return
        origin, route = resolved
        handoff = handoff.model_copy(
            update={
                "route": route,
                "origin_id": origin.origin_id,
                "listener_generation": listener.listener_generation,
            }
        )
        session_id = f"{claim.invocation_id}:{claim.admission_epoch}"
        try:
            boot = await deps.transport.bootstrap(
                origin_node,
                session_id=session_id,
                route=route,
                handoff=handoff,
                request_payload=env.request_payload,
            )
        except NativeTransportError as exc:
            await self._hold_and_redrive(env, claim, f"bootstrap poke failed: {exc}")
            return
        deps.network.record_observations(origin, listener, boot.observations)
        if boot.acked:
            # Track the held session so a fenced cancellation terminal reaps both ends
            # of the data-direct channel and stops the engine promptly. A claim already
            # settled by a concurrent terminal is not tracked: its credit is gone and
            # nothing would pop the entry again.
            if claim.state is not ClaimState.TERMINAL:
                self._live_sessions[env.invocation_id] = (origin_node, session_id)
            await self._stream_native(
                env, claim, profile, origin, origin_node, session_id
            )
            return
        if boot.rejection is not None:
            # A definite fence rejection releases the reservation and invalidates the
            # incarnation so the family re-materializes; it is never re-driven.
            self._release_definite(
                env,
                claim,
                f"resident bootstrap refused: {boot.rejection}",
                pre_acceptance=True,
                preempt=True,
            )
            return
        # An ambiguous delivery or an unreachable path is uncertain, not a completion:
        # hold the credit and re-drive under the held claim.
        detail = (
            "resident bootstrap delivery is uncertain"
            if boot.uncertain
            else "resident bootstrap unreachable"
        )
        await self._hold_and_redrive(env, claim, detail)

    async def _stream_native(
        self,
        env: ToolInvocationEnvelope,
        claim: ServiceClaim,
        profile: AdmissionProfile,
        origin: RouteOrigin,
        origin_node: str,
        session_id: str,
    ) -> None:
        deps = self._native
        assert deps is not None
        origin_id = origin.origin_id
        if claim.state is ClaimState.RESERVED:
            auth = self._admission.accept_and_authorize(
                claim,
                idempotency_key=env.idempotency_key,
                origin_id=origin_id,
                operation="inference",
                deadline_at=profile.deadline_at,
            )
            self._admission.on_stream_started(claim)
        else:
            # A resumed in-flight claim keeps its held credit; re-mint the fence only.
            auth = self._admission.reauthorize(
                claim,
                idempotency_key=env.idempotency_key,
                origin_id=origin_id,
                operation="inference",
                deadline_at=profile.deadline_at,
            )
        try:
            stream = await deps.transport.stream(
                origin_node, session_id=session_id, auth=auth
            )
        except NativeTransportError as exc:
            await self._hold_and_redrive(env, claim, f"resident stream failed: {exc}")
            return
        if not stream.ok:
            if stream.definite:
                # A definite engine refusal held no slot: release through the fenced
                # terminal and fail the boundary per task policy.
                self._release_definite(
                    env,
                    claim,
                    f"resident engine refused: {stream.rejection or 'unknown'}",
                    pre_acceptance=False,
                    preempt=False,
                )
                return
            # A post-acceptance stream loss is uncertain — the engine may still hold the
            # slot — so hold the credit and re-drive rather than releasing.
            await self._hold_and_redrive(
                env, claim, f"resident stream lost: {stream.rejection or 'unknown'}"
            )
            return
        self._settle(env.task_id, env.call_correlation, stream.completion)

    async def _hold_and_redrive(
        self, env: ToolInvocationEnvelope, claim: ServiceClaim, detail: str
    ) -> None:
        """Hold the credit uncertain and re-drive the boundary under its held claim.

        A transient or ambiguous native loss neither completes nor releases: the claim
        stays uncertain and the boundary re-drives to resume on the same fenced replica.
        A path that keeps failing preempts the replica so the next resume's rebuild
        returns None and the fenced terminal releases — the hold is bounded by replica
        health, never a timer that could release while the engine still holds the slot.
        """
        if claim.state is ClaimState.TERMINAL:
            # A concurrent terminal (e.g. a cancel) already settled the boundary and
            # released the credit; there is nothing to hold or re-drive.
            return
        self._admission.on_route_loss(claim)
        count = self._transient_failures.get(env.invocation_id, 0) + 1
        self._transient_failures[env.invocation_id] = count
        if count >= self._max_transient_redrives and claim.replica_id is not None:
            self._lifecycle.on_preempt(claim.replica_id)
        self._logger.info(
            "resident delivery held uncertain (attempt %d): %s", count, detail
        )
        await asyncio.sleep(self._redrive_backoff)
        self._redispatch(env.task_id, env.call_correlation)

    async def _reap_native(self, origin_node: str, session_id: str) -> None:
        """Reap a held native session so a terminal stops the engine and both ends."""
        deps = self._native
        if deps is None:
            return
        try:
            await deps.transport.cancel(origin_node, session_id=session_id)
        except NativeTransportError:
            pass

    def _release_definite(
        self,
        env: ToolInvocationEnvelope,
        claim: ServiceClaim,
        detail: str,
        *,
        pre_acceptance: bool,
        preempt: bool,
    ) -> None:
        """Terminalize a definite native failure so the fenced terminal releases credit.

        A pre-acceptance refusal releases the still-reserved credit directly; the settle
        then terminalizes the boundary, and the fenced DS terminal is idempotent over
        the already-released claim.
        """
        if pre_acceptance and claim.state is ClaimState.RESERVED:
            self._admission.on_enqueue_failed(claim)
        if preempt and claim.replica_id is not None:
            self._lifecycle.on_preempt(claim.replica_id)
        self._settle(env.task_id, env.call_correlation, None, error=detail)

    async def _ensure_sidecar(
        self, replica: ReplicaIncarnation
    ) -> ReplicaListenerAdvertisement | None:
        """Bind (or rebind) the replica's sidecar and stamp its listener advertisement.

        Reuses a listener already advertised for the current incarnation; a superseded
        incarnation forces a rebind under a fresh listener generation.
        """
        deps = self._native
        if deps is None or replica.endpoint is None:
            return None
        if (
            replica.listener is not None
            and replica.listener.incarnation == replica.incarnation
        ):
            return replica.listener
        node_id = deps.node_of_replica(replica)
        if node_id is None:
            return None
        generation = replica.listener_generation + 1
        # The sidecar reaches its co-located engine with the endpoint's own key, or the
        # deployment forward key so a keyless stand-in can still forward to a keyed
        # upstream — mirroring the in-server adapter.
        engine = replica.endpoint
        if engine.api_key is None and deps.forward_api_key is not None:
            engine = engine.model_copy(update={"api_key": deps.forward_api_key})
        try:
            host, port = await deps.transport.bind_sidecar(
                node_id,
                replica_id=replica.replica_id,
                incarnation=replica.incarnation,
                listener_generation=generation,
                route=f"{deps.sidecar_bind_host}:0",
                engine=engine,
            )
        except NativeTransportError:
            return None
        replica.listener_generation = generation
        replica.listener = ReplicaListenerAdvertisement(
            replica_id=replica.replica_id,
            family=replica.family,
            incarnation=replica.incarnation,
            listener_generation=generation,
            node_id=node_id,
            worker_id=replica.worker_id,
            routes=(f"{host}:{port}",),
            protocols=("resident",),
            directly_routable=deps.directly_routable,
        )
        self._persist()
        return replica.listener

    def _ensure_family(self, family: str, model_ref: str) -> bool:
        if family in self._stores.families:
            return True
        if self._limits.allowed_models and model_ref not in self._limits.allowed_models:
            return False
        self._stores.families.register(
            ServiceFamily(
                family=family,
                engine_batch_key=family,
                model_ref=model_ref,
                selection_strategy=self._limits.selection_strategy,
            )
        )
        return True

    async def _acquire_capacity(
        self,
        env: ToolInvocationEnvelope,
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
                handoff = self._admission.admit(
                    claim, profile, idempotency_key=env.idempotency_key
                )
                if handoff is not None:
                    return handoff
                plan = self._lifecycle.plan_capacity(family, model_ref)
                if plan.action == "deny" and plan.denial is not None:
                    self._admission.on_denied(claim)
                    self._fail(env, plan.denial.reason, plan.denial.detail or "")
                    return None
                if plan.action == "materialize" and not self._has_materializing(family):
                    definition = self._stores.families.get(family)
                    if definition is not None:
                        try:
                            await self._lifecycle.materialize(definition)
                        except Exception as exc:  # cold start could not be started
                            self._admission.on_denied(claim)
                            self._fail(
                                env,
                                ProvisioningDenialReason.RESOURCE_CAP,
                                f"resident materialization failed: {exc}",
                            )
                            return None
            if loop.time() >= deadline:
                self._admission.on_expired(claim)
                self._fail(
                    env,
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
        env: ToolInvocationEnvelope,
        reason: ProvisioningDenialReason | None,
        detail: str,
    ) -> None:
        label = reason.value if reason is not None else "denied"
        self._logger.info("resident admission denied (%s): %s", label, detail)
        self._settle(
            env.task_id,
            env.call_correlation,
            None,
            error=f"resident admission denied ({label}): {detail}",
        )
