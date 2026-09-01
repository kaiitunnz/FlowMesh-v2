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

from shared.tasks.specs import ModelBindingMode

from ..orchestration.tool_dispatch import ToolInvocationEnvelope
from ..task.v2.compiler.agent_binding import service_family_for_ref
from ..task.v2.representations.operators import AgentModelGatewayBinding
from .adapter import AdapterError, EngineInvocationAdapter
from .admission import AdmissionController
from .lifecycle import LifecycleScaleManager
from .policy import ResidentPolicyLimits
from .state import (
    SERVABLE_REPLICA_STATES,
    AdmissionHandoff,
    AdmissionProfile,
    ClaimState,
    ClaimTerminalReason,
    ProvisioningDenialReason,
    ReplicaEndpoint,
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
# Reads a serve substrate's reported endpoint once ready, else None.
EndpointProbe = Callable[[str], ReplicaEndpoint | None]
# Persists the authoritative CS snapshot.
PersistCallback = Callable[[], None]


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
        endpoint_probe: EndpointProbe,
        logger: logging.Logger | None = None,
        poll_interval_sec: float = 1.0,
        idle_sweep_interval_sec: float = 0.0,
    ) -> None:
        self._stores = stores
        self._admission = admission
        self._lifecycle = lifecycle
        self._adapter = adapter
        self._limits = limits
        self._resolve_binding = binding_resolver
        self._settle = settle_cb
        self._probe_endpoint = endpoint_probe
        self._logger = logger or logging.getLogger("resident-capacity")
        self._poll_interval = poll_interval_sec
        self._idle_sweep_interval = idle_sweep_interval_sec
        self._loop: asyncio.AbstractEventLoop | None = None
        self._admit_lock = asyncio.Lock()
        self._sweep_task: asyncio.Task[None] | None = None

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
                    await self._lifecycle.sweep_idle()
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
            handoff = self._admission.rebuild_handoff(existing)
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
        try:
            completion = await self._adapter.issue(handoff, env.request_payload)
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
                # A post-acceptance loss (or a failure on a resumed claim) holds the
                # credit; the settle below records the fenced DS terminal releasing it.
                self._admission.on_route_loss(claim)
            self._settle(
                env.task_id, env.call_correlation, None, error=f"resident issue: {exc}"
            )
            return
        if claim.state is ClaimState.RESERVED:
            self._admission.on_enqueue_ack(claim)
        self._settle(env.task_id, env.call_correlation, completion)

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
                handoff = self._admission.admit(claim, profile)
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
