"""Resident-capacity control wired to the mediated model-settle seam.

A resident model binding routes here instead of the external agent-model gateway. The
service raises a durable ServiceClaim for the invocation the engine already minted,
drives
the Admission controller and Lifecycle & scale manager, delivers the request through the
inference adapter, and settles the outcome back at the originating call. Execution
defaults
to the in-server relay; the credit releases only when the fenced DS terminal returns
through
``invocation_id``. It runs the invocation off the calling lane, mirroring the gateway.
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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._admit_lock = asyncio.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop the off-lane invocation coroutines run on."""
        self._loop = loop

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

    def on_invocation_terminal(self, invocation_id: str) -> None:
        """Release the admission credit from a fenced DS terminal outcome."""
        self._admission.on_ds_terminal(invocation_id, ClaimTerminalReason.COMPLETED)

    def rehydrate(self, snapshot: ResidentSnapshot) -> None:
        """Rebuild the authoritative CS facts and reconcile in-flight claims after a
        restart.

        A credit-bearing claim whose adapter call did not survive the restart moves to
        ``UNCERTAIN`` rather than being re-admitted fresh, so its credit is not released
        until the linked invocation reaches a fenced terminal outcome.
        """
        self._stores.load_snapshot(snapshot)
        for claim in self._stores.claims.all():
            if claim.state in (
                ClaimState.RESERVED,
                ClaimState.ACCEPTED,
                ClaimState.STREAMING,
            ):
                self._admission.on_route_loss(claim)

    async def _serve_invocation(self, env: ToolInvocationEnvelope) -> None:
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
        if not self._ensure_family(family, model_ref):
            self._fail(
                env,
                ProvisioningDenialReason.MODEL_NOT_ALLOWED,
                f"model {model_ref!r} is not in the allowed catalog",
            )
            return
        profile = AdmissionProfile(engine_batch_key=family)
        claim = self._admission.raise_claim(
            invocation_id=env.invocation_id,
            workflow_id=workflow_id,
            family=family,
            profile=profile,
        )
        handoff = await self._acquire_capacity(env, family, model_ref, claim, profile)
        if handoff is None:
            return
        try:
            completion = await self._adapter.issue(handoff, env.request_payload)
        except AdapterError as exc:
            if exc.pre_acceptance:
                self._admission.on_enqueue_failed(claim)
            else:
                self._admission.on_route_loss(claim)
                self._admission.reconcile(env.invocation_id)
            self._settle(
                env.task_id, env.call_correlation, None, error=f"resident issue: {exc}"
            )
            return
        self._admission.on_enqueue_ack(claim)
        self._settle(env.task_id, env.call_correlation, completion)

    def _ensure_family(self, family: str, model_ref: str) -> bool:
        if family in self._stores.families:
            return True
        if self._limits.allowed_models and model_ref not in self._limits.allowed_models:
            return False
        self._stores.families.register(
            ServiceFamily(family=family, engine_batch_key=family, model_ref=model_ref)
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
                        await self._lifecycle.materialize(definition)
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
