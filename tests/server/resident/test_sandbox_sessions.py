"""Admission and reconciliation for a co-located sandbox session's host capacity.

A session's host claim is reconciled against its reserved worker being alive rather than
a data route it never has: a restart over a surviving host resumes it, a host that is
gone settles the claim so a successor admits on live capacity, and neither leaves the
session holding an admission no worker can honor.
"""

import asyncio

from server.resident import (
    AdmissionController,
    AdmissionProfile,
    ClaimState,
    LifecycleScaleManager,
    MaterializedAllocation,
    NoSandboxCapacity,
    ReplicaState,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
    ServiceFamily,
)
from server.resident.service import SandboxSessionOpen
from server.resident.state import (
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaIncarnation,
)
from server.task.v2.representations.operators import ServiceDependency, ServiceInterface

_HOST_WORKER = "wkr-host"
_INVOCATION = "inv-session"
_DEPENDENCY = ServiceDependency(
    service_ref="posix-default", interface=ServiceInterface.SANDBOX
)
_FAMILY = _DEPENDENCY.service_family
_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")


def _build(live_workers: set[str]) -> tuple[ResidentCapacityControl, ResidentStores]:
    stores = ResidentStores()
    limits = ResidentPolicyLimits(max_replicas_per_family=1)

    async def materialize_fn(
        family: ServiceFamily, replica: ReplicaIncarnation
    ) -> MaterializedAllocation:
        # A reservation only ever names a live worker, as the wired selector does.
        if _HOST_WORKER not in live_workers:
            raise NoSandboxCapacity("no sandbox-capable worker")
        return MaterializedAllocation(worker_id=_HOST_WORKER)

    svc = ResidentCapacityControl(
        stores=stores,
        admission=AdmissionController(stores),
        lifecycle=LifecycleScaleManager(
            stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
        ),
        limits=limits,
        dependency_resolver=lambda task_id: ("wfl-1", _DEPENDENCY),
        settle_cb=lambda *a, **k: True,
        redispatch_cb=lambda *a, **k: True,
        endpoint_probe=lambda serve_task_id: None,
        sandbox_worker_probe=lambda worker_id: worker_id in live_workers,
        poll_interval_sec=0.01,
    )
    return svc, stores


def _request() -> SandboxSessionOpen:
    return SandboxSessionOpen(
        invocation_id=_INVOCATION,
        idempotency_key="idm-1",
        task_id="tsk-session",
        subject=_SUBJECT,
        dependency=_DEPENDENCY,
        profile=AdmissionProfile(engine_batch_key=_DEPENDENCY.engine_batch_key),
        deny=lambda detail: None,
    )


def _admitted(svc: ResidentCapacityControl) -> SandboxSessionOpen:
    """Drive one session to an ACCEPTED claim over a reserved host."""
    request = _request()

    async def drive() -> None:
        svc.bind_loop(asyncio.get_running_loop())
        await svc._open_sandbox_session(request)

    asyncio.run(drive())
    return request


def _reopen(svc: ResidentCapacityControl, request: SandboxSessionOpen) -> None:
    """Re-enter the admit-once entry point the dispatcher calls."""

    async def drive() -> None:
        svc.bind_loop(asyncio.get_running_loop())
        svc.open_sandbox_session(request)
        await asyncio.sleep(0.05)

    asyncio.run(drive())


def _holds_credit(stores: ResidentStores) -> bool:
    return any(claim.holds_credit for claim in stores.claims.all())


def test_a_session_admits_once_against_a_reserved_host() -> None:
    live = {_HOST_WORKER}
    svc, stores = _build(live)
    request = _admitted(svc)

    claim = svc._admission.active_claim(_INVOCATION)
    assert claim is not None and claim.state is ClaimState.ACCEPTED
    assert svc.sandbox_session_worker(_INVOCATION) == _HOST_WORKER

    # A second open over the same live admission raises no second claim.
    _reopen(svc, request)
    assert len(stores.claims.by_invocation(_INVOCATION)) == 1


def test_a_restart_over_a_surviving_host_keeps_the_session_placeable() -> None:
    live = {_HOST_WORKER}
    svc, stores = _build(live)
    _admitted(svc)

    svc.rehydrate(stores.to_snapshot())

    claim = svc._admission.active_claim(_INVOCATION)
    assert claim is not None
    # A co-located session lost no route, so its admission still places it.
    assert claim.state is ClaimState.ACCEPTED
    assert svc.sandbox_session_worker(_INVOCATION) == _HOST_WORKER


def test_a_restart_whose_host_is_gone_settles_the_claim_and_frees_the_credit() -> None:
    live = {_HOST_WORKER}
    svc, stores = _build(live)
    _admitted(svc)
    live.clear()  # the reserved worker did not survive the restart

    svc.rehydrate(stores.to_snapshot())

    assert svc._admission.active_claim(_INVOCATION) is None
    assert svc.sandbox_session_worker(_INVOCATION) is None
    assert not _holds_credit(stores)


def test_a_host_lost_mid_session_releases_the_admission_for_a_successor() -> None:
    live = {_HOST_WORKER}
    svc, stores = _build(live)
    request = _admitted(svc)
    assert svc.sandbox_session_worker(_INVOCATION) == _HOST_WORKER

    admitted = svc._admission.active_claim(_INVOCATION)
    assert admitted is not None

    live.clear()  # the reserved worker is gone before the session finished
    assert svc.sandbox_session_worker(_INVOCATION) is None

    # Re-opening reconciles the admission its host can no longer honor rather than
    # short-circuiting on it, so the session is not wedged holding a dead claim.
    _reopen(svc, request)

    assert admitted.state is ClaimState.TERMINAL
    assert not _holds_credit(stores)
    # The stale reservation is invalidated too, so the family is not stuck at its quota
    # against a worker no session can be placed on.
    assert all(
        replica.state is ReplicaState.PREEMPTED
        for replica in stores.directory.by_family(_FAMILY)
        if replica.worker_id == _HOST_WORKER
    )
