"""A gated serve request in flight at a root restart fails and frees its slot.

The request's origin is the root's serve edge, so the restart takes its client and its
data path with it, and nothing durable can re-drive it. Startup records a failed
terminal for it, reaps the replica's engine request through the serve task's worker,
and releases the credit, while a workflow's claim waits for its own ledger terminal.
"""

import asyncio
from typing import Any, cast

import pytest

from server.resident import ClaimState, ReplicaState, ResidentSnapshot
from server.resident.state import ClaimTerminalReason
from server.serve import (
    ServeBindingStore,
    ServeStatusTerminal,
    ServeTerminalStatus,
    ServeTerminalStore,
)
from server.serve.forward_exposure import ForwardIngressDirectory
from server.serve.ingress import ServeIngressRegistry
from server.serve.service import GatedServe
from server.startup import rehydrate_root_state
from shared.resident.reports import ResidentBootstrapOutcome
from tests.server.resident.test_serve_origination import (
    _FAMILY,
    _SERVE_TASK,
    _ack,
    _adopt,
    _build,
    _origination,
    _ServeDelivery,
)
from tests.server.resident.test_service import _build as _build_workflow_control
from tests.server.resident.test_workflow_terminal_reconcile import _admit, _held
from tests.server.task.test_v2_orchestration import FakeRegistry, _runtime

_LOST = "serve request lost in a root restart"


class _Runtime:
    """The restored runtime: the serve task still runs on its worker."""

    async def rehydrate(self) -> None:
        return None

    def live_resident_task_ids(self) -> frozenset[str]:
        return frozenset({_SERVE_TASK})

    def release_ended_reservations(self) -> None:
        return None

    def resident_invocation_completed(
        self, workflow_id: str, invocation_id: str
    ) -> bool | None:
        return None


class _SnapshotRegistry:
    def __init__(self, snapshot: ResidentSnapshot) -> None:
        self._snapshot = snapshot

    async def load_snapshot_async(self) -> ResidentSnapshot:
        return self._snapshot


def _in_flight(*, drain: bool = False, uncertain: bool = False) -> ResidentSnapshot:
    """A root holding both of a standing replica's slots for gated serve requests: one
    accepted and streaming, one reserved."""
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    asyncio.run(svc._originate_serve(_origination(_ServeDelivery(), "inv-1")))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-1"))
    )
    asyncio.run(svc._originate_serve(_origination(_ServeDelivery(), "inv-2")))
    if uncertain:
        # A claim an earlier start left uncertain, before startup settled serve claims.
        (claim,) = stores.claims.by_invocation("inv-1")
        svc._admission.on_route_loss(claim)
    if drain:
        svc.drain_serve_replica(_SERVE_TASK)
    (replica,) = stores.directory.by_family(_FAMILY)
    assert stores.credit_ledger.held(replica.replica_id) == 2
    return stores.to_snapshot()


def _edge(control: Any, terminals: ServeTerminalStore) -> GatedServe:
    return GatedServe(
        bindings=ServeBindingStore(),
        terminals=terminals,
        control=control,
        relay=cast(Any, None),
        ingresses=ServeIngressRegistry("serve-edge"),
        exposures=ForwardIngressDirectory("serve.example", 34000, 34009),
    )


def _start(runtime: Any, control: Any, snapshot: ResidentSnapshot, edge: GatedServe):
    async def start() -> None:
        try:
            await rehydrate_root_state(
                runtime, control, cast(Any, _SnapshotRegistry(snapshot)), edge
            )
        finally:
            control.shutdown()

    asyncio.run(start())


def _restart(
    snapshot: ResidentSnapshot, terminals: ServeTerminalStore | None = None
) -> tuple[Any, Any, Any, ServeTerminalStore]:
    svc, stores, _settled, deps = _build()
    terminals = terminals if terminals is not None else ServeTerminalStore()
    _start(_Runtime(), svc, snapshot, _edge(svc, terminals))
    return svc, stores, deps, terminals


@pytest.mark.parametrize("uncertain", [False, True])
def test_a_restart_fails_each_serve_request_it_ended_and_frees_its_slot(
    uncertain: bool,
) -> None:
    svc, stores, deps, terminals = _restart(_in_flight(uncertain=uncertain))

    for invocation_id in ("inv-1", "inv-2"):
        (claim,) = stores.claims.by_invocation(invocation_id)
        assert claim.state is ClaimState.TERMINAL
        assert claim.terminal_reason is ClaimTerminalReason.FAILED
        terminal = terminals.get(invocation_id)
        assert terminal is not None
        assert (terminal.status, terminal.detail) == (ServeTerminalStatus.FAILED, _LOST)
        # The replica's engine request is reaped once, through the serve task's worker.
        assert (
            deps.relays.count(
                (
                    "wkr-replica",
                    "resident_sidecar_reap",
                    {"invocation_id": invocation_id},
                )
            )
            == 1
        )
    (replica,) = stores.directory.by_family(_FAMILY)
    assert stores.credit_ledger.held(replica.replica_id) == 0

    # The freed slots admit a new request.
    delivery = _ServeDelivery()

    async def admit() -> None:
        svc.bind_loop(asyncio.get_running_loop())
        try:
            await asyncio.wait_for(
                svc._originate_serve(_origination(delivery, "inv-3")), 5.0
            )
        finally:
            svc.shutdown()

    asyncio.run(admit())
    assert len(delivery.opened) == 1
    assert stores.claims.by_invocation("inv-3")[0].holds_credit


def test_a_restart_stops_a_drained_serve_replica_its_lost_requests_held() -> None:
    _svc, stores, _deps, _terminals = _restart(_in_flight(drain=True))

    (replica,) = stores.directory.by_family(_FAMILY)
    assert replica.state is ReplicaState.STOPPED


def test_a_crash_between_the_failed_terminal_and_its_release_heals_on_next_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _in_flight()
    terminals = ServeTerminalStore()

    def crash(self: Any, invocation_id: str, reason: ClaimTerminalReason) -> None:
        raise RuntimeError("root crashed")

    with monkeypatch.context() as patched:
        patched.setattr(
            "server.resident.service.ResidentCapacityControl.reconcile_serve_terminal",
            crash,
        )
        with pytest.raises(RuntimeError):
            _restart(snapshot, terminals)
    # The failed terminals were recorded before the crash; the claims, never released,
    # are still the snapshot's.
    assert {t.invocation_id for t in terminals.all()} == {"inv-1", "inv-2"}

    _svc, stores, _deps, _terminals = _restart(snapshot, terminals)

    (replica,) = stores.directory.by_family(_FAMILY)
    assert stores.credit_ledger.held(replica.replica_id) == 0


def test_a_restart_leaves_a_workflow_claim_to_its_ledger() -> None:
    registry = FakeRegistry()
    _runtime_before, workflow_id, env = _held(registry)
    snapshot = _admit(workflow_id, env.invocation_id)
    svc, stores, _settled, _delivery = _build_workflow_control()
    terminals = ServeTerminalStore()

    _start(_runtime(registry), svc, snapshot, _edge(svc, terminals))

    (claim,) = stores.claims.by_invocation(env.invocation_id)
    assert claim.state is ClaimState.UNCERTAIN and claim.replica_id is not None
    assert stores.credit_ledger.held(claim.replica_id) == 1
    assert terminals.all() == []


def test_a_restart_reaps_nothing_for_requests_that_already_settled() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    terminals = ServeTerminalStore()
    for invocation_id in ("inv-1", "inv-2"):
        asyncio.run(svc._originate_serve(_origination(_ServeDelivery(), invocation_id)))
        asyncio.run(
            svc._on_ack(
                _ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id=invocation_id)
            )
        )
        terminals.record(
            ServeStatusTerminal(
                invocation_id=invocation_id, status=ServeTerminalStatus.COMPLETED
            )
        )
        svc.reconcile_serve_terminal(invocation_id, ClaimTerminalReason.COMPLETED)

    _svc, _stores, deps, _terminals = _restart(stores.to_snapshot(), terminals)

    assert [r for r in deps.relays if r[1] == "resident_sidecar_reap"] == []
