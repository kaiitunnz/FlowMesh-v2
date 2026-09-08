"""Resident-capacity control drives a task-addressed serve invocation edge-origin.

The gated edge is the transport-only origin: control admits the same claim against only
the serve task's own allocation family and opens the edge relay (never a caller-worker
handoff). The engine ack authorizes the stream through the edge; a fenced external
status terminal — with no manifest — releases the credit and finalizes the client, and
only that recorded terminal releases it. A give-up terminalizes through the fenced
FAILED path, so the credit never strands and no ``DS`` state is fabricated.
"""

import asyncio
from typing import Any

from server.network.state import (
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteOrigin,
)
from server.resident import (
    AdmissionController,
    ClaimState,
    LifecycleScaleManager,
    ReplicaEndpoint,
    ReplicaState,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentStores,
)
from server.resident.service import ResidentWorkerDelivery, ServeOrigination
from server.resident.state import (
    AdmissionProfile,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
)
from server.task.v2.representations.operators import ServiceDependency
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamStatus,
)

_SERVE_TASK = "tsk-serve"
_FAMILY = "serve/tsk-serve"


def _held(stores: ResidentStores, replica_id: str | None) -> int:
    assert replica_id is not None
    return stores.credit_ledger.held(replica_id)


class _FakeNetwork:
    async def resolve(
        self, origin_node_id: str, listener: ReplicaListenerAdvertisement
    ) -> tuple[RouteOrigin, ResolvedRoute]:
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="ep-root",
            reachability_class=ReachabilityClass.ROUTABLE,
            trust_domain="td",
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=listener.node_id,
            listener_generation=listener.listener_generation,
            route_epoch=1,
            candidates=(),
        )
        return origin, route


class _FakeSessions:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []

    async def update(self, session_id: str, **fields: str | int) -> None:
        self.records.setdefault(session_id, {}).update(
            {k: str(v) for k, v in fields.items()}
        )

    async def delete(self, session_id: str) -> None:
        self.deleted.append(session_id)


class _ServeDelivery:
    """Records every seam call control routes a serve invocation's outcome through."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, AdmissionHandoff]] = []
        self.authorized: list[tuple[str, RouteAuthorization]] = []
        self.closed: list[str] = []
        self.chunks: list[str] = []
        self.terminals: list[tuple[ClaimTerminalReason, str | None]] = []
        self.completed = False
        self.failed: str | None = None
        self.redrives = 0

    def open(self, session_id: str, handoff: AdmissionHandoff) -> None:
        self.opened.append((session_id, handoff))

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        self.authorized.append((session_id, auth))

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    def tee(self, payload: str) -> None:
        self.chunks.append(payload)

    def record_terminal(self, reason: ClaimTerminalReason, detail: str | None) -> None:
        self.terminals.append((reason, detail))

    def complete(self) -> None:
        self.completed = True

    def fail(self, detail: str) -> None:
        self.failed = detail

    def redrive(self) -> None:
        self.redrives += 1


class _Deps:
    def __init__(self) -> None:
        self.relays: list[tuple[str, str, dict[str, Any]]] = []
        self.sessions = _FakeSessions()

    def build(self) -> ResidentWorkerDelivery:
        return ResidentWorkerDelivery(
            relay=self._relay,
            origin_worker_of_task=lambda task_id: None,
            serve_worker_of=lambda replica: (
                "wkr-replica" if replica.serve_task_id is not None else None
            ),
            node_of_worker=lambda worker_id: "node-1" if worker_id else None,
            network=_FakeNetwork(),
            sessions=self.sessions,
            root_node_id=lambda: "node-root",
            edge_id="serve-edge",
        )

    def _relay(self, worker_id: str, frame_kind: str, payload: dict[str, Any]) -> bool:
        self.relays.append((worker_id, frame_kind, payload))
        return True

    def kinds(self) -> list[str]:
        return [k for _w, k, _p in self.relays]


def _build() -> tuple[ResidentCapacityControl, ResidentStores, list[Any], _Deps]:
    stores = ResidentStores()
    limits = ResidentPolicyLimits()
    settled: list[Any] = []

    def settle_cb(*args: Any, **kwargs: Any) -> bool:
        settled.append((args, kwargs))
        return True

    def redispatch_cb(task_id: str, call_correlation: str) -> bool:
        settled.append(("redispatch", task_id, call_correlation))
        return True

    async def materialize_fn(family: Any, replica: Any) -> str:
        raise AssertionError("a standing serve replica is adopted, never materialized")

    admission = AdmissionController(stores)
    lifecycle = LifecycleScaleManager(
        stores, limits=limits, admission_slots=2, materialize_fn=materialize_fn
    )
    deps = _Deps()
    svc = ResidentCapacityControl(
        stores=stores,
        admission=admission,
        lifecycle=lifecycle,
        limits=limits,
        dependency_resolver=lambda task_id: None,
        settle_cb=settle_cb,
        redispatch_cb=redispatch_cb,
        endpoint_probe=lambda serve_task_id: ReplicaEndpoint(
            base_url="http://replica", model="m"
        ),
        delivery=deps.build(),
        poll_interval_sec=0.01,
        redrive_backoff_sec=0.0,
    )
    return svc, stores, settled, deps


def _adopt(svc: ResidentCapacityControl) -> None:
    svc.adopt_serve_replica(
        serve_task_id=_SERVE_TASK,
        family=_FAMILY,
        dependency=ServiceDependency(service_ref="m"),
        endpoint=ReplicaEndpoint(base_url="http://replica", model="m"),
        binding_generation=0,
    )


def _origination(delivery: _ServeDelivery, invocation_id: str = "inv-1"):
    return ServeOrigination(
        invocation_id=invocation_id,
        idempotency_key="idm-1",
        task_id=invocation_id,
        call_correlation=f"serve/{invocation_id}",
        subject=InvocationSubject(
            kind=InvocationSubjectKind.EXTERNAL, id="p1", tenant="acme"
        ),
        family=_FAMILY,
        dependency=ServiceDependency(service_ref="m"),
        profile=AdmissionProfile(
            engine_batch_key="m|chat",
            serve_task_id=_SERVE_TASK,
            binding_generation=0,
            descriptor_digest="sha-req",
        ),
        request_payload='{"messages": []}',
        delivery=delivery,
    )


def _ack(
    svc, outcome, *, invocation_id="inv-1", rejection=None
) -> ResidentBootstrapAck:
    return ResidentBootstrapAck(
        task_id=invocation_id,
        call_correlation=f"serve/{invocation_id}",
        invocation_id=invocation_id,
        session_id=svc._attempts[invocation_id].session_id,
        outcome=outcome,
        rejection=rejection,
    )


def _outcome(
    svc, status, *, invocation_id="inv-1", manifest=None, error=None
) -> ResidentOpOutcome:
    return ResidentOpOutcome(
        task_id=invocation_id,
        call_correlation=f"serve/{invocation_id}",
        invocation_id=invocation_id,
        session_id=svc._attempts[invocation_id].session_id,
        status=status,
        manifest=manifest,
        error=error,
    )


def test_adopt_registers_a_standing_non_teardown_replica() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    replica = stores.directory.by_family(_FAMILY)[0]
    assert replica.standing is True
    assert replica.state is ReplicaState.WARM
    assert replica.serve_task_id == _SERVE_TASK
    # An idle sweep never tears down a standing serve replica.
    svc._lifecycle._idle_retain_sec = 0.001
    svc._lifecycle.sweep_idle(now_ts=10_000_000.0)
    assert stores.directory.by_family(_FAMILY)[0].state is ReplicaState.WARM


def test_originate_admits_against_the_family_and_opens_the_edge_relay() -> None:
    svc, stores, _settled, deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    # The edge is the origin: the sidecar is bound and the edge relay opens, but no
    # caller-worker resident_handoff is ever relayed.
    assert deps.kinds() == ["resident_sidecar_bind"]
    assert len(delivery.opened) == 1
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.RESERVED
    assert claim.family == _FAMILY
    assert _held(stores, claim.replica_id) == 1
    # The handoff carries the serve fence, and the session routes from the edge stream.
    _sid, handoff = delivery.opened[0]
    assert handoff.serve_task_id == _SERVE_TASK
    assert handoff.binding_generation == 0
    record = deps.sessions.records[svc._attempts["inv-1"].session_id]
    assert record["origin_node"] == "serve-edge"
    assert record["origin_worker"] == ""


def test_ack_authorizes_through_the_edge_then_terminal_releases_credit() -> None:
    svc, stores, settled, deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))

    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.STREAMING
    assert len(delivery.authorized) == 1  # authorized via the edge, not a worker relay
    assert "resident_authorization" not in deps.kinds()

    # A SUCCESS outcome with NO manifest still finalizes: the live relay is the serve
    # data mode. The external terminal is recorded before the credit releases.
    asyncio.run(svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS)))
    assert delivery.terminals == [(ClaimTerminalReason.COMPLETED, None)]
    assert delivery.completed is True
    assert claim.state is ClaimState.TERMINAL
    assert _held(stores, claim.replica_id) == 0
    # No DS settle for an external subject.
    assert settled == []


def test_definite_failure_finalizes_the_client_and_releases_credit() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    asyncio.run(
        svc._on_outcome(
            _outcome(svc, ResidentStreamStatus.DEFINITE_FAILURE, error="engine refused")
        )
    )
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.TERMINAL
    assert delivery.terminals[-1][0] is ClaimTerminalReason.FAILED
    assert delivery.failed is not None and "engine refused" in delivery.failed


def test_a_reject_on_a_standing_replica_fails_only_the_request() -> None:
    # BLOCKER guard: a definite per-request rejection on the user's standing serve
    # replica fails ONLY that request via its fenced terminal; it never preempts or
    # reaps the shared endpoint (which cannot re-materialize). Without the guard,
    # _release_definite(preempt=True) invalidates the incarnation and cancels the user's
    # serve task, tearing the endpoint down for every other client.
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    replica = stores.directory.by_family(_FAMILY)[0]
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.REJECTED, rejection="stale"))
    )
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.TERMINAL
    assert delivery.terminals[-1][0] is ClaimTerminalReason.FAILED
    assert delivery.failed is not None
    assert _held(stores, replica.replica_id) == 0  # this request's credit released
    survivor = stores.directory.get(replica.replica_id)
    assert survivor is not None
    assert survivor.state is ReplicaState.WARM  # not preempted
    assert survivor.incarnation == 1  # not invalidated


def test_redrive_exhaustion_on_a_standing_replica_fails_only_the_request() -> None:
    # BLOCKER guard: an uncertain per-request loss that exhausts its re-drives on a
    # standing replica fails ONLY that request via a fenced terminal: _hold_locked
    # never
    # preempts the shared endpoint (its max-redrive preempt would reap the user's task).
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    claim = stores.claims.by_invocation("inv-1")[0]
    replica = stores.directory.by_family(_FAMILY)[0]
    attempt = svc._attempts["inv-1"]
    # One loss short of the threshold; the next uncertain loss exhausts the re-drives.
    svc._transient_failures["inv-1"] = svc._max_transient_redrives - 1
    asyncio.run(svc._hold_and_redrive_claim(attempt, claim, "stream lost"))
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.TERMINAL
    assert delivery.terminals[-1][0] is ClaimTerminalReason.FAILED
    assert _held(stores, replica.replica_id) == 0
    survivor = stores.directory.get(replica.replica_id)
    assert survivor is not None
    assert survivor.state is ReplicaState.WARM  # never preempted
    assert survivor.incarnation == 1


def test_stream_chunk_tees_only_to_a_matching_session() -> None:
    svc, _stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    session_id = svc._attempts["inv-1"].session_id

    svc._tee_chunk(
        ResidentStreamChunk(invocation_id="inv-1", session_id=session_id, payload="hi")
    )
    svc._tee_chunk(
        ResidentStreamChunk(
            invocation_id="inv-1", session_id="rly-stale", payload="dropped"
        )
    )
    assert delivery.chunks == ["hi"]


def test_a_client_close_alone_never_releases_credit() -> None:
    # Only a recorded fenced terminal releases the credit; nothing here settles it.
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.holds_credit
    assert _held(stores, claim.replica_id) == 1  # still held; no terminal


def test_give_up_terminalizes_through_the_fenced_failed_path() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.holds_credit

    # A re-drive that finds no deputy/route gives up: fail_serve releases the credit
    # through a fenced FAILED external terminal rather than stranding it.
    svc.fail_serve("inv-1", delivery, "no route to re-drive")
    assert delivery.terminals[-1][0] is ClaimTerminalReason.FAILED
    assert delivery.failed is not None
    assert claim.state is ClaimState.TERMINAL
    assert _held(stores, claim.replica_id) == 0


def test_reconcile_serve_terminal_settles_a_rehydrated_uncertain_claim() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    # Simulate a restart leaving the accepted claim UNCERTAIN with credit held.
    claim = stores.claims.by_invocation("inv-1")[0]
    svc._admission.on_route_loss(claim)
    assert claim.state is ClaimState.UNCERTAIN

    svc.reconcile_serve_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL
    assert _held(stores, claim.replica_id) == 0


def test_drain_serve_replica_denies_new_claims() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    replica = stores.directory.by_family(_FAMILY)[0]
    assert replica.state is ReplicaState.DRAINING
