"""Resident-capacity control drives a task-addressed serve invocation edge-origin.

The gated edge is the transport-only origin: control admits the same claim against only
the serve task's own allocation family and opens the edge relay (never a caller-worker
handoff). The engine ack authorizes the stream through the edge; a fenced external
status terminal — with no manifest — releases the credit and finalizes the client, and
only that recorded terminal releases it. A give-up terminalizes through the fenced
FAILED path, so the credit never strands and no ``DS`` state is fabricated.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from server.network.reachability import NetworkReachabilityView
from server.network.resolver import resolve_route
from server.network.state import (
    NetworkEndpointAdvertisement,
    PolicyClass,
    ReachabilityClass,
    ReplicaListenerAdvertisement,
    ResolvedRoute,
    RouteCandidate,
    RouteOrigin,
    Transport,
    TrustedPeerPolicy,
)
from server.resident import (
    AdmissionController,
    ClaimState,
    LifecycleScaleManager,
    ReplicaEndpoint,
    ReplicaState,
    ResidentCapacityControl,
    ResidentPolicyLimits,
    ResidentSnapshot,
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
from server.telemetry.tracing import format_traceparent, serve_trace_id_int
from shared.resident.carriage import CONTROL_RELAY, ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import freeze_request_envelope
from shared.resident.reports import (
    ResidentBootstrapAck,
    ResidentBootstrapOutcome,
    ResidentOpOutcome,
    ResidentStreamChunk,
    ResidentStreamHead,
    ResidentStreamStatus,
)
from shared.schemas.network import PEER_PROTOCOL
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.ids import SpanIdKind, derived_span_id
from tests.server.telemetry_helpers import recording_control_tracer

_SERVE_TASK = "tsk-serve"
_FAMILY = "serve/tsk-serve"


def _held(stores: ResidentStores, replica_id: str | None) -> int:
    assert replica_id is not None
    return stores.credit_ledger.held(replica_id)


class _FakeNetwork:
    def __init__(self, base_candidate: bool = True) -> None:
        self._base_candidate = base_candidate
        self.policy_classes: list[PolicyClass] = []

    async def resolve(
        self,
        origin_node_id: str,
        listener: ReplicaListenerAdvertisement,
        *,
        trust: TrustedPeerPolicy | None = None,
        policy_class: PolicyClass = PolicyClass.DEFAULT,
    ) -> tuple[RouteOrigin, ResolvedRoute]:
        self.policy_classes.append(policy_class)
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="ep-root",
            reachability_class=ReachabilityClass.ROUTABLE,
            trust_domain="td",
        )
        candidates = (
            (RouteCandidate(transport=Transport.CONTROL_RELAY, hops=()),)
            if self._base_candidate
            else ()
        )
        route = ResolvedRoute(
            origin_id="rog-1",
            target_node_id=listener.node_id,
            listener_generation=listener.listener_generation,
            route_epoch=1,
            candidates=candidates,
        )
        return origin, route

    def record_observations(self, origin, listener, observations) -> None:
        self.observations = list(observations)

    async def endpoint_for(self, node_id: str):
        return None


class _TrustedPeerNetwork(_FakeNetwork):
    """A deployment whose policy trusts the root node's pair with a directly routable
    replica, graded by the real resolver."""

    _TRUSTED = TrustedPeerPolicy(
        enabled=True,
        trust_domain="td",
        classes=frozenset(ReachabilityClass),
        protocol=PEER_PROTOCOL,
    )

    async def resolve(
        self,
        origin_node_id: str,
        listener: ReplicaListenerAdvertisement,
        *,
        trust: TrustedPeerPolicy | None = None,
        policy_class: PolicyClass = PolicyClass.DEFAULT,
    ) -> tuple[RouteOrigin, ResolvedRoute]:
        self.policy_classes.append(policy_class)
        origin = RouteOrigin(
            origin_id="rog-1",
            endpoint_id="ep-root",
            node_id=origin_node_id,
            reachability_class=ReachabilityClass.ROUTABLE,
            trust_domain="td",
            protocols=(PEER_PROTOCOL,),
            relay_attachment_id="att-root",
        )
        target = listener.model_copy(
            update={
                "routes": ("10.0.0.2:9500",),
                "protocols": (PEER_PROTOCOL,),
                "directly_routable": True,
            }
        )
        endpoint = NetworkEndpointAdvertisement(
            endpoint_id="ep-1",
            node_id=listener.node_id,
            url="10.0.0.2:9101",
            peer_url="10.0.0.2:9102",
            generation=1,
            trust_domain="td",
            reachability_class=ReachabilityClass.ROUTABLE,
            protocols=(PEER_PROTOCOL,),
            relay_attachment_id="att-1",
        )
        route = resolve_route(
            origin,
            target,
            endpoint,
            NetworkReachabilityView(),
            trust=self._TRUSTED if trust is None else trust,
            now=0.0,
            route_epoch=1,
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

    def __init__(self, dials_peers: bool = False) -> None:
        self.dials_peers = dials_peers
        self.opened: list[tuple[str, AdmissionHandoff]] = []
        self.plans: list[ResidentCarriagePlan] = []
        self.traceparents: list[str | None] = []
        self.authorized: list[tuple[str, RouteAuthorization]] = []
        self.closed: list[str] = []
        self.heads: list[tuple[int, tuple[tuple[str, str], ...]]] = []
        self.chunks: list[bytes] = []
        self.terminals: list[tuple[ClaimTerminalReason, str | None]] = []
        self.completed = False
        self.failed: str | None = None
        self.redrives = 0

    def open(
        self,
        session_id: str,
        handoff: AdmissionHandoff,
        plan: ResidentCarriagePlan,
        traceparent: str | None = None,
    ) -> None:
        self.opened.append((session_id, handoff))
        self.plans.append(plan)
        self.traceparents.append(traceparent)

    def authorize(self, session_id: str, auth: RouteAuthorization) -> None:
        self.authorized.append((session_id, auth))

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    def head(self, status: int, headers: tuple[tuple[str, str], ...]) -> None:
        self.heads.append((status, headers))

    def tee(self, payload: bytes) -> None:
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
    def __init__(
        self, base_candidate: bool = True, trusted_peers: bool = False
    ) -> None:
        self.relays: list[tuple[str, str, dict[str, Any]]] = []
        self.sessions = _FakeSessions()
        self.network = (
            _TrustedPeerNetwork() if trusted_peers else _FakeNetwork(base_candidate)
        )

    def build(self) -> ResidentWorkerDelivery:
        return ResidentWorkerDelivery(
            relay=self._relay,
            origin_worker_of_task=lambda task_id: None,
            serve_worker_of=lambda replica: (
                "wkr-replica" if replica.serve_task_id is not None else None
            ),
            node_of_worker=lambda worker_id: "node-1" if worker_id else None,
            network=self.network,
            sessions=self.sessions,
            root_node_id=lambda: "node-root",
            edge_id="serve-edge",
        )

    def _relay(self, worker_id: str, frame_kind: str, payload: dict[str, Any]) -> bool:
        self.relays.append((worker_id, frame_kind, payload))
        return True

    def kinds(self) -> list[str]:
        return [k for _w, k, _p in self.relays]


def _build(
    base_candidate: bool = True,
    stop_fn: Callable[[str], None] | None = None,
    trusted_peers: bool = False,
    control: Any = None,
) -> tuple[ResidentCapacityControl, ResidentStores, list[Any], _Deps]:
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
        stores,
        limits=limits,
        admission_slots=2,
        materialize_fn=materialize_fn,
        stop_fn=stop_fn,
    )
    deps = _Deps(base_candidate=base_candidate, trusted_peers=trusted_peers)
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
        control=control,
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
        envelope=freeze_request_envelope(
            method="POST",
            upstream_path="v1/chat/completions",
            query="",
            headers=[("content-type", "application/json")],
            body=b'{"messages": []}',
        ),
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
    # The carriage plan control selected rides beside the handoff and names the base
    # transport; the session record keeps the selection as diagnostics.
    assert delivery.plans[0].selected_transport == CONTROL_RELAY
    assert record["selected_transport"] == CONTROL_RELAY


def test_a_root_that_dials_no_peer_rides_the_relay_beside_a_trusted_pair() -> None:
    # The deployment trusts the pair and the replica is directly routable, but the root
    # dials no peer, so its call takes control_relay rather than a plan its carriage
    # refuses.
    svc, stores, _settled, _deps = _build(trusted_peers=True)
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert len(delivery.opened) == 1
    assert delivery.plans[0].selected_transport == CONTROL_RELAY
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.RESERVED


def test_a_root_that_dials_peers_takes_the_trusted_pairs_peer_transport() -> None:
    svc, stores, _settled, deps = _build(trusted_peers=True)
    _adopt(svc)
    delivery = _ServeDelivery(dials_peers=True)
    asyncio.run(svc._originate_serve(_origination(delivery)))

    plan = delivery.plans[0]
    assert plan.selected_transport == Transport.WORKER_DIRECT.value
    assert plan.selected_endpoint == "10.0.0.2:9500"
    # The root's serve ingress is its own route origin, apart from the root node's
    # workers.
    assert deps.network.policy_classes == [PolicyClass.SERVE_INGRESS]
    assert stores.claims.by_invocation("inv-1")[0].state is ClaimState.RESERVED


def test_the_serve_origin_opens_under_the_serve_requests_invocation_span() -> None:
    control, _exporter = recording_control_tracer(TelemetryLevel.COARSE)
    svc, _stores, _settled, _deps = _build(control=control)
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    (traceparent,) = delivery.traceparents
    assert traceparent == format_traceparent(
        serve_trace_id_int("inv-1", ""), derived_span_id(SpanIdKind.INVOCATION, "inv-1")
    )


def test_the_serve_origin_opens_with_no_traceparent_when_telemetry_is_off() -> None:
    svc, _stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert delivery.traceparents == [None]


def test_no_control_relay_candidate_holds_the_credit_without_opening() -> None:
    # control_relay is the base every attempt can fall back to; a resolved route
    # without it holds the credit rather than opening on no transport.
    svc, stores, _settled, deps = _build(base_candidate=False)
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert delivery.opened == []
    # The hold moves the claim UNCERTAIN, which still holds the credit — the base
    # transport being unavailable never releases it.
    claim = stores.claims.by_invocation("inv-1")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert _held(stores, claim.replica_id) == 1


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
        ResidentStreamChunk(invocation_id="inv-1", session_id=session_id, payload=b"hi")
    )
    svc._tee_chunk(
        ResidentStreamChunk(
            invocation_id="inv-1", session_id="rly-stale", payload=b"dropped"
        )
    )
    assert delivery.chunks == [b"hi"]


def test_stream_head_routes_only_to_a_matching_session() -> None:
    svc, _stores, _settled, _deps = _build()
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    session_id = svc._attempts["inv-1"].session_id

    svc._head(
        ResidentStreamHead(
            invocation_id="inv-1",
            session_id=session_id,
            status=200,
            headers=(("content-type", "text/event-stream"),),
        )
    )
    svc._head(
        ResidentStreamHead(
            invocation_id="inv-1",
            session_id="rly-stale",
            status=500,
            headers=(("content-type", "application/json"),),
        )
    )
    assert delivery.heads == [(200, (("content-type", "text/event-stream"),))]


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


def test_drain_stops_a_drained_standing_replica_with_no_credit() -> None:
    reaped: list[str] = []
    svc, stores, _settled, _deps = _build(stop_fn=reaped.append)
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    replica = stores.directory.by_family(_FAMILY)[0]
    # With no admitted work, the stopped serve task's replica leaves the live directory
    # rather than lingering DRAINING forever (the idle sweep skips standing replicas).
    assert replica.state is ReplicaState.STOPPED
    assert stores.directory.live_by_family(_FAMILY) == []
    # The serve task owns its standing replica: a drain on its requeue leaves the task
    # to re-run, so stopping the replica never cancels it.
    assert reaped == []


def test_drain_keeps_an_in_flight_standing_replica_draining_until_it_settles() -> None:
    reaped: list[str] = []
    svc, stores, _settled, _deps = _build(stop_fn=reaped.append)
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    assert _held(stores, stores.claims.by_invocation("inv-1")[0].replica_id) == 1

    svc.drain_serve_replica(_SERVE_TASK)
    replica = stores.directory.by_family(_FAMILY)[0]
    # Admitted work still holds credit, so the replica drains rather than stopping,
    # letting the accepted claim reconcile on its own fenced terminal.
    assert replica.state is ReplicaState.DRAINING

    # When the in-flight claim settles, its last credit release stops the drained
    # standing replica so it does not linger DRAINING in the directory (the idle sweep
    # is off by default and never reaps it).
    asyncio.run(svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS)))
    settled_replica = stores.directory.get(replica.replica_id)
    assert settled_replica is not None and settled_replica.state is ReplicaState.STOPPED
    assert stores.directory.live_by_family(_FAMILY) == []
    assert reaped == []


def _recording_materializations(svc: ResidentCapacityControl) -> list[str]:
    materialized: list[str] = []

    async def record(family: Any, replica: Any) -> str:
        materialized.append(family.family)
        return "tsk-cold"

    svc._lifecycle._materialize_fn = record
    return materialized


def test_a_request_queued_on_a_stopped_serve_task_starts_no_replica() -> None:
    async def run() -> None:
        svc, stores, _settled, _deps = _build()
        svc.bind_loop(asyncio.get_running_loop())
        materialized = _recording_materializations(svc)
        _adopt(svc)
        deliveries = [_ServeDelivery() for _ in range(3)]
        # Two requests fill the standing replica's admission slots.
        for i in range(2):
            await svc._originate_serve(_origination(deliveries[i], f"inv-{i}"))
        queued = asyncio.create_task(
            svc._originate_serve(_origination(deliveries[2], "inv-2"))
        )
        await asyncio.sleep(0.05)
        assert not queued.done()

        svc.drain_serve_replica(_SERVE_TASK)
        (standing,) = stores.directory.by_family(_FAMILY)
        assert standing.state is ReplicaState.DRAINING
        await asyncio.wait_for(queued, timeout=1.0)

        assert materialized == []
        assert stores.directory.by_family(_FAMILY) == [standing]
        assert deliveries[2].failed is not None
        assert "no live standing allocation" in deliveries[2].failed

    asyncio.run(run())


def test_a_request_on_a_stopped_serve_replica_starts_no_replica() -> None:
    svc, stores, _settled, _deps = _build()
    materialized = _recording_materializations(svc)
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.directory.by_family(_FAMILY)[0].state is ReplicaState.STOPPED

    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert materialized == []
    assert delivery.failed is not None
    assert "no live standing allocation" in delivery.failed


def test_a_serve_family_stays_standing_across_a_snapshot() -> None:
    svc, stores, _settled, _deps = _build()
    _adopt(svc)
    restored = ResidentStores()
    restored.load_snapshot(
        ResidentSnapshot.model_validate_json(stores.to_snapshot().model_dump_json())
    )

    family = restored.families.get(_FAMILY)
    assert family is not None and family.standing is True
