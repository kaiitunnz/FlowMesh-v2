"""The worker-hosted forward serve ingress's control-side admission and transport.

A forward request arrives on a worker's public ingress, which relays it to control for
admission over the worker's attachment. Control authenticates it, resolves the live
FORWARD binding, admits it against that binding's own allocation group, and relays the
fence back down to the ingress worker — which tees the response to its own client. These
cover that control side: the admission threads the ingress worker as the request's
origin, a fail-closed leg refuses a request whose ingress is unregistered, and the
transport relays the two-phase fence down and reaps the ingress rendezvous on teardown.
"""

import asyncio
from collections.abc import Callable

from server.serve import (
    GatedServe,
    ServeBindingStore,
    ServeForwardTransport,
    ServeTerminalStore,
)
from server.serve.ingress import ServeAccessMode, ServeIngressRegistry
from server.task.v2.representations.operators import ServiceInterface
from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff, RouteAuthorization
from shared.resident.envelope import ServeRequestEnvelope
from shared.resident.serve_ingress import ServeIngressRequest

_RelayLog = list[tuple[str, str, dict]]


def _recorder(log: "_RelayLog") -> Callable[[str, str, dict], bool]:
    def relay(worker_id: str, frame_kind: str, payload: dict) -> bool:
        log.append((worker_id, frame_kind, payload))
        return True

    return relay


def _handoff(invocation_id: str = "inv-1") -> AdmissionHandoff:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id=invocation_id,
        family="serve/tsk-1",
        replica_id="rpl-1",
        incarnation=1,
        serve_task_id="tsk-1",
        binding_generation=0,
        descriptor_digest="d1",
    )


def _auth(invocation_id: str = "inv-1") -> RouteAuthorization:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id=invocation_id,
        replica_id="rpl-1",
        incarnation=1,
    )


def _request(
    request_id: str = "srq-1",
    method: str = "POST",
    digest: str = "d1",
) -> ServeIngressRequest:
    return ServeIngressRequest(
        request_id=request_id,
        serve_task_id="tsk-1",
        credential=None,
        method=method,
        path="/v1/chat/completions",
        query="",
        descriptor_digest=digest,
        body_bytes=8,
    )


class _ForwardControl:
    """A control double that records originations and the frames relayed to workers."""

    def __init__(self) -> None:
        self.originations: list = []
        self.relayed: _RelayLog = []

    def schedule(self, coro) -> None:
        asyncio.run(coro)

    def originate_serve(self, origination) -> None:
        self.originations.append(origination)

    def relay_to_worker(self, worker_id: str, frame_kind: str, payload: dict) -> bool:
        self.relayed.append((worker_id, frame_kind, payload))
        return True

    def node_of_worker(self, worker_id: str | None) -> str | None:
        return f"node-{worker_id}" if worker_id else None

    # Unused by the forward path but part of the control surface the edge may touch.
    def call_on_loop(self, fn: Callable[[], None]) -> None:
        fn()


def _edge(
    control: _ForwardControl,
    transport: ServeForwardTransport,
    *,
    register: bool,
) -> GatedServe:
    registry = ServeIngressRegistry("serve-edge")
    if register:
        registry.register_forward(
            origin_id="node-wrk-a",
            worker_id="wrk-a",
            public_url="http://ingress.example:8100",
            generation=1,
        )
    edge = GatedServe(
        bindings=ServeBindingStore(),
        terminals=ServeTerminalStore(),
        control=control,  # type: ignore[arg-type]
        relay=None,  # type: ignore[arg-type]
        ingresses=registry,
        forward_transport=transport,
    )
    edge._bindings.adopt(
        "tsk-1",
        service_ref="org/model",
        interface=ServiceInterface.CHAT,
        isolation=None,
        adapter=None,
        adapter_source=None,
        engine_batch_key="org/model|chat",
        max_output_tokens=None,
        access_mode=ServeAccessMode.FORWARD,
    )
    return edge


def _kinds(log: _RelayLog) -> list[tuple[str, str]]:
    return [(worker, kind) for worker, kind, _payload in log]


def test_admit_forward_threads_the_ingress_worker_as_the_request_origin() -> None:
    control = _ForwardControl()
    transport = ServeForwardTransport(control.relay_to_worker)
    edge = _edge(control, transport, register=True)

    edge.admit_forward(_request(), "wrk-a")

    assert len(control.originations) == 1
    orig = control.originations[0]
    # The forward serve resolves its route from the ingress worker's own node, so it
    # carries that worker as its origin; a proxy origination leaves it unset.
    assert orig.origin_worker == "wrk-a"
    assert orig.request_id == "srq-1"
    # The server holds only the worker-computed descriptor digest, never the body.
    assert orig.profile.descriptor_digest == "d1"

    # When admission mints the handoff, the transport relays the decision down to the
    # ingress worker, keyed by the request id its rendezvous is waiting on.
    orig.delivery.open(
        "rly-1", _handoff(orig.invocation_id), ResidentCarriagePlan(session_id="rly-1")
    )
    admitted = [p for w, k, p in control.relayed if k == "serve_ingress_admitted"]
    assert len(admitted) == 1
    assert admitted[0]["request_id"] == "srq-1"
    assert admitted[0]["session_id"] == "rly-1"


def test_admit_forward_fails_closed_when_no_ingress_is_registered() -> None:
    control = _ForwardControl()
    transport = ServeForwardTransport(control.relay_to_worker)
    edge = _edge(control, transport, register=False)

    edge.admit_forward(_request("srq-9"), "wrk-a")

    # Fail closed: no admission, and the ingress is refused promptly rather than at its
    # own admission timeout.
    assert control.originations == []
    denied = [p for w, k, p in control.relayed if k == "serve_ingress_denied"]
    assert len(denied) == 1
    assert denied[0]["request_id"] == "srq-9"
    assert denied[0]["status"] == 503


def test_admit_forward_refuses_a_method_the_binding_forbids() -> None:
    control = _ForwardControl()
    transport = ServeForwardTransport(control.relay_to_worker)
    edge = _edge(control, transport, register=True)
    # Narrow the binding to GET; a POST is refused before any credit.
    binding = edge._bindings.get("tsk-1")
    assert binding is not None
    edge._bindings._bindings["tsk-1"] = binding.model_copy(
        update={"allowed_methods": ("GET",)}
    )

    edge.admit_forward(_request("srq-3", method="POST"), "wrk-a")

    assert control.originations == []
    denied = [p for w, k, p in control.relayed if k == "serve_ingress_denied"]
    assert denied and denied[0]["status"] == 405


def test_forward_transport_relays_the_two_phase_and_reaps_the_rendezvous() -> None:
    log: _RelayLog = []
    transport = ServeForwardTransport(_recorder(log))
    transport.track(invocation_id="inv-1", worker_id="wrk-a", request_id="srq-1")
    transport.open(
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        task_id="inv-1",
        call_correlation="serve/inv-1",
        handoff=_handoff(),
        envelope=ServeRequestEnvelope(method="POST", path="/v1/chat/completions"),
        plan=ResidentCarriagePlan(session_id="rly-1"),
    )
    transport.authorize("rly-1", _auth())
    transport.close("rly-1")

    assert _kinds(log) == [
        ("wrk-a", "serve_ingress_admitted"),
        ("wrk-a", "serve_ingress_authorized"),
        ("wrk-a", "serve_ingress_reaped"),
    ]
    # The reap names the request and invocation so the ingress drops the rendezvous
    # entry and closes a live drive — no leaked entry or pinned client connection.
    reaped = log[-1][2]
    assert reaped["request_id"] == "srq-1"
    assert reaped["invocation_id"] == "inv-1"
    # The tracking is gone: a late authorize or reap for the same session is a no-op.
    transport.authorize("rly-1", _auth())
    transport.close("rly-1")
    assert len(log) == 3


def test_forward_transport_denies_a_request_that_settled_before_it_opened() -> None:
    log: _RelayLog = []
    transport = ServeForwardTransport(_recorder(log))
    transport.track(invocation_id="inv-2", worker_id="wrk-a", request_id="srq-2")

    # An admission that gave up before opening a session leaves the ingress rendezvous
    # waiting; forgetting the settled request denies it so the client is not pinned.
    transport.forget("inv-2")

    assert _kinds(log) == [("wrk-a", "serve_ingress_denied")]
    assert log[0][2]["request_id"] == "srq-2"
    # Forgetting again is a no-op; the entry is already gone.
    transport.forget("inv-2")
    assert len(log) == 1
