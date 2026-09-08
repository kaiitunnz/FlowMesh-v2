"""The gated serve edge resolves a live binding, admits, streams, and adopts/drains.

The edge authenticates and authorizes at the router; here it resolves only a live
binding, rejects a method the binding does not permit before any credit, mints an
external-principal origination against the binding's own allocation family, and relays
the engine's opaque frames to the client. Any path the client sends is forwarded rather
than matched against an allowlist. Adoption is idempotent and gated by the allowed-model
policy; a stop drains.
"""

import asyncio
from collections.abc import Callable

from server.resident.state import ClaimTerminalReason, InvocationSubjectKind
from server.routers.v1.serve import _stream
from server.serve import (
    GatedServe,
    ServeBindingStore,
    ServeResult,
    ServeStatusTerminal,
    ServeTerminalStatus,
    ServeTerminalStore,
)
from server.serve.ingress import ServeAccessMode, ServeIngressRegistry
from server.serve.service import (
    BindingNotFound,
    IngressUnavailable,
    MethodNotAllowed,
    WrongIngress,
)
from server.task.v2.representations.operators import ServiceInterface
from shared.resident.contracts import ReplicaEndpoint
from shared.resident.envelope import ServeRequestEnvelope, freeze_request_envelope


class _FakeControl:
    """Records the control calls the edge makes; runs loop work inline."""

    def __init__(
        self,
        *,
        endpoint: ReplicaEndpoint | None = None,
        model_allowed: bool = True,
    ) -> None:
        self.originations: list = []
        self.redrives: list = []
        self.failed_serve: list[tuple[str, str]] = []
        self.adopt_calls: list[dict] = []
        self.drained: list[str] = []
        self.reconciled: list[tuple[str, ClaimTerminalReason]] = []
        self._endpoint = endpoint
        self._model_allowed = model_allowed

    def call_on_loop(self, fn: Callable[[], None]) -> None:
        fn()

    def originate_serve(self, origination) -> None:
        self.originations.append(origination)

    def redrive_serve(self, origination) -> None:
        self.redrives.append(origination)

    def fail_serve(self, invocation_id: str, delivery, detail: str) -> None:
        self.failed_serve.append((invocation_id, detail))
        delivery.fail(detail)

    def serve_model_allowed(self, model_ref: str) -> bool:
        return self._model_allowed

    def probe_serve_endpoint(self, serve_task_id: str) -> ReplicaEndpoint | None:
        return self._endpoint

    def adopt_serve_replica(self, **kwargs) -> None:
        self.adopt_calls.append(kwargs)

    def drain_serve_replica(self, serve_task_id: str) -> None:
        self.drained.append(serve_task_id)

    def reconcile_serve_terminal(
        self, invocation_id: str, reason: ClaimTerminalReason
    ) -> None:
        self.reconciled.append((invocation_id, reason))


class _FakeRelay:
    """A no-op edge relay; the origin transport is exercised in the resident tests."""

    def open(self, *args, **kwargs) -> None:
        pass

    def authorize(self, *args, **kwargs) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass


def _edge(
    control: _FakeControl,
    ingresses: ServeIngressRegistry | None = None,
    forward_transport: object | None = None,
) -> GatedServe:
    bindings = ServeBindingStore()
    return GatedServe(
        bindings=bindings,
        terminals=ServeTerminalStore(),
        control=control,  # type: ignore[arg-type]
        relay=_FakeRelay(),  # type: ignore[arg-type]
        ingresses=ingresses or ServeIngressRegistry("serve-edge"),
        forward_transport=forward_transport,  # type: ignore[arg-type]
    )


def _bind(
    edge: GatedServe,
    task_id: str = "tsk-1",
    model: str = "org/model",
    access_mode: ServeAccessMode = ServeAccessMode.PROXY,
) -> None:
    edge._bindings.adopt(
        task_id,
        service_ref=model,
        interface=ServiceInterface.CHAT,
        isolation=None,
        adapter=None,
        adapter_source=None,
        engine_batch_key=f"{model}|chat",
        max_output_tokens=None,
        access_mode=access_mode,
    )


async def _events(result: ServeResult) -> list:
    return [ev async for ev in result.events()]


def _envelope(
    method: str = "POST",
    path: str = "v1/chat/completions",
    body: bytes = b"{}",
    query: str = "",
    headers: list[tuple[str, str]] | None = None,
) -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method=method,
        upstream_path=path,
        query=query,
        headers=[("content-type", "application/json")] if headers is None else headers,
        body=body,
    )


def test_submit_without_a_live_binding_raises_before_any_credit() -> None:
    control = _FakeControl()
    edge = _edge(control)
    try:
        edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
        raise AssertionError("expected BindingNotFound")
    except BindingNotFound:
        pass
    assert control.originations == []


def test_submit_rejects_a_method_the_binding_does_not_permit() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    store = edge._bindings
    store._bindings["tsk-1"] = store._bindings["tsk-1"].model_copy(
        update={"allowed_methods": ("POST",)}
    )
    try:
        edge.submit(
            "p1", "acme", "tsk-1", _envelope(method="GET"), ServeAccessMode.PROXY
        )
        raise AssertionError("expected MethodNotAllowed")
    except MethodNotAllowed:
        pass
    assert control.originations == []


def test_submit_forwards_any_engine_path_rather_than_an_allowlist() -> None:
    # The binding's interface selects the adopted family; it must not restrict which
    # engine endpoint the client may drive, or a transparent proxy is not transparent.
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    for method, path in (
        ("GET", "v1/models"),
        ("POST", "v1/responses"),
        ("POST", "v1/messages"),
        ("POST", "v1/embeddings"),
    ):
        edge.submit(
            "p1",
            "acme",
            "tsk-1",
            _envelope(method=method, path=path),
            ServeAccessMode.PROXY,
        )
    assert [o.envelope.method for o in control.originations] == [
        "GET",
        "POST",
        "POST",
        "POST",
    ]
    assert [o.envelope.path for o in control.originations] == [
        "/v1/models",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
    ]


def test_forward_fails_closed_without_a_registered_ingress() -> None:
    # A task pinned to forward must never be quietly served over the root-local proxy:
    # with no forward ingress registered the request is refused before any credit.
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge, access_mode=ServeAccessMode.FORWARD)
    try:
        edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
        raise AssertionError("expected IngressUnavailable")
    except IngressUnavailable:
        pass
    assert control.originations == []


def test_forward_is_admitted_once_its_ingress_is_registered() -> None:
    control = _FakeControl()
    registry = ServeIngressRegistry("serve-edge")
    edge = _edge(control, registry, forward_transport=_FakeRelay())
    _bind(edge, access_mode=ServeAccessMode.FORWARD)
    registry.register_forward("node-a", generation=1)
    edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
    assert len(control.originations) == 1


def test_a_proxy_binding_is_admitted_without_any_forward_ingress() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge, access_mode=ServeAccessMode.PROXY)
    edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
    assert len(control.originations) == 1


def test_submit_originates_an_external_subject_against_the_binding_family() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    envelope = _envelope(body=b'{"messages": []}')
    edge.submit("p1", "acme", "tsk-1", envelope, ServeAccessMode.PROXY)
    assert len(control.originations) == 1
    orig = control.originations[0]
    assert orig.subject.kind is InvocationSubjectKind.EXTERNAL
    assert orig.subject.id == "p1" and orig.subject.tenant == "acme"
    assert orig.family == "serve/tsk-1"
    assert orig.profile.descriptor_digest == envelope.digest()


def test_submit_streams_teed_frames_then_terminates() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
        delivery = control.originations[0].delivery
        delivery.tee(b"he")
        delivery.tee(b"llo")
        delivery.complete()
        events = await _events(result)
        assert [e.payload for e in events if e.kind == "chunk"] == [b"he", b"llo"]
        assert events[-1].kind == "done"

    asyncio.run(run())


def test_head_event_precedes_chunks_and_carries_status_and_headers() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit(
            "p1",
            "acme",
            "tsk-1",
            _envelope(body=b'{"stream": true}'),
            ServeAccessMode.PROXY,
        )
        delivery = control.originations[0].delivery
        delivery.head(
            200, (("content-type", "text/event-stream"), ("x-request-id", "r1"))
        )
        delivery.tee(b"data: {}\n\n")
        delivery.complete()
        events = await _events(result)
        assert events[0].kind == "head"
        assert events[0].status == 200
        assert events[0].headers == (
            ("content-type", "text/event-stream"),
            ("x-request-id", "r1"),
        )
        assert [e.payload for e in events if e.kind == "chunk"] == [b"data: {}\n\n"]

    asyncio.run(run())


def test_router_sets_the_client_status_and_headers_from_the_head() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit(
            "p1",
            "acme",
            "tsk-1",
            _envelope(body=b'{"stream": true}'),
            ServeAccessMode.PROXY,
        )
        delivery = control.originations[0].delivery
        delivery.head(
            201, (("content-type", "text/event-stream"), ("x-request-id", "r1"))
        )
        delivery.tee(b"data: {}\n\n")
        delivery.complete()
        response = await _stream(result)
        assert response.status_code == 201
        # Every non-hop-by-hop engine header reaches the client, not just content type.
        assert response.headers["content-type"] == "text/event-stream"
        assert response.headers["x-request-id"] == "r1"
        parts = [chunk async for chunk in response.body_iterator]
        body = b"".join(
            part.encode() if isinstance(part, str) else bytes(part) for part in parts
        )
        assert body == b"data: {}\n\n"

    asyncio.run(run())


def test_a_non_draining_client_cannot_pin_unbounded_memory() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
        stream = control.originations[0].delivery
        for i in range(
            stream.queue.maxsize * 3
        ):  # flood past the bound, never draining
            stream.tee(f"f{i}".encode())
        assert stream.queue.qsize() <= stream.queue.maxsize
        stream.complete()
        events = await _events(result)
        # The terminal still lands despite the overflow, so the client stream closes.
        assert events[-1].kind == "done"

    asyncio.run(run())


def test_client_disconnect_stops_teeing_and_records_no_terminal() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
    stream = control.originations[0].delivery
    stream.tee(b"early")
    result.close_client()
    size = stream.queue.qsize()
    stream.tee(b"after")
    # The disconnect stops teeing: a later frame is dropped, not buffered.
    assert stream.queue.qsize() == size
    # Closing the client records no fenced terminal, so the disconnect never releases
    # the credit — only the drive's own fenced terminal does.
    assert edge._terminals.all() == []


def test_preflush_loss_redrives_while_postflush_loss_fails_the_client() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    # A loss before any flush re-drives transparently, not failing the client.
    edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
    control.originations[0].delivery.redrive()
    assert len(control.redrives) == 1

    async def postflush() -> None:
        result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
        delivery = control.originations[1].delivery
        delivery.tee(b"partial")
        delivery.redrive()  # flushed: fail via the fenced terminal, do NOT re-run
        events = await _events(result)
        assert (events[0].kind, events[0].payload) == ("chunk", b"partial")
        assert events[-1].kind == "error"
        assert all(e.payload != b"partial" for e in events[1:])

    asyncio.run(postflush())
    # A flushed loss terminalizes FAILED through the fenced path and does NOT re-run the
    # engine (the output could not reach the client) — exactly one drive, no re-drive.
    assert len(control.redrives) == 1
    assert len(control.failed_serve) == 1


def test_a_disconnected_client_fails_rather_than_re_running_the_engine() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
    delivery = control.originations[0].delivery
    result.close_client()  # the client is gone
    delivery.redrive()  # a loss under a gone client fails, not re-runs the engine
    assert control.redrives == []
    assert len(control.failed_serve) == 1


def test_adopt_is_idempotent_and_model_gated() -> None:
    endpoint = ReplicaEndpoint(base_url="http://x/v1", model="org/model")
    control = _FakeControl(endpoint=endpoint)
    edge = _edge(control)

    edge.adopt("tsk-1")
    edge.adopt("tsk-1")  # a second endpoint report is a no-op
    assert len(control.adopt_calls) == 1
    assert edge._bindings.live("tsk-1") is not None
    assert control.adopt_calls[0]["family"] == "serve/tsk-1"

    denied = _FakeControl(endpoint=endpoint, model_allowed=False)
    denied_edge = _edge(denied)
    denied_edge.adopt("tsk-2")
    assert denied.adopt_calls == []
    assert denied_edge._bindings.live("tsk-2") is None


def test_drain_refuses_new_calls_and_drains_the_replica() -> None:
    endpoint = ReplicaEndpoint(base_url="http://x/v1", model="org/model")
    control = _FakeControl(endpoint=endpoint)
    edge = _edge(control)
    edge.adopt("tsk-1")
    edge.drain("tsk-1")
    assert control.drained == ["tsk-1"]
    assert edge._bindings.get("tsk-1") is None


def test_reconcile_replays_recorded_terminals() -> None:
    control = _FakeControl()
    edge = _edge(control)
    edge._terminals.record(
        ServeStatusTerminal(invocation_id="inv-x", status=ServeTerminalStatus.COMPLETED)
    )
    edge._terminals.record(
        ServeStatusTerminal(invocation_id="inv-y", status=ServeTerminalStatus.FAILED)
    )
    edge.reconcile_terminals()
    assert {inv for inv, _ in control.reconciled} == {"inv-x", "inv-y"}


def test_a_binding_is_refused_on_an_ingress_it_does_not_pin() -> None:
    # accessMode is policy, not a hint: a task pinned to one ingress must not be
    # servable through the other, or an operator who chose forward to keep serve
    # traffic off the root still carries it there whenever a client uses the root URL.
    control = _FakeControl()
    registry = ServeIngressRegistry("serve-edge")
    registry.register_forward("node-a", generation=1)

    edge = _edge(control, registry)
    _bind(edge, task_id="tsk-fwd", access_mode=ServeAccessMode.FORWARD)
    _bind(edge, task_id="tsk-pxy", access_mode=ServeAccessMode.PROXY)

    for task_id, arrived_on in (
        ("tsk-fwd", ServeAccessMode.PROXY),
        ("tsk-pxy", ServeAccessMode.FORWARD),
    ):
        try:
            edge.submit("p1", "acme", task_id, _envelope(), arrived_on)
            raise AssertionError(f"expected WrongIngress for {task_id}")
        except WrongIngress:
            pass
    assert control.originations == []


def _forward_edge() -> tuple[_FakeControl, GatedServe]:
    control = _FakeControl()
    registry = ServeIngressRegistry("serve-edge")
    edge = _edge(control, registry, forward_transport=_FakeRelay())
    _bind(edge, access_mode=ServeAccessMode.FORWARD)
    registry.register_forward("node-a", generation=1)
    return control, edge


def test_a_forward_loss_before_any_delivery_re_drives_transparently() -> None:
    control, edge = _forward_edge()
    edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
    delivery = control.originations[0].delivery
    delivery.redrive()
    # Nothing reached the client yet, so the request re-runs and the caller still sees
    # exactly one clean response.
    assert len(control.redrives) == 1
    assert control.failed_serve == []


def test_a_forward_loss_after_the_response_is_committed_never_re_drives() -> None:
    # The ingress delivers frames to its own client, so control never sees them. Without
    # the commit signal this loss would re-run the engine over a response the client
    # already holds — duplicate bytes under a status it cannot take back.
    control, edge = _forward_edge()
    edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
    invocation_id = control.originations[0].invocation_id
    delivery = control.originations[0].delivery

    edge.committed(invocation_id, 200, (("content-type", "application/json"),))
    delivery.redrive()

    assert control.redrives == []
    assert len(control.failed_serve) == 1


def test_a_committed_forward_response_enqueues_nothing_for_the_root() -> None:
    # A forward ingress already wrote its head to its own client; control must not also
    # queue it, or the root would hold frames no one drains.
    control, edge = _forward_edge()
    result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
    invocation_id = control.originations[0].invocation_id
    edge.committed(invocation_id, 200, (("content-type", "application/json"),))
    control.originations[0].delivery.tee(b"body")
    assert result._stream.queue.qsize() == 0
