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

import pytest

from server.resident.state import ClaimTerminalReason, InvocationSubjectKind
from server.routers.v1.serve import _ServeStreamTruncated, _stream
from server.serve import (
    ForwardIngressDirectory,
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
        self.relayed: list[tuple[str, str, dict]] = []
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

    def node_of_worker(self, worker_id: str) -> str | None:
        return f"node-{worker_id}" if worker_id else None

    def relay_to_worker(self, worker_id: str, kind: str, payload: dict) -> bool:
        self.relayed.append((worker_id, kind, payload))
        return True

    def schedule(self, coro) -> None:
        asyncio.get_event_loop().run_until_complete(coro)


class _FakeRelay:
    """A no-op edge relay; the origin transport is exercised in the resident tests."""

    def open(self, *args, **kwargs) -> None:
        pass

    def authorize(self, *args, **kwargs) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass


class _FakeForwardListener:
    """Records the port binds and releases the edge asks the root listener for."""

    def __init__(self) -> None:
        self.bound: list[tuple[str, int, int]] = []
        self.released: list[int] = []

    def schedule_bind(
        self, serve_task_id: str, exposure_generation: int, port: int
    ) -> None:
        self.bound.append((serve_task_id, exposure_generation, port))

    def schedule_release(self, port: int) -> None:
        self.released.append(port)


def _edge(
    control: _FakeControl,
    ingresses: ServeIngressRegistry | None = None,
    forward_listener: object | None = None,
    exposures: ForwardIngressDirectory | None = None,
) -> GatedServe:
    bindings = ServeBindingStore()
    return GatedServe(
        bindings=bindings,
        terminals=ServeTerminalStore(),
        control=control,  # type: ignore[arg-type]
        relay=_FakeRelay(),  # type: ignore[arg-type]
        ingresses=ingresses or ServeIngressRegistry("serve-edge"),
        exposures=exposures or ForwardIngressDirectory("serve.example", 34000, 34009),
        forward_listener=forward_listener,  # type: ignore[arg-type]
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


def test_forward_adoption_reserves_a_port_and_commits_on_bound_evidence() -> None:
    control = _FakeControl(
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m")
    )
    listener = _FakeForwardListener()
    edge = _edge(control, forward_listener=listener)
    edge.adopt("tsk-1", ServeAccessMode.FORWARD)

    # Adoption reserves a port on the root authority and asks the listener to bind it.
    assert len(listener.bound) == 1 and listener.bound[0][0] == "tsk-1"
    exposure = edge.exposures.current("tsk-1")
    assert exposure is not None and 34000 <= exposure.public_port <= 34009
    # The exposure is not live until the root listener's bound evidence commits it.
    assert edge.exposures.live("tsk-1") is None

    edge.commit_forward("tsk-1", exposure.exposure_generation, 1)
    live = edge.exposures.live("tsk-1")
    assert live is not None and live.listener_generation == 1


def test_forward_adoption_without_a_configured_authority_fails_closed() -> None:
    # With no root forward authority/range configured, adoption reserves no port and
    # publishes no exposure, so the task fails closed rather than serving on proxy.
    control = _FakeControl(
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m")
    )
    listener = _FakeForwardListener()
    edge = _edge(
        control,
        forward_listener=listener,
        exposures=ForwardIngressDirectory("", 0, 0),
    )
    edge.adopt("tsk-1", ServeAccessMode.FORWARD)
    assert listener.bound == []
    assert edge.exposures.current("tsk-1") is None


def test_drain_retires_the_exposure_and_releases_the_root_listener() -> None:
    control = _FakeControl(
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m")
    )
    listener = _FakeForwardListener()
    edge = _edge(control, forward_listener=listener)
    edge.adopt("tsk-1", ServeAccessMode.FORWARD)
    exposure = edge.exposures.current("tsk-1")
    assert exposure is not None
    edge.commit_forward("tsk-1", exposure.exposure_generation, 1)

    edge.drain("tsk-1")
    assert listener.released == [exposure.public_port]
    assert edge.exposures.current("tsk-1") is None


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


def test_a_non_draining_client_that_overflows_is_failed_not_reported_complete() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.PROXY)
        stream = control.originations[0].delivery
        stream.head(200, ())
        for i in range(
            stream.queue.maxsize * 3
        ):  # flood past the bound, never draining
            stream.tee(f"f{i}".encode())
        # Memory stays bounded: frames past the backlog bound are dropped, not buffered.
        assert stream.queue.qsize() <= stream.queue.maxsize
        stream.complete()
        events = await _events(result)
        # A response that dropped frames is never reported complete: the drop poisons
        # the stream so its terminal lands as an error the consumer aborts on, never a
        # clean done over the gaps.
        assert events[-1].kind == "error"
        assert not any(e.kind == "done" for e in events)

    asyncio.run(run())


def test_the_router_aborts_the_response_on_a_poisoned_terminal() -> None:
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
        delivery.head(200, (("content-type", "text/event-stream"),))
        delivery.tee(b"data: 1\n\n")
        delivery.fail("resident stream lost")  # a mid-stream error terminal
        response = await _stream(result)
        assert response.status_code == 200
        # The head was already committed, so the router cannot change the status; it
        # aborts the body instead of ending it cleanly over the lost frames.
        received: list[bytes] = []
        with pytest.raises(_ServeStreamTruncated):
            async for chunk in response.body_iterator:
                received.append(
                    chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                )
        assert b"".join(received) == b"data: 1\n\n"

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
    # servable through the other, or an operator who chose forward to keep serve traffic
    # off the root still carries it there whenever a client uses the root URL.
    control = _FakeControl()
    edge = _edge(control)
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


def _forward_live_edge() -> tuple[_FakeControl, GatedServe]:
    """A gated edge with one live forward exposure for ``tsk-1``.

    The root binds the exposure's port and reports it bound, committing it live, so a
    forward request resolves and admits over the same shared root relay as proxy.
    """
    control = _FakeControl(
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m")
    )
    edge = _edge(control, forward_listener=_FakeForwardListener())
    edge.adopt("tsk-1", ServeAccessMode.FORWARD)
    exposure = edge.exposures.current("tsk-1")
    assert exposure is not None
    edge.commit_forward("tsk-1", exposure.exposure_generation, 1)
    return control, edge


def test_a_forward_request_admits_and_delivers_over_the_shared_root_relay() -> None:
    # A forward request over a live exposure originates the same local-delivery stream
    # as a proxy request; the engine's frames tee to the client here, not to a worker.
    control, edge = _forward_live_edge()

    async def run() -> None:
        result = edge.submit(
            "p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD
        )
        delivery = control.originations[0].delivery
        delivery.head(200, (("content-type", "application/json"),))
        delivery.tee(b"body")
        delivery.complete()
        events = await _events(result)
        assert events[0].kind == "head"
        assert [e.payload for e in events if e.kind == "chunk"] == [b"body"]
        assert events[-1].kind == "done"

    asyncio.run(run())


def test_a_forward_request_fails_closed_without_a_live_exposure() -> None:
    # A forward binding with no live root exposure is unavailable: submit fails closed
    # before any credit rather than serving over the proxy.
    control = _FakeControl()
    edge = _edge(control, forward_listener=_FakeForwardListener())
    _bind(edge, access_mode=ServeAccessMode.FORWARD)
    try:
        edge.submit("p1", "acme", "tsk-1", _envelope(), ServeAccessMode.FORWARD)
        raise AssertionError("expected IngressUnavailable")
    except IngressUnavailable:
        pass
    assert control.originations == []
