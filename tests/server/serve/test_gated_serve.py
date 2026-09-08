"""The gated serve edge resolves a live binding, admits, streams, and adopts/drains.

The edge authenticates and authorizes at the router; here it resolves only a live
binding, rejects a bad method/path before any credit, mints an external-principal
origination against the binding's own allocation family, and relays opaque frames to the
client. Adoption is idempotent and gated by the allowed-model policy; a stop drains.
"""

import asyncio
from collections.abc import Callable

from server.resident.state import ClaimTerminalReason, InvocationSubjectKind
from server.serve import (
    GatedServe,
    ServeBindingStore,
    ServeResult,
    ServeStatusTerminal,
    ServeTerminalStatus,
    ServeTerminalStore,
)
from server.serve.service import BindingNotFound, MethodNotAllowed, PathNotAllowed
from server.task.v2.representations.operators import ServiceInterface
from shared.resident.contracts import ReplicaEndpoint
from shared.resident.wire import resident_request_digest


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


def _edge(control: _FakeControl) -> GatedServe:
    bindings = ServeBindingStore()
    return GatedServe(
        bindings=bindings,
        terminals=ServeTerminalStore(),
        control=control,  # type: ignore[arg-type]
        relay=_FakeRelay(),  # type: ignore[arg-type]
    )


def _bind(edge: GatedServe, task_id: str = "tsk-1", model: str = "org/model") -> None:
    edge._bindings.adopt(
        task_id,
        service_ref=model,
        interface=ServiceInterface.CHAT,
        isolation=None,
        adapter=None,
        adapter_source=None,
        engine_batch_key=f"{model}|chat",
        max_output_tokens=None,
    )


async def _events(result: ServeResult) -> list:
    return [ev async for ev in result.events()]


def test_submit_without_a_live_binding_raises_before_any_credit() -> None:
    control = _FakeControl()
    edge = _edge(control)
    try:
        edge.submit("p1", "acme", "tsk-1", "POST", "v1/chat/completions", "{}")
        raise AssertionError("expected BindingNotFound")
    except BindingNotFound:
        pass
    assert control.originations == []


def test_submit_rejects_bad_method_and_path() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    try:
        edge.submit("p1", "acme", "tsk-1", "GET", "v1/chat/completions", "{}")
        raise AssertionError("expected MethodNotAllowed")
    except MethodNotAllowed:
        pass
    try:
        edge.submit("p1", "acme", "tsk-1", "POST", "v1/embeddings", "{}")
        raise AssertionError("expected PathNotAllowed")
    except PathNotAllowed:
        pass
    assert control.originations == []


def test_submit_originates_an_external_subject_against_the_binding_family() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)
    body = '{"messages": []}'
    edge.submit("p1", "acme", "tsk-1", "POST", "v1/chat/completions", body)
    assert len(control.originations) == 1
    orig = control.originations[0]
    assert orig.subject.kind is InvocationSubjectKind.EXTERNAL
    assert orig.subject.id == "p1" and orig.subject.tenant == "acme"
    assert orig.family == "serve/tsk-1"
    assert orig.profile.descriptor_digest == resident_request_digest(body)


def test_submit_streams_teed_frames_then_terminates() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    async def run() -> None:
        result = edge.submit("p1", "acme", "tsk-1", "POST", "v1/chat/completions", "{}")
        delivery = control.originations[0].delivery
        delivery.tee("he")
        delivery.tee("llo")
        delivery.complete()
        events = await _events(result)
        assert [e.payload for e in events if e.kind == "chunk"] == ["he", "llo"]
        assert events[-1].kind == "done"

    asyncio.run(run())


def test_preflush_loss_redrives_while_postflush_loss_fails_the_client() -> None:
    control = _FakeControl()
    edge = _edge(control)
    _bind(edge)

    # A loss before any flush re-drives transparently, not failing the client.
    edge.submit("p1", "acme", "tsk-1", "POST", "v1/chat/completions", "{}")
    control.originations[0].delivery.redrive()
    assert len(control.redrives) == 1

    async def postflush() -> None:
        result = edge.submit("p1", "acme", "tsk-1", "POST", "v1/chat/completions", "{}")
        delivery = control.originations[1].delivery
        delivery.tee("partial")
        delivery.redrive()  # flushed: must fail the client, not re-stream onto it
        events = await _events(result)
        assert (events[0].kind, events[0].payload) == ("chunk", "partial")
        assert events[-1].kind == "error"
        assert all(e.payload != "partial" for e in events[1:])

    asyncio.run(postflush())
    # The held credit still reconciles through a re-drive on either loss.
    assert len(control.redrives) == 2


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
