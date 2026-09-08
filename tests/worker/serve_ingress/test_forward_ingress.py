"""The worker-hosted forward ingress admits through control and relays opaquely.

It authenticates nothing itself: it freezes the client's transparent envelope, hands
the presented credential and the request's descriptor to control, and serves the request
only on control's admission. The engine's own response is relayed back to the client
unchanged.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from shared.resident.contracts import AdmissionHandoff
from shared.resident.envelope import ServeRequestEnvelope
from worker.serve_ingress.channel import ServeIngressChannel
from worker.serve_ingress.listener import ServeForwardIngress, ServeIngressRequest
from worker.serve_ingress.rendezvous import ServeIngressAdmission, ServeIngressDenied


def _handoff() -> AdmissionHandoff:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="serve/tsk-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )


class _Control:
    """Stands in for control: records the request and answers with a fixed decision."""

    def __init__(self, decision: object = None) -> None:
        self.requests: list[ServeIngressRequest] = []
        self.begun: list[tuple[ServeRequestEnvelope, ServeIngressChannel]] = []
        self.decision = decision
        self.ingress: ServeForwardIngress | None = None

    def propose(self, request: ServeIngressRequest) -> None:
        self.requests.append(request)
        decision = self.decision
        if decision is None:
            decision = ServeIngressAdmission(session_id="rly-1", handoff=_handoff())
        assert self.ingress is not None
        # Control answers on its own thread, as it would over the attachment.
        threading.Thread(
            target=self.ingress.rendezvous.deliver,
            args=(request.request_id, decision),
            daemon=True,
        ).start()

    def begin(self, admission, envelope, channel) -> None:
        self.begun.append((envelope, channel))


@contextmanager
def _running(control: _Control) -> Iterator[str]:
    counter = iter(f"req-{i}" for i in range(1, 1000))
    ingress = ServeForwardIngress(
        bind_host="127.0.0.1",
        port=0,
        public_url="http://ingress.example",
        propose=control.propose,
        begin=control.begin,
        new_request_id=lambda: next(counter),
        admission_timeout_sec=5.0,
        stream_idle_timeout_sec=5.0,
    )
    control.ingress = ingress
    port = ingress.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        ingress.stop()


def _serve_body(control: _Control, status: int = 200) -> None:
    """Deliver an engine response onto the channel the ingress opened."""
    for _ in range(500):
        if control.begun:
            break
        threading.Event().wait(0.01)
    _envelope, channel = control.begun[-1]
    channel.head(status, (("content-type", "application/json"),))
    channel.chunk(b'{"ok":')
    channel.chunk(b"true}")
    channel.complete()


def test_the_client_request_is_frozen_and_admitted_through_control() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control,), daemon=True).start()
        response = httpx.post(
            f"{base}/api/v1/serve/tasks/tsk-1/v1/chat/completions?stream=1",
            content=b'{"model":"m"}',
            headers={"authorization": "Bearer client-token", "x-trace": "t1"},
            timeout=10.0,
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    request = control.requests[0]
    assert request.serve_task_id == "tsk-1"
    assert request.method == "POST"
    assert request.path == "/v1/chat/completions"
    assert request.query == "stream=1"
    # The credential reaches control on the control message, and only there.
    assert request.credential == "Bearer client-token"
    # Control is given the descriptor, never the body: the raw request stays worker
    # private behind its digest.
    assert request.body_bytes == len(b'{"model":"m"}')
    assert not hasattr(request, "body")

    envelope, _channel = control.begun[0]
    assert envelope.body == b'{"model":"m"}'
    assert envelope.digest() == request.descriptor_digest
    # The client credential is stripped from what will reach the engine.
    assert "authorization" not in {name.lower() for name, _ in envelope.headers}
    assert ("x-trace", "t1") in [(n.lower(), v) for n, v in envelope.headers]


def test_a_denied_request_never_starts_a_drive() -> None:
    control = _Control(decision=ServeIngressDenied(403, "denied"))
    with _running(control) as base:
        response = httpx.get(f"{base}/api/v1/serve/tasks/tsk-1/v1/models", timeout=10.0)
    assert response.status_code == 403
    assert control.begun == []


def test_an_unframeable_request_is_refused_before_control_is_asked() -> None:
    control = _Control()
    with _running(control) as base:
        response = httpx.get(
            f"{base}/api/v1/serve/tasks/tsk-1/v1/models",
            headers={"upgrade": "websocket"},
            timeout=10.0,
        )
    assert response.status_code == 400
    assert control.requests == []


@pytest.mark.parametrize("path", ["/api/v1/serve/tasks/tsk-1", "/other"])
def test_a_request_outside_the_task_route_is_not_served(path: str) -> None:
    control = _Control()
    with _running(control) as base:
        response = httpx.get(f"{base}{path}", timeout=10.0)
    assert response.status_code == 404
    assert control.requests == []


def test_the_engine_status_and_headers_reach_the_client() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control, 201), daemon=True).start()
        response = httpx.get(f"{base}/api/v1/serve/tasks/tsk-1/v1/models", timeout=10.0)
    assert response.status_code == 201
    assert response.headers["content-type"] == "application/json"


def test_a_head_request_carries_no_body() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control,), daemon=True).start()
        response = httpx.head(
            f"{base}/api/v1/serve/tasks/tsk-1/v1/models", timeout=10.0
        )
    assert response.status_code == 200
    assert response.content == b""
