"""The worker-hosted forward ingress binds one port per task and relays opaquely.

Control reserves a public port for a forward serve task and this host binds a listener
on it, mapped to the task's exposure. A client reaches the task at that port with the
engine's own paths — no task-qualified prefix — so the listener resolves the serve task
from its own exposure, freezes the client's envelope, and serves the request only on
control's admission. The engine's own response is relayed back to the client unchanged.
"""

import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from shared.resident.carriage import ResidentCarriagePlan
from shared.resident.contracts import AdmissionHandoff
from shared.resident.envelope import ServeRequestEnvelope
from shared.resident.serve_ingress import ServeIngressBound, ServeIngressReserve
from worker.serve_ingress.channel import ServeIngressChannel
from worker.serve_ingress.listener import ForwardIngressHost, ServeIngressRequest
from worker.serve_ingress.rendezvous import ServeIngressAdmission, ServeIngressDenied

_TASK = "tsk-1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _handoff() -> AdmissionHandoff:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family=f"serve/{_TASK}",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    )


class _Control:
    """Stands in for control: records the request and answers with a fixed decision."""

    def __init__(self, decision: object = None) -> None:
        self.requests: list[ServeIngressRequest] = []
        self.begun: list[tuple[ServeRequestEnvelope, ServeIngressChannel]] = []
        self.bounds: list[ServeIngressBound] = []
        self.decision = decision
        self.host: ForwardIngressHost | None = None

    def propose(self, request: ServeIngressRequest) -> None:
        self.requests.append(request)
        decision = self.decision
        if decision is None:
            decision = ServeIngressAdmission(
                session_id="rly-1",
                task_id="inv-1",
                call_correlation="serve/inv-1",
                handoff=_handoff(),
                carriage_plan=ResidentCarriagePlan(session_id="rly-1"),
            )
        assert self.host is not None
        threading.Thread(
            target=self.host.rendezvous.deliver,
            args=(request.request_id, decision),
            daemon=True,
        ).start()

    def begin(self, admission, envelope, channel) -> None:
        self.begun.append((envelope, channel))


@contextmanager
def _running(control: _Control) -> Iterator[str]:
    counter = iter(f"req-{i}" for i in range(1, 1000))
    host = ForwardIngressHost(
        bind_host="127.0.0.1",
        authority="127.0.0.1",
        propose=control.propose,
        begin=control.begin,
        new_request_id=lambda: next(counter),
        report_bound=control.bounds.append,
        admission_timeout_sec=5.0,
        stream_idle_timeout_sec=5.0,
    )
    control.host = host
    port = _free_port()
    host.reserve(
        ServeIngressReserve(
            serve_task_id=_TASK,
            binding_generation=0,
            exposure_generation=0,
            public_port=port,
        )
    )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        host.stop()


def _serve_body(control: _Control, status: int = 200) -> None:
    for _ in range(500):
        if control.begun:
            break
        threading.Event().wait(0.01)
    _envelope, channel = control.begun[-1]
    channel.head(status, (("content-type", "application/json"),))
    channel.chunk(b'{"ok":')
    channel.chunk(b"true}")
    channel.complete()


def test_reserve_binds_a_port_and_reports_it_bound() -> None:
    control = _Control()
    with _running(control):
        assert len(control.bounds) == 1
        bound = control.bounds[0]
        assert bound.serve_task_id == _TASK
        assert bound.exposure_generation == 0
        assert bound.listener_generation >= 1


def test_the_client_request_is_frozen_and_admitted_through_control() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control,), daemon=True).start()
        response = httpx.post(
            f"{base}/v1/chat/completions?stream=1",
            content=b'{"model":"m"}',
            headers={"authorization": "Bearer client-token", "x-trace": "t1"},
            timeout=10.0,
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    request = control.requests[0]
    # The serve task is the listener's own exposure, not a client-supplied path.
    assert request.serve_task_id == _TASK
    assert request.binding_generation == 0
    assert request.exposure_generation == 0
    assert request.method == "POST"
    # The engine-native path is forwarded verbatim, with no task-qualified prefix.
    assert request.path == "/v1/chat/completions"
    assert request.query == "stream=1"
    assert request.credential == "Bearer client-token"
    assert request.body_bytes == len(b'{"model":"m"}')
    assert not hasattr(request, "body")

    envelope, _channel = control.begun[0]
    assert envelope.body == b'{"model":"m"}'
    assert envelope.digest() == request.descriptor_digest
    assert "authorization" not in {name.lower() for name, _ in envelope.headers}
    assert ("x-trace", "t1") in [(n.lower(), v) for n, v in envelope.headers]


def test_a_denied_request_never_starts_a_drive() -> None:
    control = _Control(decision=ServeIngressDenied(403, "denied"))
    with _running(control) as base:
        response = httpx.get(f"{base}/v1/models", timeout=10.0)
    assert response.status_code == 403
    assert control.begun == []


def test_an_unframeable_request_is_refused_before_control_is_asked() -> None:
    control = _Control()
    with _running(control) as base:
        response = httpx.get(
            f"{base}/v1/models",
            headers={"upgrade": "websocket"},
            timeout=10.0,
        )
    assert response.status_code == 400
    assert control.requests == []


def test_the_engine_status_and_headers_reach_the_client() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control, 201), daemon=True).start()
        response = httpx.get(f"{base}/v1/models", timeout=10.0)
    assert response.status_code == 201
    assert response.headers["content-type"] == "application/json"


def test_a_head_request_carries_no_body() -> None:
    control = _Control()
    with _running(control) as base:
        threading.Thread(target=_serve_body, args=(control,), daemon=True).start()
        response = httpx.head(f"{base}/v1/models", timeout=10.0)
    assert response.status_code == 200
    assert response.content == b""


def _serve_status_only(control: _Control, status: int) -> None:
    """Deliver a head-then-terminal response, as a no-body status produces."""
    for _ in range(500):
        if control.begun:
            break
        threading.Event().wait(0.01)
    _envelope, channel = control.begun[-1]
    channel.head(status, (("x-probe", "1"),))
    channel.complete()


@pytest.mark.parametrize("status", [204, 304])
def test_a_no_body_status_is_not_chunk_framed(status: int) -> None:
    # A status defined to carry no body must not be chunk-framed, or a real client
    # hangs.
    control = _Control()
    with _running(control) as base:
        threading.Thread(
            target=_serve_status_only, args=(control, status), daemon=True
        ).start()
        response = httpx.get(f"{base}/v1/models", timeout=10.0)
    assert response.status_code == status
    assert "transfer-encoding" not in {k.lower() for k in response.headers}
    assert response.content == b""
