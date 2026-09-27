"""Which credential a managed-model egress sends, observed at a capture endpoint."""

import json
import logging
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from shared.tools.contract import MediatedOperationPermit, ToolOperationEnvelope
from shared.tools.model.schema import MODEL_INTERFACE, ModelRequest
from shared.utils.ids import new_mediated_permit_id
from worker.egress.backends import ModelEgress

_DEPLOYMENT_KEY = "sk-deployment"


class _CaptureServer(ThreadingHTTPServer):
    authorizations: list[str | None]


class _CaptureHandler(BaseHTTPRequestHandler):
    server: _CaptureServer

    def do_POST(self) -> None:
        self.server.authorizations.append(self.headers.get("Authorization"))
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def capture() -> Iterator[_CaptureServer]:
    server = _CaptureServer(("127.0.0.1", 0), _CaptureHandler)
    server.authorizations = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _request(server: _CaptureServer) -> ModelRequest:
    host, port = server.server_address[:2]
    return ModelRequest(
        interface=MODEL_INTERFACE,
        url=f"http://{host!s}:{port}/v1",
        body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )


def _envelope() -> ToolOperationEnvelope:
    return ToolOperationEnvelope(
        interface=MODEL_INTERFACE,
        idempotency_key="idm-1",
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=1000,
    )


def _permit(**overrides: Any) -> MediatedOperationPermit:
    fields: dict[str, Any] = {
        "permit_id": new_mediated_permit_id(),
        "agent_task_id": "tsk-agent",
        "call_correlation": "call-1",
        "interface": MODEL_INTERFACE,
        "subject": MODEL_INTERFACE,
        "invocation_id": "inv-1",
        "idempotency_key": "idm-1",
        "request_digest": "d",
        "target_id": "w-1",
        "target_generation": 1,
        "deadline_epoch": 2_000_000_000.0,
        "max_results": 1,
        "timeout_sec": 10.0,
        "result_char_cap": 1000,
    }
    fields.update(overrides)
    return MediatedOperationPermit(**fields)


def _execute(
    egress: ModelEgress, request: ModelRequest, permit: MediatedOperationPermit
) -> None:
    egress.execute(_envelope(), request, permit)


def _complete(
    egress: ModelEgress, request: ModelRequest, permit: MediatedOperationPermit
) -> None:
    egress.complete(_envelope(), request, permit)


_Egress = Callable[[ModelEgress, ModelRequest, MediatedOperationPermit], None]
_PATHS = pytest.mark.parametrize("egress_call", [_execute, _complete])


@_PATHS
def test_a_permit_without_a_grant_sends_no_deployment_key(
    capture: _CaptureServer, egress_call: _Egress
) -> None:
    egress = ModelEgress(_DEPLOYMENT_KEY, logging.getLogger("test"))
    egress_call(egress, _request(capture), _permit())
    assert capture.authorizations == [None]


@_PATHS
def test_a_granted_permit_sends_the_deployment_key(
    capture: _CaptureServer, egress_call: _Egress
) -> None:
    egress = ModelEgress(_DEPLOYMENT_KEY, logging.getLogger("test"))
    egress_call(egress, _request(capture), _permit(deployment_credential=True))
    assert capture.authorizations == [f"Bearer {_DEPLOYMENT_KEY}"]


@_PATHS
def test_a_pinned_credential_is_sent_instead_of_the_deployment_key(
    capture: _CaptureServer, egress_call: _Egress
) -> None:
    egress = ModelEgress(_DEPLOYMENT_KEY, logging.getLogger("test"))
    egress_call(egress, _request(capture), _permit(credential="sk-workflow"))
    assert capture.authorizations == ["Bearer sk-workflow"]


def test_a_permit_cannot_carry_both_credential_sources() -> None:
    with pytest.raises(ValueError, match="credential"):
        _permit(credential="sk-workflow", deployment_credential=True)
