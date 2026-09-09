"""The gated serve route is a transparent HTTP surface over the task ID.

Every transparent method reaches the edge with the client's own path, query, headers,
and raw body; a request that cannot be framed unambiguously, or one past the body
bound, is refused before it is admitted and so before any engine work.
"""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routers.v1 import serve as serve_router
from server.serve import ServeAccessMode
from server.serve.service import (
    IngressUnavailable,
    ServeEvent,
    WrongIngress,
)
from shared.resident.envelope import TRANSPARENT_METHODS

PREFIX = "/api/v1"


class _Result:
    """A submitted request whose response is one head, one chunk, and a terminal."""

    def __init__(self) -> None:
        self.closed = False

    async def events(self):
        yield ServeEvent(
            kind="head", status=200, headers=(("content-type", "application/json"),)
        )
        yield ServeEvent(kind="chunk", payload=b"ok")
        yield ServeEvent(kind="done")

    def close_client(self) -> None:
        self.closed = True


class _Edge:
    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, principal_id, tenant, serve_task_id, envelope, arrived_on):
        self.submitted.append(
            (principal_id, tenant, serve_task_id, envelope, arrived_on)
        )
        return _Result()


def _client(edge: _Edge) -> TestClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.gated-serve-router")
    app.state.gated_serve = edge
    app.include_router(serve_router.router, prefix=PREFIX)
    return TestClient(app)


def _url(path: str = "v1/chat/completions") -> str:
    return f"{PREFIX}/serve/tasks/tsk-1/{path}"


def test_every_transparent_method_reaches_the_edge() -> None:
    edge = _Edge()
    client = _client(edge)
    for method in TRANSPARENT_METHODS:
        response = client.request(method, _url("v1/models"))
        assert response.status_code == 200, method
    assert [e.method for _p, _t, _task, e, _mode in edge.submitted] == list(
        TRANSPARENT_METHODS
    )


def test_the_client_path_query_and_body_are_frozen_verbatim() -> None:
    edge = _Edge()
    client = _client(edge)
    client.post(
        _url("v1/responses") + "?stream=1",
        content=b'{"model":"other"}',
        headers={"content-type": "application/json", "x-trace": "t1"},
    )
    envelope = edge.submitted[0][3]
    assert envelope.path == "/v1/responses"
    assert envelope.query == "stream=1"
    assert envelope.body == b'{"model":"other"}'
    assert ("x-trace", "t1") in [(k.lower(), v) for k, v in envelope.headers]


def test_an_empty_body_method_is_admitted() -> None:
    edge = _Edge()
    response = _client(edge).get(_url("v1/models"))
    assert response.status_code == 200
    assert edge.submitted[0][3].body == b""


def test_the_engine_response_headers_and_status_reach_the_client() -> None:
    response = _client(_Edge()).get(_url("v1/models"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content == b"ok"


def test_a_client_credential_is_never_forwarded() -> None:
    edge = _Edge()
    _client(edge).get(_url("v1/models"), headers={"authorization": "Bearer client"})
    names = {name.lower() for name, _ in edge.submitted[0][3].headers}
    assert "authorization" not in names


def test_an_unframeable_request_is_refused_before_admission() -> None:
    edge = _Edge()
    response = _client(edge).get(_url("v1/models"), headers={"upgrade": "websocket"})
    assert response.status_code == 400
    assert edge.submitted == []


def test_a_body_past_the_bound_is_refused_before_admission() -> None:
    edge = _Edge()
    oversized = b"x" * (serve_router._MAX_REQUEST_BYTES + 1)
    response = _client(edge).post(_url(), content=oversized)
    assert response.status_code == 413
    assert edge.submitted == []


def test_a_task_whose_ingress_is_unregistered_fails_closed_to_the_client() -> None:
    class _Closed(_Edge):
        def submit(self, principal_id, tenant, serve_task_id, envelope, arrived_on):
            raise IngressUnavailable("forward")

    response = _client(_Closed()).get(_url("v1/models"))
    assert response.status_code == 503


def test_the_root_router_presents_itself_as_the_proxy_ingress() -> None:
    # The arriving ingress is what lets the edge enforce a binding's pinned mode.
    edge = _Edge()
    _client(edge).get(_url("v1/models"))
    assert edge.submitted[0][4] is ServeAccessMode.PROXY


def test_a_task_pinned_to_another_ingress_is_not_found_here() -> None:
    class _Wrong(_Edge):
        def submit(self, principal_id, tenant, serve_task_id, envelope, arrived_on):
            raise WrongIngress("forward")

    response = _client(_Wrong()).get(_url("v1/models"))
    assert response.status_code == 404
