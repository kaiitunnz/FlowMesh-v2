"""The finalization index's HTTP surface: bind once, resolve, and stay in scope."""

import logging
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.content import FinalizationIndex
from server.routers.v1 import content as content_router
from shared.content import reference_for
from shared.outcome import OutcomeManifest

PREFIX = "/api/v1"
_CONTENT = reference_for("local", b"result-body", media_type="application/json")


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set_value(self, key: str, value: str) -> None:
        self.values[key] = value

    def set_value_if_absent(self, key: str, value: str) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    def expire(self, key: str, ttl_sec: int) -> bool:
        return True


def _client() -> TestClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.content_router")
    app.state.finalization_index = FinalizationIndex(
        cast(Any, type("C", (), {"sync": _FakeRedis()})())
    )
    app.include_router(content_router.router, prefix=PREFIX)
    return TestClient(app)


def _bind(client: TestClient, idem: str, content: Any = None) -> Any:
    return client.put(
        f"{PREFIX}/content/finalizations",
        params={"idem": idem},
        json=(content or _CONTENT).model_dump(mode="json"),
    )


def test_a_finalization_binds_and_resolves(tmp_path) -> None:
    client = _client()
    bound = _bind(client, "idm-1")
    assert bound.status_code == 200
    manifest = OutcomeManifest.model_validate(bound.json())
    assert manifest.content == _CONTENT
    assert manifest.idempotency_key == "idm-1"

    resolved = client.get(f"{PREFIX}/content/finalizations", params={"idem": "idm-1"})
    assert resolved.status_code == 200
    assert OutcomeManifest.model_validate(resolved.json()) == manifest


def test_the_first_binding_stands(tmp_path) -> None:
    client = _client()
    first = _bind(client, "idm-2").json()
    other = reference_for("local", b"a different result", media_type="application/json")
    second = _bind(client, "idm-2", other).json()
    assert first == second


def test_an_unbound_key_is_404(tmp_path) -> None:
    client = _client()
    assert (
        client.get(
            f"{PREFIX}/content/finalizations", params={"idem": "idm-absent"}
        ).status_code
        == 404
    )


def test_content_outside_the_admitted_scope_is_refused(tmp_path) -> None:
    client = _client()
    elsewhere = _CONTENT.model_copy(update={"authorization_scope": "other"})
    # The test principal is the deployment-wide admin, so it may name a scope — but the
    # content it binds has to be in the scope the request acts in.
    assert _bind(client, "idm-3", elsewhere).status_code == 403


def test_no_payload_crosses_the_surface() -> None:
    # The content router binds finalizations and nothing else: bytes reach the shared
    # store directly, never through the server.
    paths = {getattr(route, "path", "") for route in content_router.router.routes}
    assert paths == {"/content/finalizations"}
