"""The finalization index's HTTP surface: bind once, resolve, and stay in scope.

The scope a binding lands in is the one control assigned the work, so these exercise the
surface against an index control has already written that assignment into — and against
keys it has not.
"""

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


def _client() -> tuple[TestClient, FinalizationIndex]:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.content_router")
    index = FinalizationIndex(cast(Any, type("C", (), {"sync": _FakeRedis()})()))
    app.state.finalization_index = index
    app.include_router(content_router.router, prefix=PREFIX)
    return TestClient(app), index


def _assigned(idem: str, scope: str = "local") -> tuple[TestClient, FinalizationIndex]:
    client, index = _client()
    index.assign_scope(idem, scope)
    return client, index


def _bind(client: TestClient, idem: str, content: Any = None, **params: str) -> Any:
    return client.put(
        f"{PREFIX}/content/finalizations",
        params={"idem": idem, **params},
        json=(content or _CONTENT).model_dump(mode="json"),
    )


def test_a_finalization_binds_and_resolves() -> None:
    client, _index = _assigned("idm-1")
    bound = _bind(client, "idm-1")
    assert bound.status_code == 200
    manifest = OutcomeManifest.model_validate(bound.json())
    assert manifest.content == _CONTENT
    assert manifest.idempotency_key == "idm-1"

    resolved = client.get(f"{PREFIX}/content/finalizations", params={"idem": "idm-1"})
    assert resolved.status_code == 200
    assert OutcomeManifest.model_validate(resolved.json()) == manifest


def test_the_first_binding_stands() -> None:
    client, _index = _assigned("idm-2")
    first = _bind(client, "idm-2").json()
    other = reference_for("local", b"a different result", media_type="application/json")
    second = _bind(client, "idm-2", other).json()
    assert first == second


def test_an_assigned_key_with_no_binding_is_404() -> None:
    client, _index = _assigned("idm-absent")
    assert (
        client.get(
            f"{PREFIX}/content/finalizations", params={"idem": "idm-absent"}
        ).status_code
        == 404
    )


def test_a_key_control_assigned_no_scope_cannot_bind() -> None:
    # The reporting worker authenticates as the deployment, which carries the
    # deployment-wide scope. That must not let it bind work control never authorized:
    # the scope comes from the assignment, and there is none.
    client, _index = _client()
    assert _bind(client, "idm-unassigned").status_code == 403
    assert (
        client.get(
            f"{PREFIX}/content/finalizations", params={"idem": "idm-unassigned"}
        ).status_code
        == 403
    )


def test_a_binding_cannot_name_a_scope_other_than_the_one_assigned() -> None:
    # The scope on the request is an assertion, not an authority: a deployment-scoped
    # principal naming another scope is refused rather than obeyed.
    client, _index = _assigned("idm-3", "local")
    elsewhere = reference_for("acme", b"result-body", media_type="application/json")
    assert _bind(client, "idm-3", elsewhere, scope="acme").status_code == 403


def test_a_fabric_component_binds_in_the_scope_assigned_to_the_work() -> None:
    # The deployment principal legitimately finalizes on a tenant's behalf — when that
    # is the scope control assigned the key, and only then.
    client, _index = _assigned("idm-4", "acme")
    content = reference_for("acme", b"tenant result", media_type="application/json")
    bound = _bind(client, "idm-4", content, scope="acme")
    assert bound.status_code == 200
    assert OutcomeManifest.model_validate(bound.json()).content == content


def test_content_outside_the_admitted_scope_is_refused() -> None:
    client, _index = _assigned("idm-5", "local")
    elsewhere = _CONTENT.model_copy(update={"authorization_scope": "other"})
    assert _bind(client, "idm-5", elsewhere).status_code == 403


def test_no_payload_crosses_the_surface() -> None:
    # The content router binds finalizations and nothing else: bytes reach the shared
    # store directly, never through the server.
    paths = {getattr(route, "path", "") for route in content_router.router.routes}
    assert paths == {"/content/finalizations"}
