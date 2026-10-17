"""The finalization index's HTTP surface: bind once, resolve, and stay in scope.

The scope a binding lands in is the one control assigned the work, so these exercise the
surface against an index control has already written that assignment into — and against
keys it has not.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

import fakeredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from server.clients.redis import RECLAIM_ROOT_EPOCH_KEY, reclaim_history_key
from server.content import FinalizationIndex
from server.reclamation.pins import Nonces
from server.routers.v1 import content as content_router
from shared.content import ContentReference, reference_for
from shared.outcome import OutcomeManifest
from tests.server.redis_helpers import fake_redis_client

PREFIX = "/api/v1"
_CONTENT = reference_for("local", b"result-body", media_type="application/json")


_EPOCH = 1


def _app(
    verify: Callable[[ContentReference], bool | None] = lambda _: True,
    on_fenced: Callable[[], None] = lambda: None,
) -> tuple[FastAPI, FinalizationIndex]:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.content_router")
    redis = fake_redis_client(fakeredis.FakeServer())
    redis.sync.set_value(RECLAIM_ROOT_EPOCH_KEY, str(_EPOCH))
    nonces = Nonces(_EPOCH)
    index = FinalizationIndex(redis, lambda: nonces, verify, on_fenced)
    app.state.finalization_index = index
    app.include_router(content_router.router, prefix=PREFIX)
    return app, index


def _client(
    verify: Callable[[ContentReference], bool | None] = lambda _: True,
    on_fenced: Callable[[], None] = lambda: None,
) -> tuple[TestClient, FinalizationIndex]:
    app, index = _app(verify, on_fenced)
    return TestClient(app), index


def _assigned(
    idem: str,
    scope: str = "local",
    verify: Callable[[ContentReference], bool | None] = lambda _: True,
    on_fenced: Callable[[], None] = lambda: None,
) -> tuple[TestClient, FinalizationIndex]:
    client, index = _client(verify, on_fenced)
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


def test_a_refused_finalization_answers_its_disposition() -> None:
    client, index = _assigned("idm-9", verify=lambda _: False)
    raw = index._rds.sync._control
    raw.hset(reclaim_history_key("local"), _CONTENT.content_digest, "1")
    resp = _bind(client, "idm-9")
    assert resp.status_code == 409
    assert resp.json()["detail"] == {"refusal": "rematerialize"}


def test_a_fenced_finalization_answers_503_and_stops_the_root() -> None:
    fenced: list[bool] = []
    client, index = _assigned("idm-10", on_fenced=lambda: fenced.append(True))
    index._rds.sync.set_value(RECLAIM_ROOT_EPOCH_KEY, str(_EPOCH + 1))
    assert _bind(client, "idm-10").status_code == 503
    assert fenced == [True]


def test_the_schema_declares_the_refusal_and_the_fence() -> None:
    app, _index = _app()
    responses = app.openapi()["paths"][f"{PREFIX}/content/finalizations"]["put"][
        "responses"
    ]
    assert {"409", "503"} <= responses.keys()


@pytest.mark.anyio
async def test_a_binding_waiting_on_the_store_does_not_hold_the_event_loop() -> None:
    # The binding below verifies a retired identity's backing, a store read that waits
    # on ``release``; a lookup must still be answered while it waits.
    started = threading.Event()
    release = threading.Event()
    waited: list[bool] = []

    def verify(_: ContentReference) -> bool:
        started.set()
        waited.append(release.wait(10))
        return True

    app, index = _app(verify)
    index.assign_scope("idm-11", "local")
    index.assign_scope("idm-12", "local")
    index._rds.sync._control.hset(
        reclaim_history_key("local"), _CONTENT.content_digest, "1"
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        binding = asyncio.create_task(
            client.put(
                f"{PREFIX}/content/finalizations",
                params={"idem": "idm-11"},
                json=_CONTENT.model_dump(mode="json"),
            )
        )
        assert await asyncio.to_thread(started.wait, 10)
        lookup = await client.get(
            f"{PREFIX}/content/finalizations", params={"idem": "idm-12"}
        )
        release.set()
        assert lookup.status_code == 404
        assert (await binding).status_code == 200
    assert waited == [True]


def test_a_lookup_reads_the_control_store_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, index = _assigned("idm-13")
    on_loop: list[bool] = []
    find = index.find

    def spy(scope: str, idem: str) -> Any:
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return find(scope, idem)

    monkeypatch.setattr(index, "find", spy)
    client.get(f"{PREFIX}/content/finalizations", params={"idem": "idm-13"})
    assert on_loop == [False]
