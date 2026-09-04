"""The content router: worker-authenticated materialize, hydrate, and idempotency."""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routers.v1 import content as content_router
from server.services.content_store import ServerContentStore
from shared.outcome import OutcomeManifest, content_digest

PREFIX = "/api/v1"


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.content_router")
    app.state.content_store = ServerContentStore(tmp_path / "content")
    app.include_router(content_router.router, prefix=PREFIX)
    return TestClient(app)


def test_put_then_hydrate_round_trip(tmp_path) -> None:
    client = _client(tmp_path)
    put = client.put(
        f"{PREFIX}/content",
        params={"idem": "idm-1"},
        content=b"result-body",
        headers={"Content-Type": "application/json"},
    )
    assert put.status_code == 200
    manifest = OutcomeManifest.model_validate(put.json())
    assert manifest.content_digest == content_digest(b"result-body")
    assert manifest.access.tenant == "local"

    got = client.get(f"{PREFIX}/content/{manifest.content_digest}")
    assert got.status_code == 200
    assert got.content == b"result-body"


def test_put_is_idempotent_under_idem(tmp_path) -> None:
    client = _client(tmp_path)
    first = client.put(
        f"{PREFIX}/content", params={"idem": "idm-2"}, content=b"x"
    ).json()
    second = client.put(
        f"{PREFIX}/content", params={"idem": "idm-2"}, content=b"x"
    ).json()
    assert first == second
    assert client.get(f"{PREFIX}/content/by-idem/idm-2").json() == first


def test_missing_content_is_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert (
        client.get(f"{PREFIX}/content/{content_digest(b'absent')}").status_code == 404
    )
    assert client.get(f"{PREFIX}/content/by-idem/idm-absent").status_code == 404


def test_store_disabled_is_404(tmp_path) -> None:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.content_router")
    app.state.content_store = None
    app.include_router(content_router.router, prefix=PREFIX)
    client = TestClient(app)
    assert (
        client.put(f"{PREFIX}/content", params={"idem": "i"}, content=b"x").status_code
        == 404
    )
