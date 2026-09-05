"""The worker HTTP content-store client, driven against the real content router."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routers.v1 import content as content_router
from server.services.content_store import ServerContentStore
from shared.outcome import OutcomeHydrationError
from shared.outcome.http_store import HttpFabricContentStore


class _RequestsShim:
    """Maps the ``requests`` surface the client uses onto a FastAPI ``TestClient``."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def get(
        self, url, params=None, headers=None, timeout=None
    ):  # noqa: ANN001 - test shim
        return self._client.get(url, params=params, headers=headers)

    def put(
        self, url, params=None, data=None, headers=None, timeout=None
    ):  # noqa: ANN001
        return self._client.put(url, params=params, content=data, headers=headers)


@pytest.fixture
def store(tmp_path, monkeypatch) -> HttpFabricContentStore:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.http_content_store")
    app.state.content_store = ServerContentStore(tmp_path / "content")
    app.include_router(content_router.router, prefix="/api/v1")
    monkeypatch.setattr(
        "shared.outcome.http_store.requests", _RequestsShim(TestClient(app))
    )
    return HttpFabricContentStore("http://testserver")


def test_materialize_then_hydrate(store) -> None:
    manifest = store.materialize("idm-1", b"payload", media_type="application/json")
    assert store.hydrate(manifest) == b"payload"


def test_materialize_finds_prior_under_idem(store) -> None:
    first = store.materialize("idm-2", b"a", media_type="application/json")
    assert store.find("idm-2") == first
    second = store.materialize("idm-2", b"a", media_type="application/json")
    assert first == second


def test_find_missing_returns_none(store) -> None:
    assert store.find("idm-absent") is None


def test_hydrate_missing_raises(store) -> None:
    manifest = store.materialize("idm-3", b"a", media_type="application/json")
    tampered = manifest.model_copy(update={"content_digest": "0" * 64})
    with pytest.raises(OutcomeHydrationError):
        store.hydrate(tampered)
