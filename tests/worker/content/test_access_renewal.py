"""A long S3 write against access renewed, or running out, while it is in flight."""

import os
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from shared.content import (
    BACKEND_S3,
    ContentOperationKind,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ContentStoreError,
    ObjectStoreConfig,
    ScopedContentCredential,
)
from tests.shared.fake_s3 import FakeS3, s3_error
from worker.content import ContentAccessRegistry

_PART = 5 << 20
_OPS = (ContentOperationKind.READ, ContentOperationKind.WRITE)

type _Hook = Callable[[], None]


def _access(grant_id: str, ttl: float = 900.0) -> ContentStoreAccess:
    return ContentStoreAccess(
        grant=ContentStoreAccessGrant(
            grant_id=grant_id,
            task_id="tsk-1",
            authorization_scope="tenant-a",
            subject="wkr-1",
            subject_generation=1,
            operations=_OPS,
            backend_policy_version="test-1",
            expires_at_epoch=time.time() + ttl,
        ),
        credential=ScopedContentCredential(material={"token": grant_id}),
    )


class _Clients:
    """One client per grant over one store, recording which grant carried each call.

    ``hooks`` runs once after the first call of an operation on a grant's client.
    """

    def __init__(self, hooks: dict[tuple[str, str], _Hook] | None = None) -> None:
        self.s3 = FakeS3()
        self.calls: list[tuple[str, str]] = []
        self.opened: list[str] = []
        self._hooks = {} if hooks is None else hooks

    def open(self, credential: ScopedContentCredential) -> Any:
        grant_id = credential.material["token"]
        self.opened.append(grant_id)
        return _GrantClient(grant_id, self)


class _GrantClient:
    def __init__(self, grant_id: str, clients: _Clients) -> None:
        self._grant_id = grant_id
        self._clients = clients

    def __getattr__(self, op: str) -> Any:
        def call(**kwargs: Any) -> Any:
            clients = self._clients
            clients.calls.append((self._grant_id, op))
            hook = clients._hooks.pop((self._grant_id, op), None)
            try:
                return getattr(clients.s3, op)(**kwargs)
            finally:
                if hook is not None:
                    hook()

        return call


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    clients: _Clients,
    request_access: Callable[[str], None] = lambda task_id: None,
) -> ContentAccessRegistry:
    cfg = ObjectStoreConfig(
        backend=BACKEND_S3, bucket="bucket", part_bytes=_PART, upload_concurrency=1
    )
    registry = ContentAccessRegistry(
        cfg, request_access=request_access, arrival_wait_sec=0.1, renewal_wait_sec=0.1
    )
    monkeypatch.setattr(registry, "_open_s3", clients.open)
    return registry


def _source(tmp_path, parts: int = 3):
    source = tmp_path / "artifact.bin"
    source.write_bytes(os.urandom(parts * _PART))
    return source


def test_a_grant_accepted_mid_upload_carries_the_rest_of_it(
    tmp_path, monkeypatch
) -> None:
    hooks: dict[tuple[str, str], _Hook] = {}
    clients = _Clients(hooks)
    registry = _registry(monkeypatch, clients)
    hooks[("csg-1", "upload_part")] = lambda: registry.accept(_access("csg-2"))
    registry.accept(_access("csg-1"))
    source = _source(tmp_path)
    store = registry.store_for("tsk-1", "tenant-a")

    reference = store.write_file("tenant-a", source)

    parts = [grant for grant, op in clients.calls if op == "upload_part"]
    assert parts == ["csg-1", "csg-2", "csg-2"]
    assert clients.calls[-1] == ("csg-2", "complete_multipart_upload")
    assert clients.opened == ["csg-1", "csg-2"]
    assert store.hydrate(reference) == source.read_bytes()


def test_parts_asking_at_once_share_the_client_one_of_them_opens(
    monkeypatch,
) -> None:
    clients = _Clients()
    registry = _registry(monkeypatch, clients)

    def slow_open(credential: ScopedContentCredential) -> Any:
        time.sleep(0.05)
        return clients.open(credential)

    monkeypatch.setattr(registry, "_open_s3", slow_open)
    registry.accept(_access("csg-1"))
    store = registry.store_for("tsk-1", "tenant-a")
    threads = [
        threading.Thread(target=store.write, args=("tenant-a", f"part {n}".encode()))
        for n in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert clients.opened == ["csg-1"]


def test_access_close_to_its_end_is_renewed_once_ahead_of_time(monkeypatch) -> None:
    requested: list[str] = []
    registry = _registry(monkeypatch, _Clients(), requested.append)
    registry.accept(_access("csg-1", ttl=60.0))
    store = registry.store_for("tsk-1", "tenant-a")

    for n in range(3):
        store.write("tenant-a", f"body {n}".encode())

    assert requested == ["tsk-1"]


def test_a_renewal_request_that_failed_is_asked_again(monkeypatch) -> None:
    requested: list[str] = []

    def flaky(task_id: str) -> None:
        requested.append(task_id)
        if len(requested) == 1:
            raise RuntimeError("Supervisor event stream not ready")

    registry = _registry(monkeypatch, _Clients(), flaky)
    registry.accept(_access("csg-1", ttl=60.0))
    store = registry.store_for("tsk-1", "tenant-a")

    for n in range(3):
        store.write("tenant-a", f"body {n}".encode())

    assert requested == ["tsk-1", "tsk-1"]


def test_an_upload_whose_access_ran_out_is_aborted_on_its_last_client(
    tmp_path, monkeypatch
) -> None:
    access = _access("csg-1", ttl=1.0)

    def expire_then_fail() -> None:
        time.sleep(max(0.0, access.grant.expires_at_epoch - time.time()) + 0.1)
        raise s3_error("InternalError", 500, "UploadPart")

    clients = _Clients({("csg-1", "upload_part"): expire_then_fail})
    registry = _registry(monkeypatch, clients)
    source = _source(tmp_path)
    registry.accept(access)
    store = registry.store_for("tsk-1", "tenant-a")

    with pytest.raises(ContentStoreError):
        store.write_file("tenant-a", source)

    assert clients.calls[-1] == ("csg-1", "abort_multipart_upload")
    assert clients.s3.aborted == ["up-1"]
    assert clients.s3.uploads == {}
