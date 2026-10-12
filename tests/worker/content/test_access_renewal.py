"""A long S3 write against access renewed, or running out, while it is in flight."""

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
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


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _access(
    grant_id: str, clock: _Clock, *, ttl: float = 900.0, session: str | None = None
) -> ContentStoreAccess:
    return ContentStoreAccess(
        grant=ContentStoreAccessGrant(
            grant_id=grant_id,
            task_id="tsk-1",
            authorization_scope="tenant-a",
            subject="wkr-1",
            subject_generation=1,
            operations=_OPS,
            backend_policy_version="test-1",
            expires_at_epoch=clock.now + ttl,
        ),
        credential=ScopedContentCredential(
            material={"access_key": session or grant_id}
        ),
    )


class _Clients:
    """One client per session over one store, recording which session carried each call.

    ``after`` runs once after the first call of an operation on a session's client;
    ``refuse`` runs once in place of it, and the call is refused for its session.
    """

    def __init__(self) -> None:
        self.s3 = FakeS3()
        self.calls: list[tuple[str, str]] = []
        self.opened: list[str] = []
        self.after: dict[tuple[str, str], _Hook] = {}
        self.refuse: dict[tuple[str, str], _Hook] = {}

    def open(self, credential: ScopedContentCredential) -> Any:
        session = credential.material["access_key"]
        self.opened.append(session)
        return _SessionClient(session, self)


class _SessionClient:
    def __init__(self, session: str, clients: _Clients) -> None:
        self._session = session
        self._clients = clients

    def __getattr__(self, op: str) -> Any:
        def call(**kwargs: Any) -> Any:
            clients = self._clients
            clients.calls.append((self._session, op))
            if (refusal := clients.refuse.pop((self._session, op), None)) is not None:
                refusal()
                raise s3_error("ExpiredToken", 400, op)
            hook = clients.after.pop((self._session, op), None)
            try:
                return getattr(clients.s3, op)(**kwargs)
            finally:
                if hook is not None:
                    hook()

        return call


class _Asks:
    """Renewal requests, which arrive on a thread of their own."""

    def __init__(self, answer: Callable[[], None] = lambda: None) -> None:
        self.count = 0
        self._answer = answer
        self._changed = threading.Condition()

    def __call__(self, task_id: str) -> None:
        with self._changed:
            self.count += 1
            self._changed.notify_all()
        self._answer()

    def settle(self, expected: int) -> int:
        """The count once it reaches ``expected``, or after a moment if not."""
        with self._changed:
            self._changed.wait_for(lambda: self.count >= expected, timeout=0.3)
        time.sleep(0.05)
        return self.count


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    clients: _Clients,
    clock: _Clock,
    request_access: Callable[[str], None] = lambda task_id: None,
    *,
    renewal_wait_sec: float = 0.1,
) -> ContentAccessRegistry:
    cfg = ObjectStoreConfig(
        backend=BACKEND_S3, bucket="bucket", part_bytes=_PART, upload_concurrency=1
    )
    registry = ContentAccessRegistry(
        cfg,
        request_access=request_access,
        arrival_wait_sec=0.1,
        renewal_wait_sec=renewal_wait_sec,
        clock=clock,
    )
    monkeypatch.setattr(registry, "_open_s3", clients.open)
    return registry


def _source(tmp_path: Path, parts: int = 3) -> Path:
    source = tmp_path / "artifact.bin"
    source.write_bytes(os.urandom(parts * _PART))
    return source


def test_a_grant_accepted_mid_upload_carries_the_rest_of_it(
    tmp_path, monkeypatch
) -> None:
    clock = _Clock()
    clients = _Clients()
    registry = _registry(monkeypatch, clients, clock)
    clients.after[("csg-1", "upload_part")] = lambda: registry.accept(
        _access("csg-2", clock)
    )
    registry.accept(_access("csg-1", clock))
    source = _source(tmp_path)
    store = registry.store_for("tsk-1", "tenant-a")

    reference = store.write_file("tenant-a", source)

    parts = [session for session, op in clients.calls if op == "upload_part"]
    assert parts == ["csg-1", "csg-2", "csg-2"]
    assert clients.calls[-1] == ("csg-2", "complete_multipart_upload")
    assert clients.opened == ["csg-1", "csg-2"]
    assert store.hydrate(reference) == source.read_bytes()


def test_parts_asking_at_once_share_the_client_one_of_them_opens(
    monkeypatch,
) -> None:
    clock = _Clock()
    clients = _Clients()
    registry = _registry(monkeypatch, clients, clock)

    entered = 0
    changed = threading.Condition()

    def slow_open(credential: ScopedContentCredential) -> Any:
        # Every caller that can get here at once does before any client opens.
        nonlocal entered
        with changed:
            entered += 1
            changed.notify_all()
            changed.wait_for(lambda: entered >= 4, timeout=0.5)
        return clients.open(credential)

    monkeypatch.setattr(registry, "_open_s3", slow_open)
    registry.accept(_access("csg-1", clock))
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
    assert entered == 1


def test_access_is_renewed_once_ahead_of_its_end(monkeypatch) -> None:
    clock = _Clock()
    asks = _Asks()
    registry = _registry(monkeypatch, _Clients(), clock, asks)
    registry.accept(_access("csg-1", clock, ttl=300.0))
    store = registry.store_for("tsk-1", "tenant-a")

    clock.now += 150.0
    store.write("tenant-a", b"early")
    assert asks.settle(1) == 0

    clock.now += 50.0
    for n in range(3):
        store.write("tenant-a", f"late {n}".encode())
    assert asks.settle(1) == 1


def test_short_lived_access_is_not_born_due_for_renewal(monkeypatch) -> None:
    clock = _Clock()
    asks = _Asks()
    registry = _registry(monkeypatch, _Clients(), clock, asks)
    registry.accept(_access("csg-1", clock, ttl=60.0))
    store = registry.store_for("tsk-1", "tenant-a")

    clock.now += 10.0
    store.write("tenant-a", b"fresh")
    assert asks.settle(1) == 0

    clock.now += 25.0
    store.write("tenant-a", b"due")
    assert asks.settle(1) == 1


def test_a_renewal_answered_with_the_same_expiry_is_asked_again_only_after_a_pause(
    monkeypatch,
) -> None:
    clock = _Clock()
    clients = _Clients()
    registry: ContentAccessRegistry
    expires = clock.now + 300.0
    regrants = iter(range(2, 100))

    def same_session_same_expiry() -> None:
        access = _access(f"csg-{next(regrants)}", clock, session="ak-1")
        grant = access.grant.model_copy(update={"expires_at_epoch": expires})
        registry.accept(access.model_copy(update={"grant": grant}))

    asks = _Asks(same_session_same_expiry)
    registry = _registry(monkeypatch, clients, clock, asks)
    registry.accept(_access("csg-1", clock, ttl=300.0, session="ak-1"))
    store = registry.store_for("tsk-1", "tenant-a")
    clock.now += 200.0

    for n in range(20):
        reference = store.write("tenant-a", f"call {n}".encode())
        store.hydrate(reference)
    assert asks.settle(2) == 1

    # The regrant is measured from its own arrival, so the next ask waits for its lead.
    clock.now += 11.0
    store.write("tenant-a", b"after the pause")
    assert asks.settle(2) == 1
    clock.now += 40.0
    store.write("tenant-a", b"within the regrant's lead")
    assert asks.settle(2) == 2
    assert clients.opened == ["ak-1"]


def test_a_failed_renewal_ask_is_asked_again_after_a_pause(monkeypatch) -> None:
    clock = _Clock()

    def unreachable() -> None:
        raise RuntimeError("Supervisor event stream not ready")

    asks = _Asks(unreachable)
    registry = _registry(monkeypatch, _Clients(), clock, asks)
    registry.accept(_access("csg-1", clock, ttl=300.0))
    store = registry.store_for("tsk-1", "tenant-a")
    clock.now += 200.0

    store.write("tenant-a", b"first")
    store.write("tenant-a", b"second")
    assert asks.settle(2) == 1

    clock.now += 11.0
    store.write("tenant-a", b"third")
    assert asks.settle(2) == 2


def test_a_renewal_ask_never_holds_a_write(monkeypatch) -> None:
    clock = _Clock()
    released = threading.Event()
    asks = _Asks(lambda: None if released.wait(5.0) else None)
    registry = _registry(monkeypatch, _Clients(), clock, asks)
    registry.accept(_access("csg-1", clock, ttl=300.0))
    store = registry.store_for("tsk-1", "tenant-a")
    clock.now += 200.0

    try:
        started = time.monotonic()
        for n in range(5):
            store.write("tenant-a", f"write {n}".encode())
        assert time.monotonic() - started < 1.0
        assert asks.settle(1) == 1
    finally:
        released.set()


def test_an_upload_whose_access_ran_out_is_aborted_at_once_on_its_last_client(
    tmp_path, monkeypatch
) -> None:
    clock = _Clock()
    clients = _Clients()
    registry = _registry(monkeypatch, clients, clock, renewal_wait_sec=5.0)
    source = _source(tmp_path)

    def expire_then_fail() -> None:
        clock.now += 1_000.0
        raise s3_error("InternalError", 500, "UploadPart")

    clients.after[("csg-1", "upload_part")] = expire_then_fail
    registry.accept(_access("csg-1", clock, ttl=300.0))
    store = registry.store_for("tsk-1", "tenant-a")

    started = time.monotonic()
    with pytest.raises(ContentStoreError):
        store.write_file("tenant-a", source)

    assert time.monotonic() - started < 2.0
    assert clients.calls[-1] == ("csg-1", "abort_multipart_upload")
    assert clients.s3.aborted == ["up-1"]
    assert clients.s3.uploads == {}


def test_a_call_refused_for_its_session_goes_again_on_a_renewed_one(
    tmp_path, monkeypatch
) -> None:
    clock = _Clock()
    clients = _Clients()
    registry = _registry(monkeypatch, clients, clock)
    clients.refuse[("csg-1", "upload_part")] = lambda: registry.accept(
        _access("csg-2", clock)
    )
    registry.accept(_access("csg-1", clock))
    source = _source(tmp_path)
    store = registry.store_for("tsk-1", "tenant-a")

    reference = store.write_file("tenant-a", source)

    parts = [session for session, op in clients.calls if op == "upload_part"]
    assert parts == ["csg-1", "csg-2", "csg-2", "csg-2"]
    assert store.hydrate(reference) == source.read_bytes()


def test_a_call_refused_with_no_renewed_session_is_not_repeated(monkeypatch) -> None:
    clock = _Clock()
    clients = _Clients()
    registry = _registry(monkeypatch, clients, clock)
    clients.refuse[("csg-1", "put_object")] = lambda: None
    registry.accept(_access("csg-1", clock))
    store = registry.store_for("tsk-1", "tenant-a")

    with pytest.raises(ContentStoreError):
        store.write("tenant-a", b"refused")

    assert [op for _, op in clients.calls] == ["put_object"]
