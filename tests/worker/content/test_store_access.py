"""The access a worker holds against the shared store, and what it refuses."""

import threading
import time
from collections.abc import Sequence
from typing import Any, cast

import pytest

from shared.content import (
    BACKEND_FILESYSTEM,
    ContentOperationKind,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ContentStoreError,
    ObjectStoreConfig,
    ScopedContentCredential,
    reference_for,
)
from worker.content import (
    ContentAccessRegistry,
    ContentLaneHost,
    WorkerContentCache,
    WorkerContentPlane,
)
from worker.content.access import ContentAccessDenied

_OPS = (ContentOperationKind.READ, ContentOperationKind.WRITE)


class _RefusingLane:
    """A cache lane that cannot represent the object at all."""

    def hydrate(self, reference: Any, task_id: str) -> bytes:
        raise ContentStoreError("this cache cannot hold that scope")


def _access(
    task_id: str, scope: str = "tenant-a", ttl: float = 900.0
) -> ContentStoreAccess:
    return ContentStoreAccess(
        grant=ContentStoreAccessGrant(
            grant_id="csg-1",
            task_id=task_id,
            authorization_scope=scope,
            subject="wkr-1",
            subject_generation=1,
            operations=_OPS,
            backend_policy_version="test-1",
            expires_at_epoch=time.time() + ttl,
        ),
        credential=ScopedContentCredential(material={"token": "opaque"}),
    )


def _registry(tmp_path, wait: float = 0.1) -> ContentAccessRegistry:
    cfg = ObjectStoreConfig(
        backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
    )
    return ContentAccessRegistry(cfg, arrival_wait_sec=wait)


def test_a_granted_task_opens_the_store(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    store = registry.store_for("tsk-1", "tenant-a")
    reference = store.write("tenant-a", b"body")
    assert store.fetch(reference) == b"body"


def test_a_task_with_no_access_is_refused(tmp_path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-a")


def test_another_scope_is_refused_to_a_granted_task(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1", scope="tenant-a"))
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-b")


def test_expired_access_is_refused_and_dropped(tmp_path) -> None:
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1", ttl=-1.0))
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-a")


def test_expired_access_is_renewed_on_request(tmp_path) -> None:
    requested: list[str] = []
    cfg = ObjectStoreConfig(
        backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
    )
    registry: ContentAccessRegistry

    def renew(task_id: str) -> None:
        requested.append(task_id)
        registry.accept(_access(task_id))

    registry = ContentAccessRegistry(
        cfg, request_access=renew, arrival_wait_sec=0.1, renewal_wait_sec=1.0
    )
    registry.accept(_access("tsk-1", ttl=0.05))
    time.sleep(0.1)

    store = registry.store_for("tsk-1", "tenant-a")
    assert requested == ["tsk-1"]
    assert store.fetch(store.write("tenant-a", b"body")) == b"body"


def test_a_renewal_control_refuses_fails_closed(tmp_path) -> None:
    requested: list[str] = []
    cfg = ObjectStoreConfig(
        backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
    )
    registry = ContentAccessRegistry(
        cfg, request_access=requested.append, arrival_wait_sec=0.1, renewal_wait_sec=0.1
    )
    registry.accept(_access("tsk-1", ttl=0.05))
    time.sleep(0.1)

    started = time.monotonic()
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-a")
    assert requested == ["tsk-1"]
    assert time.monotonic() - started < 1.0


def test_a_read_waits_for_access_control_is_still_relaying(tmp_path) -> None:
    """A task reaches its first write before its access lands, and still runs.

    Control relays the access on its own path, so a dispatch that writes content
    immediately can arrive first. The read waits out that gap rather than failing a
    task that was in fact given access.
    """
    registry = _registry(tmp_path, wait=5.0)
    threading.Timer(0.2, lambda: registry.accept(_access("tsk-1"))).start()

    started = time.monotonic()
    store = registry.store_for("tsk-1", "tenant-a")
    assert store.write("tenant-a", b"body") is not None
    assert time.monotonic() - started >= 0.2


def test_the_wait_is_bounded_and_fails_closed(tmp_path) -> None:
    registry = _registry(tmp_path, wait=0.2)
    started = time.monotonic()
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-a")
    assert time.monotonic() - started < 2.0


def test_a_plane_with_no_cache_still_reaches_the_shared_store(tmp_path) -> None:
    """Content lives in the shared store, so a deployment running no cache still writes.

    The cache is the optional half of the plane. Without it there is no lane to hold a
    copy or serve a peer, and every read and write goes to the store instead.
    """
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    plane = WorkerContentPlane(None, registry)

    reference = plane.write("tsk-1", "tenant-a", b"body", media_type="text/plain")
    assert plane.hydrate("tsk-1", reference) == b"body"


def test_a_cacheless_plane_takes_the_access_control_relays(tmp_path) -> None:
    registry = _registry(tmp_path)
    plane = WorkerContentPlane(None, registry)
    plane.route("content_access", _access("tsk-1").model_dump(mode="json"))

    assert registry.store_for("tsk-1", "tenant-a") is not None


def test_a_cache_that_cannot_serve_still_reads_the_object_from_the_store(
    tmp_path,
) -> None:
    """Whatever the cache raises, the read falls through to the store that has it.

    The cache keys objects by path segment, so a scope carrying a character a segment
    cannot hold is refused there — a limit of the copy, not of the object. A read that
    surfaced it would fail work the shared store could serve.
    """
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    plane = WorkerContentPlane(cast(Any, _RefusingLane()), registry)
    reference = registry.store_for("tsk-1", "tenant-a").write("tenant-a", b"body")

    assert plane.hydrate("tsk-1", reference) == b"body"


class _BrokenLane:
    """A cache whose backing directory has gone unreadable under it."""

    def hydrate(self, reference: Any, task_id: str) -> bytes:
        raise OSError("input/output error")


def test_a_cache_that_raises_anything_at_all_still_falls_through(tmp_path) -> None:
    """The fall-through is unconditional, not a list of anticipated failures.

    A content directory that has gone unreadable, a lane that is not running, a bug in
    the cache: none of them is a reason to fail a read the shared store can serve, and
    each raises something different.
    """
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    plane = WorkerContentPlane(cast(Any, _BrokenLane()), registry)
    reference = registry.store_for("tsk-1", "tenant-a").write("tenant-a", b"body")

    assert plane.hydrate("tsk-1", reference) == b"body"


class _RecordingLane:
    """A lane that records what the plane put in its cache."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def keep(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
        self.store.write(scope, data, media_type=media_type)
        return reference_for(scope, data, media_type=media_type)

    def hydrate(self, reference: Any, task_id: str) -> bytes:
        raise ContentStoreError("nothing cached here")


def test_an_outcome_is_cached_and_announced_like_any_other_content(tmp_path) -> None:
    """An outcome is content, so the copy and the announcement are the same as a write.

    Materializing an outcome goes through the plane rather than straight to the store,
    so the worker keeps the copy and control learns of it — without which no peer could
    ever be served an outcome and the whole cache-to-cache path would be dead for the
    fabric's main producer.
    """
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    cached: list[tuple[str, bytes]] = []
    announced: list[Sequence[tuple[str, str]]] = []

    class _Store:
        def write(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
            cached.append((scope, data))

    class _Index:
        def find(self, scope: str, idem: str) -> Any:
            return None

        def record(self, scope: str, idem: str, content: Any) -> Any:
            return content

    plane = WorkerContentPlane(
        cast(Any, _RecordingLane(_Store())),
        registry,
        announce=announced.append,
        finalizations=cast(Any, _Index()),
    )
    store = plane.outcome_store("tsk-1")
    assert store is not None
    reference = store.write("tenant-a", b"outcome body", media_type="text/plain")

    assert cached == [("tenant-a", b"outcome body")]
    assert announced == [[("tenant-a", reference.content_digest)]]


def test_a_copy_the_cache_cannot_keep_is_never_announced(tmp_path) -> None:
    """A write past the whole disk budget lands in the store but claims no copy.

    Announcing it would point a peer's read at a holder that already dropped it.
    """
    registry = _registry(tmp_path)
    registry.accept(_access("tsk-1"))
    announced: list[Sequence[tuple[str, str]]] = []
    lane = ContentLaneHost(
        store=WorkerContentCache(tmp_path / "cache", max_bytes=4),
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id: None,
        worker_id="wkr-1",
        generation=1,
    )
    plane = WorkerContentPlane(lane, registry, announce=announced.append)

    reference = plane.write("tsk-1", "tenant-a", b"larger than the budget")

    assert not lane.store.holds(reference)
    assert announced == []
    assert registry.store_for("tsk-1", "tenant-a").hydrate(reference) == (
        b"larger than the budget"
    )


def test_a_renewal_request_that_cannot_be_sent_fails_closed(tmp_path) -> None:
    def unreachable(task_id: str) -> None:
        raise RuntimeError("Supervisor event stream not ready")

    cfg = ObjectStoreConfig(
        backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
    )
    registry = ContentAccessRegistry(
        cfg, request_access=unreachable, arrival_wait_sec=0.1, renewal_wait_sec=5.0
    )
    started = time.monotonic()
    with pytest.raises(ContentAccessDenied):
        registry.store_for("tsk-1", "tenant-a")
    assert time.monotonic() - started < 1.0
