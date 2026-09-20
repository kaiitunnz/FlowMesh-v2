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
)
from worker.content import ContentAccessRegistry, WorkerContentPlane
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


class _RecordingLane:
    """A lane that records what the plane put in its cache."""

    def __init__(self, store: Any) -> None:
        self.store = store

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
