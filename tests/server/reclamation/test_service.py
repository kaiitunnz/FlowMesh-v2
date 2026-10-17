"""The content audit service: one sweep marks the control store, reads each bound tree
once, and lists the store against both; it runs until stopped on its own thread."""

import logging
import threading
from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest

from server.clients.redis import task_artifacts_key
from server.reclamation.audit import AuditTrail
from server.reclamation.closures import TreeClosures
from server.reclamation.collector import AuditCollector
from server.reclamation.marking import Marker
from server.reclamation.service import ContentAuditService, start_content_audit
from server.reclamation.store_admin import StoreAdmin
from server.registries.artifacts import artifact_field
from shared.content import (
    BACKEND_FILESYSTEM,
    ContentReference,
    ObjectStoreConfig,
    reference_for,
)
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.reclamation.test_store_admin import NOW, PREFIX, ReadOnlyS3
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_LOGGER = logging.getLogger("test.reclamation.service")
_JOIN_SEC = 5


class _Waits(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.waits = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        return self.is_set()


class _Fixture:
    def __init__(self, reads_per_pause: int = 1000) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.s3 = ReadOnlyS3()
        self.manifests: dict[str, bytes] = {}
        self.opened: list[str] = []
        self.stop = _Waits()
        self.waits_at_open: list[int] = []
        self.marker = Marker(
            self.rds, _LOGGER, min_pause_sec=0, duty=1.0, clock=lambda: NOW
        )
        self.service = ContentAuditService(
            self.marker,
            TreeClosures(self.open, _LOGGER),
            AuditCollector(
                StoreAdmin(self.s3, "bucket", PREFIX),
                _LOGGER,
                AuditTrail(self.rds),
                duty=1.0,
                min_pause_sec=0,
            ),
            _LOGGER,
            interval_sec=0,
            reads_per_pause=reads_per_pause,
            duty=1.0,
            min_pause_sec=0,
        )

    def open(self, reference: ContentReference) -> Iterator[bytes]:
        self.opened.append(reference.content_digest)
        self.waits_at_open.append(self.stop.waits)
        return iter([self.manifests[reference.content_digest]])

    def stored(self, data: bytes) -> ContentReference:
        self.s3.put(_A, data)
        return reference_for(_A, data)

    def bind(self, *bindings: ArtifactBinding) -> None:
        self.raw.hset(
            task_artifacts_key("tsk-1"),
            mapping={artifact_field(*b.key): b.model_dump_json() for b in bindings},
        )

    def tree(self, name: str, manifest: bytes, files: int = 0) -> ArtifactBinding:
        reference = reference_for(_A, manifest, media_type=TREE_MEDIA_TYPE)
        self.s3.put(_A, manifest)
        self.manifests[reference.content_digest] = manifest
        return ArtifactBinding(
            name=name,
            kind=ArtifactKind.TREE,
            reference=reference,
            size_bytes=len(manifest),
            files=files,
        )


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def _seed(fx: _Fixture) -> None:
    weights = fx.stored(b"weights")
    members = [fx.stored(data) for data in (b"member-1", b"member-2")]
    manifest = TreeManifest.of(
        TreeMember(f"f{i}", ref.content_digest, ref.size_bytes)
        for i, ref in enumerate(members)
    ).encode()
    fx.bind(
        ArtifactBinding(
            name="weights",
            kind=ArtifactKind.FILE,
            reference=weights,
            size_bytes=weights.size_bytes,
        ),
        fx.tree("ckpt", manifest, files=2),
    )


def test_a_sweep_reports_only_the_object_no_binding_names(fx) -> None:
    _seed(fx)
    fx.s3.put(_A, b"orphan")

    result = fx.service.sweep(threading.Event())

    (scope,) = result.scopes
    assert scope.complete
    assert (scope.objects, scope.bound, scope.unbound) == (5, 4, 1)
    assert scope.unbound_bytes == len(b"orphan")


def test_a_second_sweep_does_not_read_a_tree_again(fx) -> None:
    _seed(fx)

    fx.service.sweep(threading.Event())
    result = fx.service.sweep(threading.Event())

    assert len(fx.opened) == 1
    assert result.scopes[0].complete and result.scopes[0].bound == 4


def test_a_malformed_tree_leaves_its_scope_incomplete(fx) -> None:
    fx.bind(fx.tree("broken", b"not a tree"))
    fx.s3.put(_A, b"orphan")

    for _ in range(2):
        result = fx.service.sweep(threading.Event())
        (scope,) = result.scopes
        assert not scope.complete and "cannot be read as a tree" in scope.gap
        assert scope.objects == 2 and scope.unbound == 0
    assert len(fx.opened) == 1


def test_a_complete_pass_sizes_the_next_mark(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(fx)
    capacities: list[dict[str, int]] = []
    real_run = fx.marker.run

    def run(stop: threading.Event, sized: Any = None) -> Any:
        capacities.append(dict(sized or {}))
        return real_run(stop, sized)

    monkeypatch.setattr(fx.marker, "run", run)
    fx.service.sweep(threading.Event())
    fx.service.sweep(threading.Event())

    assert capacities == [{}, {_A: 2}]


def test_run_survives_a_failed_sweep_and_stops_on_stop(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = threading.Event()
    calls: list[int] = []

    def sweep(stop_: threading.Event) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("store unreachable")
        stop.set()

    monkeypatch.setattr(fx.service, "sweep", sweep)
    thread = threading.Thread(target=fx.service.run, args=(stop,), daemon=True)
    thread.start()
    thread.join(_JOIN_SEC)

    assert not thread.is_alive()
    assert len(calls) == 2


def test_start_content_audit_runs_a_daemon_thread_until_stopped(fx) -> None:
    stop = threading.Event()

    thread = start_content_audit(
        fx.rds,
        ObjectStoreConfig(backend=BACKEND_FILESYSTEM),
        fx.open,
        _LOGGER,
        stop,
    )

    assert thread.name == "content-audit" and thread.daemon
    stop.set()
    thread.join(_JOIN_SEC)
    assert not thread.is_alive()


def test_reading_many_small_trees_rests_between_them() -> None:
    fx = _Fixture(reads_per_pause=10)
    trees = []
    for t in range(50):
        members = [fx.stored(f"tree-{t}-member-{m}".encode()) for m in range(3)]
        manifest = TreeManifest.of(
            TreeMember(f"f{i}", ref.content_digest, ref.size_bytes)
            for i, ref in enumerate(members)
        ).encode()
        trees.append(fx.tree(f"tree-{t}", manifest, files=3))
    fx.bind(*trees)

    result = fx.service.sweep(fx.stop)

    assert len(fx.opened) == 50 and result.scopes[0].complete
    # A tick per tree and per member, a rest every ten.
    assert fx.waits_at_open[-1] - fx.waits_at_open[0] >= 19
