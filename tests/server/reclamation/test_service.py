"""The reclamation service: it marks the hooks live, backfills what its generation has
not read, then sweeps, and starts nothing when switched off."""

import logging
import threading
from typing import Any

import fakeredis
import pytest

from server.clients.redis import RECLAIM_ROOT_EPOCH_KEY
from server.config import ContentStoreConfig
from server.reclamation.backfill import FAMILIES, Backfill
from server.reclamation.certification import CoverageStore
from server.reclamation.collector import AuditCollector
from server.reclamation.multipart import MultipartOwner
from server.reclamation.service import ReclamationService, start_reclamation
from server.reclamation.store_admin import StoreAdmin
from server.reclamation.upkeep import MultipartUpkeep
from shared.content import BACKEND_FILESYSTEM, ObjectStoreConfig
from tests.server.reclamation.test_collector import StubS3
from tests.server.redis_helpers import fake_redis_client

_LOGGER = logging.getLogger("test.reclamation.service")


class _Fixture:
    def __init__(self) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.coverage = CoverageStore(self.rds, FAMILIES)
        self.s3 = StubS3()
        self.backfill = Backfill(
            self.rds, self.coverage, _LOGGER, page_pause_sec=0, retry_pause_sec=0
        )
        self.collector = AuditCollector(
            self.rds,
            self.coverage,
            StoreAdmin(self.s3, "bucket", "fabric"),
            _LOGGER,
            min_age_sec=0,
            page_pause_sec=0,
        )

    def service(self, epoch: int) -> ReclamationService:
        return ReclamationService(
            lambda: epoch,
            self.backfill,
            self.collector,
            self.coverage,
            _LOGGER,
            interval_sec=0,
        )


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def test_the_service_backfills_then_sweeps_under_its_own_generation(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.s3.put("org-a", "orphan", at=0)
    stop = threading.Event()
    sweeps: list[Any] = []
    real_sweep = fx.collector.sweep

    def sweep_once(stop_: threading.Event | None = None) -> Any:
        sweeps.append(real_sweep(stop_))
        stop.set()
        return sweeps[-1]

    monkeypatch.setattr(fx.collector, "sweep", sweep_once)
    fx.service(7).run(stop)

    assert fx.coverage.generation() == 7
    assert fx.coverage.completed(7) == set(FAMILIES)
    assert len(sweeps) == 1 and sweeps[0].orphans == 1


def test_a_restarted_root_keeps_the_generation_and_reads_nothing_again(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.s3.put("org-a", "orphan", at=0)
    stop = threading.Event()
    monkeypatch.setattr(fx.collector, "sweep", lambda stop_=None: stop.set())
    fx.service(7).run(stop)
    assert fx.coverage.certify("org-a").certified

    fx.raw.set(RECLAIM_ROOT_EPOCH_KEY, "8")
    pages: list[int] = []
    original = fx.backfill._scan_page

    def counted(cursor: int) -> Any:
        pages.append(cursor)
        return original(cursor)

    monkeypatch.setattr(fx.backfill, "_scan_page", counted)
    stop.clear()
    fx.service(8).run(stop)

    assert pages == []
    assert fx.coverage.generation() == 7
    assert fx.coverage.certify("org-a").certified


def test_the_service_does_not_start_under_a_later_root(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.raw.set(RECLAIM_ROOT_EPOCH_KEY, "9")
    backfilled: list[int] = []
    monkeypatch.setattr(
        fx.backfill, "run", lambda generation, stop: backfilled.append(generation)
    )

    fx.service(7).run(threading.Event())

    assert backfilled == []
    assert fx.coverage.generation() is None


@pytest.mark.parametrize("later", ["root", "generation"])
def test_the_service_stops_sweeping_once_a_later_root_takes_over(
    fx, monkeypatch: pytest.MonkeyPatch, later: str
) -> None:
    swept: list[int] = []

    def sweep(stop: threading.Event | None = None) -> None:
        swept.append(1)
        if later == "root":
            fx.raw.set(RECLAIM_ROOT_EPOCH_KEY, "8")
        else:
            fx.coverage.record_fault(7, None)
            fx.coverage.mark_hooks_on(8)

    monkeypatch.setattr(fx.collector, "sweep", sweep)
    fx.service(7).run(threading.Event())

    assert swept == [1]


def _no_owner(
    task_id: str, worker_id: str, dispatch_id: str | None
) -> tuple[str, MultipartOwner] | None:
    return None


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> dict[str, threading.Event]:
    events = {
        "content-reclamation": threading.Event(),
        "multipart-upkeep": threading.Event(),
    }
    monkeypatch.setattr(
        ReclamationService,
        "run",
        lambda self, stop: events["content-reclamation"].set(),
    )
    monkeypatch.setattr(
        MultipartUpkeep,
        "run",
        lambda self, epoch, stop: events["multipart-upkeep"].set(),
    )
    return events


def _started(enabled: bool, fx: _Fixture, store: ObjectStoreConfig) -> list[str]:
    threads = start_reclamation(
        enabled, fx.rds, store, lambda: 1, _no_owner, _LOGGER, threading.Event()
    )
    for thread in threads:
        thread.join(5)
    return sorted(thread.name for thread in threads)


def test_reclamation_switched_off_still_keeps_multipart_records(fx, ran) -> None:
    assert _started(False, fx, ObjectStoreConfig()) == ["multipart-upkeep"]

    assert ran["multipart-upkeep"].is_set()
    assert not ran["content-reclamation"].is_set()
    assert not list(fx.raw.scan_iter("rc:*"))


def test_reclamation_switched_on_runs_both_on_their_own_threads(fx, ran) -> None:
    assert _started(True, fx, ObjectStoreConfig()) == [
        "content-reclamation",
        "multipart-upkeep",
    ]

    assert all(event.is_set() for event in ran.values())


def test_a_filesystem_store_needs_no_multipart_upkeep(fx, ran) -> None:
    store = ObjectStoreConfig(backend=BACKEND_FILESYSTEM)

    assert _started(False, fx, store) == []
    assert _started(True, fx, store) == ["content-reclamation"]


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("false", False)])
def test_reclamation_is_on_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("CONTENT_RECLAMATION_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CONTENT_RECLAMATION_ENABLED", value)

    assert ContentStoreConfig.from_env().reclamation_enabled is expected
