"""The reclamation service: it marks the hooks live under its own epoch, backfills,
then sweeps, and starts nothing when switched off."""

import logging
import threading
from typing import Any

import fakeredis
import pytest

from server.config import ContentStoreConfig
from server.reclamation.backfill import FAMILIES, Backfill
from server.reclamation.certification import CoverageStore
from server.reclamation.collector import AuditCollector
from server.reclamation.service import ReclamationService, start_reclamation
from server.reclamation.store_admin import StoreAdmin
from shared.content import ObjectStoreConfig
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


def test_the_service_does_not_start_under_a_later_root(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.coverage.mark_hooks_on(9)
    backfilled: list[int] = []
    monkeypatch.setattr(
        fx.backfill, "run", lambda generation, stop: backfilled.append(generation)
    )

    fx.service(7).run(threading.Event())

    assert backfilled == []
    assert fx.coverage.generation() == 9


def test_the_service_stops_sweeping_once_a_later_root_takes_the_generation(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    swept: list[int] = []

    def sweep(stop: threading.Event | None = None) -> None:
        swept.append(1)
        fx.coverage.mark_hooks_on(8)

    monkeypatch.setattr(fx.collector, "sweep", sweep)
    fx.service(7).run(threading.Event())

    assert swept == [1]


def test_reclamation_switched_off_starts_nothing(fx) -> None:
    before = threading.active_count()

    thread = start_reclamation(
        False, fx.rds, ObjectStoreConfig(), lambda: 1, _LOGGER, threading.Event()
    )

    assert thread is None
    assert threading.active_count() == before
    assert not list(fx.raw.scan_iter("rc:*"))


def test_reclamation_switched_on_runs_on_its_own_thread(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = threading.Event()
    monkeypatch.setattr(ReclamationService, "run", lambda self, stop: ran.set())

    thread = start_reclamation(
        True, fx.rds, ObjectStoreConfig(), lambda: 1, _LOGGER, threading.Event()
    )

    assert thread is not None and thread.name == "content-reclamation"
    thread.join(5)
    assert ran.is_set()


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("false", False)])
def test_reclamation_is_on_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("CONTENT_RECLAMATION_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CONTENT_RECLAMATION_ENABLED", value)

    assert ContentStoreConfig.from_env().reclamation_enabled is expected
