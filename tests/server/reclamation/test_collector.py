"""The audit collector: it counts what the store holds that no binding names, apart
from what is recent or uncovered, and reports it to its metrics and audit stream."""

import logging
import threading
import time
from typing import Any, cast

import fakeredis
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from server.clients.redis import CONTENT_AUDIT_KEY
from server.reclamation.audit import AuditTrail
from server.reclamation.closures import TreeClosures
from server.reclamation.collector import RECENT_SEC, AuditCollector
from server.reclamation.mark import Mark
from server.reclamation.marking import MarkPass
from server.reclamation.store_admin import (
    Eligible,
    Ineligibility,
    Ineligible,
    StoreAdmin,
)
from shared.content import content_digest, reference_for
from shared.content.tree import TREE_MEDIA_TYPE
from tests.server.reclamation.test_store_admin import NOW, OLD, PREFIX, ReadOnlyS3
from tests.server.redis_helpers import fake_redis_client

_A = "org-a"
_B = "org-b"
_LOGGER = logging.getLogger("test.collector")
_METRIC = "flowmesh.content.audit."


class _Fixture:
    def __init__(
        self,
        *,
        admin: bool = True,
        uploads_complete: bool = False,
        duty: float = 1.0,
        **s3: Any,
    ) -> None:
        server = fakeredis.FakeServer()
        self.rds = fake_redis_client(server)
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.s3 = ReadOnlyS3(**s3)
        self.reader = InMemoryMetricReader()
        self.collector = AuditCollector(
            StoreAdmin(self.s3, "bucket", PREFIX) if admin else None,
            _LOGGER,
            AuditTrail(self.rds),
            MeterProvider(metric_readers=[self.reader]).get_meter("test"),
            uploads_complete=uploads_complete,
            duty=duty,
            min_pause_sec=0,
        )
        self.marked = MarkPass(Mark(), NOW, complete=True)
        self.closures = TreeClosures(lambda ref: iter(()), _LOGGER)

    def sweep(
        self,
        tree_gaps: dict[str, str] | None = None,
        stop: threading.Event | None = None,
    ) -> Any:
        match self.collector.eligibility():
            case Ineligible() as ineligible:
                return self.collector.report_ineligible(ineligible, NOW)
            case Eligible() as eligible:
                return self.collector.sweep(
                    eligible,
                    self.marked,
                    self.closures,
                    tree_gaps or {},
                    time.monotonic(),
                    stop,
                )

    def gauges(self) -> dict[str, float]:
        data = self.reader.get_metrics_data()
        assert data is not None
        return {
            metric.name.removeprefix(_METRIC): point.value
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            for point in metric.data.data_points
        }

    def audit(self, kind: str) -> list[dict[str, str]]:
        entries = cast(list[Any], self.raw.xrange(CONTENT_AUDIT_KEY))
        return [fields for _, fields in entries if fields["kind"] == kind]


@pytest.fixture
def fx() -> _Fixture:
    return _Fixture()


def test_a_sweep_counts_each_object_by_what_binds_it(fx) -> None:
    bound = fx.s3.put(_A, b"bound")
    member = fx.s3.put(_A, b"tree member")
    fx.s3.put(_A, b"unbound", at=OLD)
    fx.s3.put(_A, b"edge", at=NOW - RECENT_SEC)
    fx.s3.put(_A, b"before the mark", at=NOW - 60)
    fx.s3.put(_A, b"after the mark", at=NOW + 60)
    fx.s3.put_key(f"{PREFIX}/{_A}/staging/leftover", size=3)
    fx.marked.mark.add(_A, bound)
    fx.closures.mark.add(_A, member)

    result = fx.sweep()

    (scope,) = result.scopes
    assert scope.complete and scope.gap == ""
    assert (scope.objects, scope.bound, scope.foreign) == (7, 2, 1)
    assert (scope.unbound, scope.unbound_bytes) == (1, len(b"unbound"))
    assert scope.recent == 3
    assert scope.recent_bytes == len(b"edge") + len(b"before the mark") + len(
        b"after the mark"
    )
    gauges = fx.gauges()
    assert (gauges["unbound"], gauges["unbound_bytes"]) == (1, len(b"unbound"))
    assert (gauges["recent_unbound"], gauges["incomplete_scopes"]) == (3, 0)
    assert gauges["eligible"] == 1
    assert gauges["mark_bytes"] == (
        fx.marked.mark.nbytes + fx.marked.trees_nbytes + fx.closures.nbytes
    )


def test_a_mark_counts_its_bound_trees_in_its_size(fx) -> None:
    fx.marked.add_tree(reference_for(_A, b"a manifest", media_type=TREE_MEDIA_TYPE))
    without = fx.marked.mark.nbytes + fx.closures.nbytes

    result = fx.sweep()

    assert result.mark_bytes == without + fx.marked.trees_nbytes > without


def test_a_digest_bound_in_another_scope_is_unbound_here(fx) -> None:
    digest = fx.s3.put(_A, b"shared bytes")
    fx.marked.mark.add(_B, digest)

    result = fx.sweep()

    assert {s.scope: s.unbound for s in result.scopes} == {_A: 1, _B: 0}


def test_an_uncovered_scope_reports_what_it_lists_and_no_unbound_count(fx) -> None:
    for scope in (_A, _B, "org-c"):
        fx.s3.put(scope, b"unbound")
        fx.s3.put(scope, b"recent", at=NOW)
    fx.marked.faults[_A] += 1

    result = fx.sweep({_B: "bound tree x is unavailable now"})

    by_scope = {s.scope: s for s in result.scopes}
    for scope, gap in ((_A, "1 undecodable value(s)"), (_B, "unavailable now")):
        audit = by_scope[scope]
        assert not audit.complete and gap in audit.gap
        assert audit.objects == 2
        assert (audit.unbound, audit.unbound_bytes, audit.recent) == (0, 0, 0)
    assert by_scope["org-c"].complete and by_scope["org-c"].unbound == 1
    assert (result.incomplete, result.unbound) == (2, 1)
    assert fx.gauges()["incomplete_scopes"] == 2
    stored = {e["scope"]: e for e in fx.audit("scope")}
    assert stored[_A]["complete"] == "false" and stored[_A]["gap"] == (
        "1 undecodable value(s)"
    )


def test_an_unfinished_mark_leaves_every_scope_incomplete(fx) -> None:
    fx.s3.put(_A, b"unbound")
    fx.marked.complete = False

    result = fx.sweep()

    assert result.scopes[0].gap == "the marking pass did not finish"
    assert result.unbound == 0


def test_a_scope_only_the_mark_names_is_audited(fx) -> None:
    fx.marked.mark.add(_A, content_digest(b"bound but absent"))

    result = fx.sweep()

    assert [(s.scope, s.objects) for s in result.scopes] == [(_A, 0)]


def test_the_audit_stream_states_the_recent_boundary(fx) -> None:
    fx.s3.put(_A, b"unbound")

    fx.sweep()

    boundary = (
        f"modified at or after {NOW - RECENT_SEC:.0f}, {RECENT_SEC:.0f}s before the "
        f"mark started at {NOW:.0f}"
    )
    (sweep,) = fx.audit("sweep")
    (scope,) = fx.audit("scope")
    assert sweep["recent_before"] == scope["recent_before"] == boundary
    assert sweep["sweep"] == scope["sweep"] == f"{NOW:.0f}"
    assert (sweep["unbound"], scope["unbound"]) == ("1", "1")


@pytest.mark.parametrize("known", [True, False])
def test_open_uploads_are_counted_only_where_their_listing_is_complete(
    known: bool,
) -> None:
    fx = _Fixture(uploads_complete=known)
    fx.s3.uploads = [{"Key": f"{PREFIX}/{_A}/sha256/x", "UploadId": "u-1"}]
    fx.s3.parts = {"u-1": [{"Size": 5}, {"Size": 6}]}

    result = fx.sweep()

    assert result.uploads is not None and result.uploads.known is known
    assert (result.uploads.uploads, result.uploads.parts, result.uploads.bytes) == (
        1,
        2,
        11,
    )
    assert result.uploads.by_scope == {_A: 1}
    gauges = fx.gauges()
    assert gauges["open_uploads_known"] == (1 if known else 0)
    if known:
        assert (gauges["open_uploads"], gauges["open_upload_bytes"]) == (1, 11)
    else:
        assert "open_uploads" not in gauges and "open_upload_bytes" not in gauges
    assert fx.audit("uploads")[0]["known"] == ("true" if known else "false")
    assert fx.audit("sweep")[0]["uploads_known"] == ("true" if known else "false")


def test_an_upload_whose_parts_cannot_be_listed_is_unsized(fx) -> None:
    fx.s3.uploads = [{"Key": f"{PREFIX}/{_A}/sha256/x", "UploadId": "u-1"}]
    fx.s3.fail.add("list_parts")

    result = fx.sweep()

    assert result.uploads is not None
    assert (result.uploads.uploads, result.uploads.unsized) == (1, 1)


@pytest.mark.parametrize("admin", [False, True])
def test_an_ineligible_store_publishes_nothing_an_audit_finds(admin: bool) -> None:
    fx = _Fixture(admin=admin)
    fx.s3.fail.add("list_multipart_uploads")

    result = fx.sweep()

    reason = Ineligibility.CANNOT_LIST if admin else Ineligibility.FILESYSTEM
    assert isinstance(result.eligibility, Ineligible)
    assert result.eligibility.reason is reason
    assert result.scopes == [] and result.uploads is None
    assert fx.gauges() == {"eligible": 0}
    (record,) = fx.audit("ineligible")
    assert record["reason"] == reason.value
    assert fx.audit("sweep") == fx.audit("scope") == []
    if admin:
        assert record["detail"] == "AccessDenied"


def test_a_scope_whose_listing_fails_is_incomplete_and_the_sweep_still_reports(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.s3.put(_A, b"unbound")
    fx.s3.put(_B, b"unbound")
    listed = fx.s3.list_object_versions

    def failing(**kwargs: Any) -> Any:
        if kwargs["Prefix"].startswith(f"{PREFIX}/{_A}/"):
            raise PermissionError("SlowDown")
        return listed(**kwargs)

    monkeypatch.setattr(fx.s3, "list_object_versions", failing)

    result = fx.sweep()

    by_scope = {s.scope: s for s in result.scopes}
    assert not by_scope[_A].complete and "SlowDown" in by_scope[_A].gap
    assert by_scope[_A].unbound == 0
    assert by_scope[_B].complete and by_scope[_B].unbound == 1
    assert fx.gauges()["incomplete_scopes"] == 1
    (sweep,) = fx.audit("sweep")
    assert (sweep["incomplete"], sweep["unbound"]) == ("1", "1")


def test_a_scope_whose_listing_is_stopped_reports_no_unbound_count(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.s3.put(_A, b"unbound")
    stop = threading.Event()
    listed = fx.s3.list_object_versions

    def stopping(**kwargs: Any) -> Any:
        if kwargs["Prefix"].startswith(f"{PREFIX}/{_A}/"):
            stop.set()
        return listed(**kwargs)

    monkeypatch.setattr(fx.s3, "list_object_versions", stopping)

    result = fx.sweep(stop=stop)

    (scope,) = [s for s in result.scopes if s.scope == _A]
    assert not scope.complete and scope.unbound == 0 and scope.gap


def test_a_sweep_that_cannot_find_the_scopes_still_reports(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.marked.mark.add(_A, content_digest(b"bound"))
    fx.s3.put(_B, b"unbound")
    listed = fx.s3.list_object_versions

    def failing(**kwargs: Any) -> Any:
        if kwargs.get("Delimiter"):
            raise PermissionError("SlowDown")
        return listed(**kwargs)

    monkeypatch.setattr(fx.s3, "list_object_versions", failing)

    result = fx.sweep()

    assert [s.scope for s in result.scopes] == [_A]
    assert "SlowDown" in result.error
    assert result.incomplete == 1 and fx.gauges()["incomplete_scopes"] == 1
    (sweep,) = fx.audit("sweep")
    assert "SlowDown" in sweep["error"]


def test_an_eligible_sweep_dates_its_gauges_and_an_ineligible_one_does_not(
    fx,
) -> None:
    before = time.time()
    fx.sweep()
    assert before <= fx.gauges()["completed_at"] <= time.time()

    fx.s3.fail.add("list_multipart_uploads")
    fx.sweep()

    assert fx.gauges() == {"eligible": 0}


class _Waits(threading.Event):
    """A stop that records each pause instead of resting, set after ``after`` pauses."""

    def __init__(self, after: int = 0) -> None:
        super().__init__()
        self.waits: list[float] = []
        self.after = after

    def wait(self, timeout: float | None = None) -> bool:
        assert timeout is not None
        self.waits.append(timeout)
        if self.after and len(self.waits) >= self.after:
            self.set()
        return self.is_set()


def test_listing_pages_rest_in_proportion_to_their_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _Fixture(duty=0.2, page=1)
    for i in range(4):
        fx.s3.put(_A, f"object-{i}".encode())
    busy = 0.02
    listed = fx.s3.list_object_versions

    def slow(**kwargs: Any) -> Any:
        end = time.thread_time() + busy
        while time.thread_time() < end:
            pass
        return listed(**kwargs)

    monkeypatch.setattr(fx.s3, "list_object_versions", slow)
    stop = _Waits()

    fx.sweep(stop=stop)

    assert len(stop.waits) >= 4
    assert all(wait >= busy * (1 - 0.2) / 0.2 for wait in stop.waits[:4])


def test_sizing_open_uploads_rests_and_stops_on_stop() -> None:
    fx = _Fixture(page=1)
    fx.s3.uploads = [
        {"Key": f"{PREFIX}/{_A}/sha256/x{i}", "UploadId": f"u-{i}"} for i in range(10)
    ]
    stop = _Waits(after=3)

    result = fx.sweep(stop=stop)

    sized = fx.s3.calls.count("list_parts")
    assert 0 < sized < 10 and len(stop.waits) >= sized
    assert result.uploads is not None and not result.uploads.known
    assert result.uploads.error == "stopped"


def test_eligibility_is_logged_only_when_it_changes(
    fx, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=_LOGGER.name)

    for _ in range(3):
        fx.collector.eligibility()
    fx.s3.fail.add("list_multipart_uploads")
    fx.collector.eligibility()
    fx.collector.eligibility()

    messages = [r.getMessage() for r in caplog.records if r.name == _LOGGER.name]
    assert len([m for m in messages if "lists the store's versions" in m]) == 1
    assert len([m for m in messages if "cannot list the store" in m]) == 1
