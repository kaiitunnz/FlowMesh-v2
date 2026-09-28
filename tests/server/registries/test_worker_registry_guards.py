"""Membership guards on the server WorkerRegistry write helpers.

A write to a worker that is no longer in ``WORKERS_SET_KEY`` (e.g. one the
watchdog reaped) must not recreate a partial record outside the set. The guard
and the write are one atomic Redis script, so the registry never reads
membership separately.
"""

from typing import Any, cast
from unittest.mock import MagicMock

from server.registries.worker import ReportOutcome, StatusReport, WorkerRegistry
from shared.schemas.worker import WorkerStatus


def _registry(wrote: int) -> WorkerRegistry:
    rds: Any = MagicMock()
    rds.sync.eval.return_value = wrote
    return WorkerRegistry(cast(Any, rds))


def _reporting(reply: Any) -> Any:
    rds: Any = MagicMock()
    rds.sync.eval.return_value = reply
    return WorkerRegistry(cast(Any, rds))


def test_a_report_from_an_unregistered_worker_is_unknown() -> None:
    registry = _reporting([0])
    unknown = StatusReport(ReportOutcome.UNKNOWN)
    assert registry.update_worker_hb("wkr-1", "ts", 120) == unknown
    assert registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None) == unknown


def test_a_fenced_report_names_the_reserved_dispatch() -> None:
    registry = _reporting([2, b"dsp-2", b"tsk-2"])
    report = registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", None, "dsp-1")
    assert report == StatusReport(ReportOutcome.FENCED, "tsk-2", "dsp-2")


def test_a_reservation_of_an_unregistered_worker_announces_nothing() -> None:
    registry: Any = _registry(wrote=0)
    assert registry.reserve_worker("wkr-1", "tsk-1", "dsp-1") is False
    assert registry.release_worker("wkr-1", "dsp-1") is False
    # A skipped status write must not announce a status it never stored.
    registry._rds.sync.publish_telemetry.assert_not_called()


def test_a_reservation_announces_the_status_it_stored() -> None:
    registry: Any = _registry(wrote=1)
    assert registry.reserve_worker("wkr-1", "tsk-1", "dsp-1") is True
    assert registry.release_worker("wkr-1", "dsp-1") is True
    assert registry._rds.sync.publish_telemetry.call_count == 2


def test_writes_are_a_single_atomic_call() -> None:
    registry: Any = _reporting([1])
    registry.update_worker_hb("wkr-1", "ts", 120, WorkerStatus.IDLE, "dsp-1")
    # No separate membership read, and no pipeline that could interleave a reap.
    registry._rds.sync.sismember.assert_not_called()
    registry._rds.sync.hget.assert_not_called()
    registry._rds.sync.control_pipeline.assert_not_called()


def test_status_extras_are_prefixed_in_the_script_arguments() -> None:
    registry: Any = _reporting([1])
    registry.set_worker_status("wkr-1", WorkerStatus.IDLE, "ts", {"gpu": 2})
    args = registry._rds.sync.eval.call_args.args
    # numkeys, the three keys, then worker_id followed by the report and its fields.
    assert args[1] == 3
    assert args[5] == "wkr-1"
    assert "extra_gpu" in args
    assert args[args.index("extra_gpu") + 1] == "2"


def test_cache_write_is_guarded_by_membership() -> None:
    registry: Any = _registry(wrote=0)
    registry._rds.sync.exists.return_value = True
    registry._rds.sync.hash_mget.return_value = [None, None]
    registry.record_worker_cache("wkr-1", models=["org/model"])
    args = registry._rds.sync.eval.call_args.args
    assert args[4] == "wkr-1"
    assert "cache_models_json" in args
    registry._rds.sync.hash_set.assert_not_called()


def test_a_reap_checks_staleness_and_deletes_in_one_call() -> None:
    registry: Any = _registry(wrote=1)

    assert registry.reap_stale_worker("wkr-1") is True
    args = registry._rds.sync.eval.call_args.args
    assert args[1] == 3
    assert args[-1] == "wkr-1"
    registry._rds.sync.ttl.assert_not_called()
    registry._rds.sync.control_pipeline.assert_not_called()
