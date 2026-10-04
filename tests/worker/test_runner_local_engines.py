"""The worker's resident lane binds its sidecar to the engines its executors publish."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from tests.worker.factories import make_worker_hardware, no_mediated_op
from worker.lifecycle import Lifecycle
from worker.resident import HttpEngineDelivery, LocalEngine
from worker.resident.lane_host import ResidentLaneHost
from worker.runner import Runner


def test_the_resident_lane_resolves_engines_from_the_worker_registry(
    tmp_path: Path,
) -> None:
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    client = cast(MagicMock, lifecycle.client)
    client.worker_id = "wkr-1"
    client.next_mediated_op.side_effect = no_mediated_op
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={},
        default_executor=cast(Any, MagicMock()),
        logger=MagicMock(),
    )
    engine = LocalEngine("/run/engine.sock", "engine-key")
    lifecycle.local_engines.publish("tsk-serve", engine)
    host = runner._ensure_resident_host()
    assert host is not None
    try:
        assert host._lookup_local_engine is not None
        assert host._lookup_local_engine("tsk-serve") == engine
        assert lifecycle.local_engines._withdraw_listeners == [host.release_engine]
        delivery = host._engine_open
        assert isinstance(delivery, HttpEngineDelivery)
        assert delivery._engine_live == lifecycle.local_engines.serves
        lifecycle.local_engines.withdraw("tsk-serve")
        assert host._lookup_local_engine("tsk-serve") is None
    finally:
        cast(ResidentLaneHost, host).stop(1.0)
