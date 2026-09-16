import logging
from pathlib import Path

import pytest

from server.config import MetricsConfig
from server.services.metrics import MetricsRecorder
from server.services.profiling import (
    NULL_PROFILER,
    ControlPlaneStage,
    StageWindow,
    build_profiler,
)


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test-control-profiling")


def _recorder(
    tmp_path: Path, logger: logging.Logger, *, enabled: bool
) -> MetricsRecorder:
    return MetricsRecorder(tmp_path, logger, enable_control_profiling=enabled)


def test_disabled_recorder_drops_stages_and_omits_the_section(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=False)
    recorder.record_control_stage(
        ControlPlaneStage.COMPILE_LOWER,
        StageWindow.SUBMIT,
        1.0,
        workflow_id="wfl-1",
    )
    assert recorder.control_plane_breakdown() == {"workflows": {}}
    assert "v2_control_plane" not in recorder.snapshot()


def test_enabled_recorder_rolls_stages_up_per_window(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    recorder.record_control_stage(
        ControlPlaneStage.COMPILE_LOWER, StageWindow.SUBMIT, 2.0, workflow_id="wfl-1"
    )
    recorder.record_control_stage(
        ControlPlaneStage.ENGINE_BUILD, StageWindow.SUBMIT, 1.0, workflow_id="wfl-1"
    )
    recorder.record_control_stage(
        ControlPlaneStage.DISPATCH, StageWindow.QUEUE, 0.5, workflow_id="wfl-1"
    )

    submit = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"][
        "submit"
    ]
    assert submit["total_sec"] == pytest.approx(3.0)
    assert submit["stages"]["compile_lower"]["count"] == 1
    assert submit["stages"]["compile_lower"]["avg_sec"] == pytest.approx(2.0)

    queue = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"]["queue"]
    assert queue["total_sec"] == pytest.approx(0.5)


def test_repeated_stage_accumulates_sum_count_and_max(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    for seconds in (0.1, 0.4, 0.2):
        recorder.record_control_stage(
            ControlPlaneStage.DS_DRIVE,
            StageWindow.POST_START,
            seconds,
            workflow_id="wfl-1",
            invocation_id="inv-1",
        )
    entry = recorder.control_plane_breakdown()["workflows"]["wfl-1"]
    stage = entry["windows"]["post_start"]["stages"]["ds_drive"]
    assert stage["count"] == 3
    assert stage["total_sec"] == pytest.approx(0.7)
    assert stage["max_sec"] == pytest.approx(0.4)
    assert entry["invocations"] == 1


def test_nested_stage_is_reported_beside_the_window_total(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    recorder.record_control_stage(
        ControlPlaneStage.DISPATCH, StageWindow.QUEUE, 1.0, workflow_id="wfl-1"
    )
    recorder.record_control_stage(
        ControlPlaneStage.LEDGER_SNAPSHOT, StageWindow.QUEUE, 0.4, workflow_id="wfl-1"
    )
    queue = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"]["queue"]
    assert queue["total_sec"] == pytest.approx(1.0)
    assert queue["nested_sec"] == pytest.approx(0.4)


def test_same_stage_in_two_windows_stays_separate(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    recorder.record_control_stage(
        ControlPlaneStage.LEDGER_SNAPSHOT, StageWindow.SUBMIT, 0.2, workflow_id="wfl-1"
    )
    recorder.record_control_stage(
        ControlPlaneStage.LEDGER_SNAPSHOT,
        StageWindow.POST_START,
        0.3,
        workflow_id="wfl-1",
    )
    windows = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"]
    assert windows["submit"]["nested_sec"] == pytest.approx(0.2)
    assert windows["post_start"]["nested_sec"] == pytest.approx(0.3)


def test_stage_without_a_workflow_is_dropped(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    recorder.record_control_stage(
        ControlPlaneStage.RELAY, StageWindow.POST_START, 1.0, workflow_id=None
    )
    assert recorder.control_plane_breakdown() == {"workflows": {}}


def test_tracked_workflows_are_bounded(
    tmp_path: Path, logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.services.metrics._MAX_PROFILED_WORKFLOWS", 2)
    recorder = _recorder(tmp_path, logger, enabled=True)
    for index in range(3):
        recorder.record_control_stage(
            ControlPlaneStage.DISPATCH,
            StageWindow.QUEUE,
            0.1,
            workflow_id=f"wfl-{index}",
        )
    tracked = recorder.control_plane_breakdown()["workflows"]
    assert set(tracked) == {"wfl-1", "wfl-2"}


def test_profiler_records_the_elapsed_stage(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    profiler = build_profiler(recorder, enabled=True)
    with profiler.stage(
        ControlPlaneStage.COMPILE_VALIDATE, StageWindow.SUBMIT, workflow_id="wfl-1"
    ):
        pass
    stages = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"][
        "submit"
    ]["stages"]
    assert stages["compile_validate"]["count"] == 1
    assert stages["compile_validate"]["total_sec"] >= 0.0


def test_profiler_records_a_stage_that_raises(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    profiler = build_profiler(recorder, enabled=True)
    with pytest.raises(RuntimeError):
        with profiler.stage(
            ControlPlaneStage.ENGINE_BUILD, StageWindow.SUBMIT, workflow_id="wfl-1"
        ):
            raise RuntimeError("boom")
    stages = recorder.control_plane_breakdown()["workflows"]["wfl-1"]["windows"][
        "submit"
    ]["stages"]
    assert stages["engine_build"]["count"] == 1


def test_build_profiler_returns_the_null_profiler_when_disabled(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    profiler = build_profiler(recorder, enabled=False)
    assert profiler is NULL_PROFILER
    with profiler.stage(
        ControlPlaneStage.DISPATCH, StageWindow.QUEUE, workflow_id="wfl-1"
    ):
        pass
    assert recorder.control_plane_breakdown() == {"workflows": {}}


def test_config_reads_the_gate_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SERVER_METRICS_CONTROL_PROFILING", raising=False)
    assert MetricsConfig.from_env(tmp_path).enable_control_profiling is False
    monkeypatch.setenv("SERVER_METRICS_CONTROL_PROFILING", "1")
    assert MetricsConfig.from_env(tmp_path).enable_control_profiling is True


def test_enabled_snapshot_carries_the_section(
    tmp_path: Path, logger: logging.Logger
) -> None:
    recorder = _recorder(tmp_path, logger, enabled=True)
    recorder.record_control_stage(
        ControlPlaneStage.DISPATCH, StageWindow.QUEUE, 0.1, workflow_id="wfl-1"
    )
    assert "wfl-1" in recorder.snapshot()["v2_control_plane"]["workflows"]
