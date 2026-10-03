"""A binding executor runs on the free GPUs its worker picks, and only there."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.executors.base_executor import ExecutionError, Executor
from worker.executors.mp_executor import MPExecutor
from worker.executors.ssh_session.config import FreeGpus
from worker.executors.transformers_executor import HFTransformersExecutor
from worker.executors.vllm_serve_executor import VLLMServeExecutor
from worker.gpu_availability import DeviceAvailability, GpuAvailabilityMonitor
from worker.gpu_binding import pick_devices
from worker.main import build_capabilities
from worker.runner import Runner

_HELD = DeviceAvailability(available=False, free_bytes=0)


def _devices(*names: str) -> list[GpuInfo]:
    return [
        GpuInfo(index=i, name=name, uuid=f"GPU-{i}", memory_total_bytes=80 * 1024**3)
        for i, name in enumerate(names)
    ]


def _free(*uuids: str, fresh: tuple[str, ...] | None = None) -> FreeGpus:
    return FreeGpus(latched=frozenset(uuids), fresh=frozenset(fresh or uuids))


class TestPickDevices:
    def test_a_count_takes_that_many_free_devices_in_order(self) -> None:
        picked = pick_devices(
            _devices("A100", "A100", "A100"),
            GPURequirements(count=2),
            _free("GPU-0", "GPU-2"),
        )
        assert picked == ("GPU-0", "GPU-2")

    def test_no_count_takes_every_free_matching_device(self) -> None:
        picked = pick_devices(
            _devices("A100", "L4", "A100"),
            GPURequirements(type="A100"),
            _free("GPU-0", "GPU-1", "GPU-2"),
        )
        assert picked == ("GPU-0", "GPU-2")

    def test_a_warm_binding_still_free_and_fitting_is_kept(self) -> None:
        picked = pick_devices(
            _devices("A100", "A100"),
            GPURequirements(count=1),
            _free("GPU-0", "GPU-1"),
            warm=("GPU-1",),
        )
        assert picked == ("GPU-1",)

    def test_a_warm_binding_on_a_held_device_is_replaced(self) -> None:
        picked = pick_devices(
            _devices("A100", "A100"),
            GPURequirements(count=1),
            _free("GPU-0"),
            warm=("GPU-1",),
        )
        assert picked == ("GPU-0",)

    def test_only_a_fresh_reading_refuses(self) -> None:
        devices = _devices("A100", "A100")
        # The latch holds GPU-1 but the reading just taken frees it.
        assert pick_devices(
            devices, GPURequirements(count=2), _free("GPU-0", fresh=("GPU-0", "GPU-1"))
        ) == ("GPU-0", "GPU-1")
        with pytest.raises(ExecutionError) as refused:
            pick_devices(devices, GPURequirements(count=2), _free("GPU-0"))
        assert refused.value.retryable


class _Plain(Executor):
    def run(self, task: Any, out_dir: Path) -> Any:
        raise NotImplementedError


class TestCapabilities:
    def test_wrapped_executors_and_vllm_serve_bind(self) -> None:
        config = make_worker_config()
        executors: dict[str, Executor] = {
            "default": MPExecutor(HFTransformersExecutor, config),
            "vllm_serve": VLLMServeExecutor(config),
        }
        caps = build_capabilities(executors)
        assert caps.gpu_binding_task_types == {
            TaskType.INFERENCE,
            TaskType.EMBEDDING,
            TaskType.SERVE,
        }

    def test_an_in_process_executor_does_not_bind(self) -> None:
        # As with WORKER_ENABLE_MP_EXECUTORS=false: the executor runs in the worker.
        config = make_worker_config()
        caps = build_capabilities({"default": HFTransformersExecutor(config)})
        assert caps.gpu_binding_task_types == frozenset()

    def test_a_type_binds_only_when_every_executor_for_it_binds(self) -> None:
        config = make_worker_config()
        registry: dict[str, type[Executor] | None] = {
            "default": HFTransformersExecutor,
            "plain": _Plain,
        }
        _Plain.supported_task_types = frozenset({TaskType.INFERENCE})
        try:
            caps = build_capabilities(
                {
                    "default": MPExecutor(HFTransformersExecutor, config),
                    "plain": _Plain(config),
                },
                registry=registry,
            )
        finally:
            _Plain.supported_task_types = frozenset()
        assert caps.gpu_binding_task_types == {TaskType.EMBEDDING}

    def test_a_mig_slice_binds_nothing(self) -> None:
        config = make_worker_config()
        caps = build_capabilities(
            {"default": MPExecutor(HFTransformersExecutor, config)}, binds_gpus=False
        )
        assert caps.gpu_binding_task_types == frozenset()


class _Binding(Executor):
    binds_devices = True

    def __init__(self) -> None:
        self.bound: list[tuple[str, ...] | None] = []

    def bind_devices(self, devices: tuple[str, ...] | None) -> None:
        self.bound.append(devices)

    def run(self, task: Any, out_dir: Path) -> Any:
        raise NotImplementedError


def _hardware(count: int) -> WorkerHardware:
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=8, model="CPU"),
        memory=MemoryInfo(total_bytes=64 * 1024**3),
        gpu=GpuPlatformInfo(
            driver_version=None, cuda_version=None, devices=_devices(*["A100"] * count)
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


def _inference(gpus: int | None = 1, **vllm: Any) -> Any:
    return make_worker_task_message(
        InferenceSpecStrict(
            taskType=TaskType.INFERENCE,
            data={"type": "list", "items": ["hi"]},
            model=ModelConfig(
                source=ModelSource(identifier="org/m"), vllm=vllm or None
            ),
            resources=ResourcesSpec(
                hardware=HardwareRequirements(gpu=GPURequirements(count=gpus))
            ),
        )
    )


class TestRunnerBinding:
    def _runner(
        self, tmp_path: Path, held: tuple[str, ...] = (), binds: bool = True
    ) -> tuple[Runner, _Binding, MagicMock]:
        lifecycle = MagicMock()
        availability = {uuid: _HELD for uuid in held}
        lifecycle.gpu_availability.return_value = availability
        lifecycle.live_gpu_availability.return_value = availability
        executor = _Binding()
        runner = Runner(
            lifecycle=lifecycle,
            task_stream=[],
            results_dir=tmp_path,
            hardware=_hardware(3),
            executors={"vllm": executor},
            default_executor=executor,
            logger=MagicMock(),
            gpu_binding_task_types=(
                frozenset({TaskType.INFERENCE}) if binds else frozenset()
            ),
        )
        runner._active_executor = executor
        return runner, executor, lifecycle

    def _dispatch(self, runner: Runner, msg: Any) -> None:
        runner._note_gpu_usage(msg)
        runner._bind_active_executor(msg)

    def test_a_dispatch_binds_free_devices_and_reuses_them_warm(
        self, tmp_path: Path
    ) -> None:
        runner, executor, lifecycle = self._runner(tmp_path, held=("GPU-0",))
        self._dispatch(runner, _inference())
        # GPU-0 frees up; the warm binding still fits, so nothing restarts.
        lifecycle.gpu_availability.return_value = {}
        lifecycle.live_gpu_availability.return_value = {}
        self._dispatch(runner, _inference())

        assert executor.bound == [("GPU-1",)]
        assert runner.gpu_devices_in_use() == {"GPU-1"}

    def test_a_dispatch_needing_more_devices_rebinds(self, tmp_path: Path) -> None:
        runner, executor, _ = self._runner(tmp_path, held=("GPU-0",))
        self._dispatch(runner, _inference(1))
        self._dispatch(runner, _inference(2))

        assert executor.bound == [("GPU-1",), ("GPU-1", "GPU-2")]

    def test_a_task_naming_its_devices_sees_every_device(self, tmp_path: Path) -> None:
        runner, executor, _ = self._runner(tmp_path)
        self._dispatch(runner, _inference())
        self._dispatch(runner, _inference(env_vars={"CUDA_VISIBLE_DEVICES": "2"}))

        assert executor.bound == [("GPU-0",), None]
        assert runner.gpu_devices_in_use() is None

    def test_a_worker_that_does_not_bind_the_type_leaves_every_device(
        self, tmp_path: Path
    ) -> None:
        runner, executor, _ = self._runner(tmp_path, binds=False)
        self._dispatch(runner, _inference())

        assert executor.bound == []
        assert runner.gpu_devices_in_use() is None

    def test_a_binding_refuses_when_too_few_devices_are_free(
        self, tmp_path: Path
    ) -> None:
        runner, _, _ = self._runner(tmp_path, held=("GPU-0", "GPU-1"))
        with pytest.raises(ExecutionError) as refused:
            self._dispatch(runner, _inference(2))
        assert refused.value.retryable


class TestMeasurementBesideABoundExecutor:
    def test_only_the_bound_devices_keep_their_latch(self) -> None:
        readings = {
            "GPU-0": {"used_mib": 40_000.0},
            "GPU-1": {"used_mib": 40_000.0},
        }
        monitor = GpuAvailabilityMonitor(
            MagicMock(enabled=True, threshold_mib=1024, consecutive=1, grace_sec=0),
            lambda: {
                uuid: MagicMock(used_mib=r["used_mib"], free_bytes=0)
                for uuid, r in readings.items()
            },
        )

        monitor.observe(True, skip=frozenset({"GPU-0"}))

        assert set(monitor.live_snapshot()) == {"GPU-1"}
        assert monitor.live_snapshot()["GPU-1"].available is False
        assert "GPU-0" not in monitor.snapshot()
