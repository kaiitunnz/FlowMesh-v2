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
from worker import hw
from worker import main as worker_main
from worker.executors import EXECUTOR_REGISTRY
from worker.executors.base_executor import ExecutionError, Executor
from worker.executors.mp_executor import MPExecutor
from worker.executors.transformers_executor import HFTransformersExecutor
from worker.executors.vllm_serve_executor import VLLMServeExecutor
from worker.gpu_availability import DeviceAvailability, GpuAvailabilityMonitor
from worker.gpu_binding import FreeGpus, pick_devices
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


def _host(devices: list[GpuInfo], unified: bool = False) -> WorkerHardware:
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=8, model="CPU"),
        memory=MemoryInfo(total_bytes=128 * 1024**3),
        gpu=GpuPlatformInfo(
            driver_version=None,
            cuda_version=None,
            devices=devices,
            memory_is_unified=unified,
            shared_memory_total_bytes=128 * 1024**3 if unified else None,
        ),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


class TestPickDevices:
    def test_a_count_takes_that_many_free_devices_in_order(self) -> None:
        picked = pick_devices(
            _host(_devices("A100", "A100", "A100")),
            GPURequirements(count=2),
            _free("GPU-0", "GPU-2"),
        )
        assert picked == ("GPU-0", "GPU-2")

    def test_no_count_takes_every_free_matching_device(self) -> None:
        picked = pick_devices(
            _host(_devices("A100", "L4", "A100")),
            GPURequirements(type="A100"),
            _free("GPU-0", "GPU-1", "GPU-2"),
        )
        assert picked == ("GPU-0", "GPU-2")

    def test_a_warm_binding_still_free_and_fitting_is_kept(self) -> None:
        picked = pick_devices(
            _host(_devices("A100", "A100")),
            GPURequirements(count=1),
            _free("GPU-0", "GPU-1"),
            warm=("GPU-1",),
        )
        assert picked == ("GPU-1",)

    def test_a_warm_binding_on_a_held_device_is_replaced(self) -> None:
        picked = pick_devices(
            _host(_devices("A100", "A100")),
            GPURequirements(count=1),
            _free("GPU-0"),
            warm=("GPU-1",),
        )
        assert picked == ("GPU-0",)

    def test_a_count_less_task_keeps_a_warm_binding_only_if_it_is_every_match(
        self,
    ) -> None:
        devices = _host(_devices("A100", "A100", "A100"))
        free = _free("GPU-0", "GPU-1", "GPU-2")
        everything = ("GPU-0", "GPU-1", "GPU-2")
        assert pick_devices(devices, None, free, warm=("GPU-0",)) == everything
        assert pick_devices(devices, None, free, warm=everything) == everything

    def test_a_unified_pool_covers_memory_a_device_does_not_report(self) -> None:
        gb10 = GpuInfo(index=0, name="GB10", uuid="GPU-0", memory_total_bytes=None)
        picked = pick_devices(
            _host([gb10], unified=True), GPURequirements(memory="40GB"), _free("GPU-0")
        )
        assert picked == ("GPU-0",)

    def test_only_a_fresh_reading_refuses(self) -> None:
        devices = _host(_devices("A100", "A100"))
        # The latch holds GPU-1 but the reading just taken frees it.
        assert pick_devices(
            devices, GPURequirements(count=2), _free("GPU-0", fresh=("GPU-0", "GPU-1"))
        ) == ("GPU-0", "GPU-1")
        with pytest.raises(ExecutionError) as refused:
            pick_devices(devices, GPURequirements(count=2), _free("GPU-0"))
        assert refused.value.retryable


_BINDING_EXECUTORS = {
    "default",
    "vllm",
    "vllm_lora",
    "vllm_embedding",
    "diffusers",
    "sft",
    "lora_sft",
    "ppo",
    "dpo",
    "image_classification_training",
}


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

    def test_a_type_binds_only_when_every_executor_for_it_binds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = make_worker_config()
        registry: dict[str, type[Executor] | None] = {
            "default": HFTransformersExecutor,
            "plain": _Plain,
        }
        monkeypatch.setattr(
            _Plain, "supported_task_types", frozenset({TaskType.INFERENCE})
        )
        caps = build_capabilities(
            {
                "default": MPExecutor(HFTransformersExecutor, config),
                "plain": _Plain(config),
            },
            registry=registry,
        )
        assert caps.gpu_binding_task_types == {TaskType.EMBEDDING}

    def test_only_wrapped_executors_that_run_on_their_visible_gpus_bind(self) -> None:
        config = make_worker_config()
        wrapped = {
            key: MPExecutor(cls, config)
            for key in worker_main._EXECUTORS_TO_WRAP
            if (cls := EXECUTOR_REGISTRY.get(key)) is not None
        }
        binding = {key for key, executor in wrapped.items() if executor.binds_devices}
        assert binding == _BINDING_EXECUTORS & wrapped.keys()
        assert {"data_profiling", "data_retrieval"} <= wrapped.keys()

    def test_a_cpu_executor_type_is_never_advertised(self) -> None:
        config = make_worker_config()
        executors: dict[str, Executor] = {
            key: MPExecutor(cls, config)
            for key in ("data_profiling", "data_retrieval")
            if (cls := EXECUTOR_REGISTRY.get(key)) is not None
        }
        caps = build_capabilities(executors)
        assert caps.supported_task_types
        assert caps.gpu_binding_task_types == frozenset()

    def test_a_wrapped_executor_without_the_promise_is_neither_advertised_nor_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _Plain, "supported_task_types", frozenset({TaskType.INFERENCE})
        )
        mp = MPExecutor(_Plain, make_worker_config())
        caps = build_capabilities({"plain": mp}, registry={"plain": _Plain})
        mp.bind_devices(("GPU-0",))

        assert caps.gpu_binding_task_types == frozenset()
        assert mp._devices is None

    def test_a_mig_slice_binds_nothing(self) -> None:
        config = make_worker_config()
        caps = build_capabilities(
            {"default": MPExecutor(HFTransformersExecutor, config)}, binds_gpus=False
        )
        assert caps.gpu_binding_task_types == frozenset()


class _Binding(Executor):
    @property
    def binds_devices(self) -> bool:
        return True

    def __init__(self) -> None:
        self.devices: tuple[str, ...] | None = None
        self.restarts: list[tuple[str, ...] | None] = []

    def bind_devices(self, devices: tuple[str, ...] | None) -> None:
        if devices != self.devices:
            self.devices = devices
            self.restarts.append(devices)

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

        assert executor.restarts == [("GPU-1",)]
        assert runner.gpu_devices_in_use() == {"GPU-1"}

    def test_a_dispatch_needing_more_devices_rebinds(self, tmp_path: Path) -> None:
        runner, executor, _ = self._runner(tmp_path, held=("GPU-0",))
        self._dispatch(runner, _inference(1))
        self._dispatch(runner, _inference(2))

        assert executor.restarts == [("GPU-1",), ("GPU-1", "GPU-2")]

    def test_a_task_naming_its_devices_sees_every_device(self, tmp_path: Path) -> None:
        runner, executor, _ = self._runner(tmp_path)
        self._dispatch(runner, _inference())
        self._dispatch(runner, _inference(env_vars={"CUDA_VISIBLE_DEVICES": "2"}))

        assert executor.restarts == [("GPU-0",), None]
        assert runner.gpu_devices_in_use() is None

    def test_an_unbound_dispatch_after_a_cleanup_restarts_on_every_device(
        self, tmp_path: Path
    ) -> None:
        runner, executor, _ = self._runner(tmp_path)
        self._dispatch(runner, _inference())
        runner._active_executor_devices = None  # as the runner's cleanup leaves it
        self._dispatch(runner, _inference(env_vars={"CUDA_VISIBLE_DEVICES": "2"}))

        assert executor.restarts == [("GPU-0",), None]
        assert executor.devices is None

    def test_a_worker_that_does_not_bind_the_type_leaves_every_device(
        self, tmp_path: Path
    ) -> None:
        runner, executor, _ = self._runner(tmp_path, binds=False)
        self._dispatch(runner, _inference())

        assert executor.restarts == []
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


class TestFreeGpusRead:
    """The one place a tri-state availability report becomes a positive set."""

    def _read(self, reported: dict[str, DeviceAvailability], n: int) -> frozenset[str]:
        readings = MagicMock()
        readings.gpu_availability.return_value = reported
        readings.live_gpu_availability.return_value = {}
        free = FreeGpus.read(readings, _devices(*["A100"] * n))
        assert free.fresh == frozenset(f"GPU-{i}" for i in range(n))
        return free.latched

    def test_no_reading_at_all_withholds_nothing(self) -> None:
        assert self._read({}, 2) == frozenset({"GPU-0", "GPU-1"})

    def test_a_held_device_is_withheld(self) -> None:
        reported = {
            "GPU-0": DeviceAvailability(available=False, free_bytes=0),
            "GPU-1": DeviceAvailability(available=True, free_bytes=1),
        }
        assert self._read(reported, 2) == frozenset({"GPU-1"})

    def test_a_device_the_reading_did_not_cover_is_still_offered(self) -> None:
        # A partial probe must not quietly shrink the session's device set.
        reported = {"GPU-0": DeviceAvailability(available=False, free_bytes=0)}
        assert self._read(reported, 4) == frozenset({"GPU-1", "GPU-2", "GPU-3"})

    def test_every_device_held_yields_an_empty_set(self) -> None:
        reported = {
            f"GPU-{i}": DeviceAvailability(available=False, free_bytes=0)
            for i in range(2)
        }
        assert self._read(reported, 2) == frozenset()


@pytest.mark.parametrize(
    ("mig", "ambiguous", "splits"),
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_a_worker_binds_only_gpus_it_can_name_exactly(
    monkeypatch: pytest.MonkeyPatch, mig: bool, ambiguous: bool, splits: bool
) -> None:
    gpu = hw.VisibleGpu(
        ordinal=0, nvml_index=0, uuid="GPU-0", name="H100", mig_slot=1 if mig else None
    )
    monkeypatch.setattr(worker_main, "visible_gpus", lambda: (gpu,))
    monkeypatch.setattr(worker_main, "positions_may_name_other_gpus", lambda: ambiguous)

    assert worker_main._gpus_split_safely() is splits
