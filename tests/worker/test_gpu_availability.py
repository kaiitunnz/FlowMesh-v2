"""Per-device GPU availability: what the worker observes and what it reports."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.content import SharedFilesystemObjectStore
from shared.harness import ServiceLeafEpisodeDispatch
from shared.schemas.result import BaseExecutorResult
from shared.schemas.worker import WorkerStatus
from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.specs import EchoSpecStrict, InferenceSpecStrict, SSHSpecStrict
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from tests.worker.factories import (
    make_worker_hardware,
    make_worker_task_message,
    no_mediated_op,
)
from worker.executors.base_executor import ExecutionError, Executor
from worker.gpu_availability import (
    MIB,
    DeviceAvailability,
    DeviceReading,
    DeviceState,
    GpuAvailabilityMonitor,
    GpuGateConfig,
    NvmlDeviceProbe,
    decide_availability,
)
from worker.lifecycle import Lifecycle
from worker.runner import Runner
from worker.utils import nvml

THRESH = 1024
GPU_A = "GPU-aaaa"
GPU_B = "GPU-bbbb"


def _reading(used_mib: float, free_bytes: int = 0) -> DeviceReading:
    return DeviceReading(used_mib=used_mib, free_bytes=free_bytes)


def _gpu_spec() -> InferenceSpecStrict:
    return InferenceSpecStrict(
        taskType=TaskType.INFERENCE,
        data={"type": "list", "items": ["hi"]},
        model=ModelConfig(source=ModelSource(identifier="org/m")),
    )


class TestDecide:
    def _decide(
        self, used_mib: float, previous: DeviceState, consecutive: int = 2
    ) -> DeviceState:
        return decide_availability(_reading(used_mib), THRESH, consecutive, previous)

    def test_needs_consecutive_observations_to_enter(self) -> None:
        state = self._decide(40_000, DeviceState())
        assert state.availability.available is True and state.streak == 1
        assert self._decide(40_000, state).availability.available is False

    def test_needs_consecutive_observations_to_leave(self) -> None:
        held = DeviceState(DeviceAvailability(available=False))
        state = self._decide(5, held)
        assert state.availability.available is False and state.streak == 1
        assert self._decide(5, state).availability.available is True

    def test_single_spike_does_not_flip(self) -> None:
        state = self._decide(40_000, DeviceState())
        assert self._decide(5, state).streak == 0

    def test_at_threshold_is_still_available(self) -> None:
        assert self._decide(THRESH, DeviceState()).availability.available is True

    def test_consecutive_one_flips_immediately(self) -> None:
        state = self._decide(40_000, DeviceState(), consecutive=1)
        assert state.availability.available is False

    def test_free_bytes_always_come_from_the_latest_reading(self) -> None:
        # Even while latched mid-streak, the reported free figure is the fresh one.
        held = DeviceState(DeviceAvailability(available=False, free_bytes=1))
        state = decide_availability(_reading(5, free_bytes=999), THRESH, 2, held)
        assert state.availability == DeviceAvailability(available=False, free_bytes=999)


class TestNvmlDeviceProbe:
    @pytest.fixture(autouse=True)
    def _every_gpu_visible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        devices: dict[int, tuple[str, str, int, int]],
        mig: dict[tuple[int, int], tuple[str, int, int]] | None = None,
    ) -> None:
        """Fake NVML with the given devices and MIG slices (by (index, slot))."""
        slices = mig or {}

        class FakeNvml:
            NVMLError = RuntimeError

            @staticmethod
            def nvmlInit() -> None:
                return None

            @staticmethod
            def nvmlDeviceGetCount() -> int:
                return len(devices)

            @staticmethod
            def nvmlDeviceGetHandleByIndex(index: int) -> int:
                return index

            @staticmethod
            def nvmlDeviceGetName(handle: int) -> str:
                return devices[handle][0]

            @staticmethod
            def nvmlDeviceGetUUID(handle: int | tuple[int, int]) -> str:
                if isinstance(handle, tuple):
                    return slices[handle][0]
                return devices[handle][1]

            @staticmethod
            def nvmlDeviceGetMaxMigDeviceCount(handle: int) -> int:
                return max((slot + 1 for i, slot in slices if i == handle), default=0)

            @staticmethod
            def nvmlDeviceGetMigDeviceHandleByIndex(
                handle: int, slot: int
            ) -> tuple[int, int]:
                if (handle, slot) not in slices:
                    raise RuntimeError("empty slot")
                return (handle, slot)

            @staticmethod
            def nvmlDeviceGetMemoryInfo(handle: int | tuple[int, int]) -> Any:
                if isinstance(handle, tuple):
                    _, used, free = slices[handle]
                else:
                    _, _, used, free = devices[handle]
                return SimpleNamespace(used=used, free=free)

        monkeypatch.setattr("worker.gpu_availability.pynvml", FakeNvml)
        monkeypatch.setattr("worker.hw.pynvml", FakeNvml)
        monkeypatch.setattr(nvml, "pynvml", FakeNvml)

    def test_keys_by_uuid_not_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Index is only meaningful relative to CUDA_VISIBLE_DEVICES.
        self._install(monkeypatch, {0: ("dedicated", GPU_A, 40_000 * MIB, 8 * MIB)})
        assert NvmlDeviceProbe()() == {GPU_A: _reading(40_000.0, free_bytes=8 * MIB)}

    def test_unified_devices_are_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Their "used" figure is system RAM, not a card another tenant holds --
        # so they report nothing rather than reporting free.
        self._install(
            monkeypatch,
            {
                0: ("unified", GPU_A, 40_000 * MIB, 0),
                1: ("dedicated", GPU_B, 0, 48 * 1024 * MIB),
            },
        )
        readings = NvmlDeviceProbe(lambda index, _name: index == 0)()
        assert set(readings) == {GPU_B}

    def test_reads_only_the_workers_own_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NVML also lists GPUs CUDA_VISIBLE_DEVICES hides from the worker; those
        # belong to someone else and are not the worker's to report.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_B)
        self._install(
            monkeypatch,
            {
                0: ("dedicated", GPU_A, 40_000 * MIB, 8 * MIB),
                1: ("dedicated", GPU_B, 0, 48 * 1024 * MIB),
            },
        )
        seen: list[int] = []

        def is_unified(ordinal: int, _name: str) -> bool:
            seen.append(ordinal)
            return False

        readings = NvmlDeviceProbe(is_unified)()
        assert set(readings) == {GPU_B}
        # The unified-memory check takes the CUDA ordinal, not the NVML index.
        assert seen == [0]

    def test_a_mig_slice_is_read_on_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A sibling slice's tenant fills the GPU; the worker's own slice is idle.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-own")
        self._install(
            monkeypatch,
            {0: ("dedicated", GPU_A, 40_000 * MIB, 0)},
            mig={
                (0, 0): ("MIG-sibling", 40_000 * MIB, 0),
                (0, 1): ("MIG-own", 5 * MIB, 10 * 1024 * MIB),
            },
        )
        assert NvmlDeviceProbe()() == {GPU_A: _reading(5.0, free_bytes=10 * 1024 * MIB)}

    def test_nvml_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            NVMLError = RuntimeError

            @staticmethod
            def nvmlInit() -> None:
                raise RuntimeError("driver wedged")

        monkeypatch.setattr("worker.gpu_availability.pynvml", Broken)
        assert NvmlDeviceProbe()() == {}


class TestMonitor:
    def _monitor(
        self, batches: list[dict[str, DeviceReading]], **cfg: Any
    ) -> GpuAvailabilityMonitor:
        config = GpuGateConfig(
            enabled=cfg.get("enabled", True),
            threshold_mib=THRESH,
            consecutive=cfg.get("consecutive", 2),
            grace_sec=0.0,
        )
        it = iter(batches)
        return GpuAvailabilityMonitor(config, lambda: next(it))

    def test_devices_are_tracked_independently(self) -> None:
        held = {GPU_A: _reading(40_000), GPU_B: _reading(5)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].available is False
        assert snapshot[GPU_B].available is True

    def test_suppressed_observation_latches(self) -> None:
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        # Nothing is probed while suppressed, so the batch list is not consumed.
        monitor.observe(False)
        assert monitor.snapshot()[GPU_A].available is False
        assert monitor.live_snapshot() == {}

    def test_total_probe_failure_clears_rather_than_latching(self) -> None:
        # Only a fresh clear reading releases a latch, and a broken probe never
        # produces one -- latching here would exclude the worker forever.
        held = {GPU_A: _reading(40_000)}
        monitor = self._monitor([held, held, {}])
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        monitor.observe(True)
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

    def test_partial_failure_leaves_unread_devices_latched(self) -> None:
        both = {GPU_A: _reading(40_000), GPU_B: _reading(40_000)}
        monitor = self._monitor([both, both, {GPU_B: _reading(5)}])
        monitor.observe(True)
        monitor.observe(True)
        monitor.observe(True)
        snapshot = monitor.snapshot()
        assert snapshot[GPU_A].available is False, "unread device keeps its state"
        assert snapshot[GPU_B].available is False, "one clear reading is not enough"

    def test_an_unread_device_is_latched_but_not_live(self) -> None:
        # A device the probe could not read this tick may still advise the
        # dispatcher, but must not hard-refuse work nobody has confirmed since.
        both = {GPU_A: _reading(40_000), GPU_B: _reading(40_000)}
        monitor = self._monitor([both, both, {GPU_B: _reading(40_000)}])
        monitor.observe(True)
        monitor.observe(True)
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].available is False
        assert GPU_A not in monitor.live_snapshot()
        assert GPU_B in monitor.live_snapshot()

    def test_a_disabled_monitor_never_probes(self) -> None:
        # The kill switch is also enforced at construction; this is the guard that
        # survives a second construction site appearing.
        def explode() -> dict[str, DeviceReading]:
            pytest.fail("probed while disabled")

        monitor = GpuAvailabilityMonitor(GpuGateConfig(enabled=False), explode)
        monitor.observe(True)
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

    def test_free_bytes_ride_along(self) -> None:
        monitor = self._monitor([{GPU_A: _reading(5, free_bytes=1234)}])
        monitor.observe(True)
        assert monitor.snapshot()[GPU_A].free_bytes == 1234


class FakeClient:
    def __init__(self) -> None:
        self.statuses: list[tuple[WorkerStatus, dict[str, Any] | None]] = []

    def dispatch_id(self, task_id: str) -> str | None:
        return None

    def set_status(
        self,
        status: WorkerStatus,
        extra: dict | None = None,
        dispatch_id: str | None = None,
    ) -> None:
        self.statuses.append((status, extra))


def _lifecycle(
    tmp_path: Path, batches: list[dict[str, DeviceReading]], grace: float = 0.0
) -> tuple[Lifecycle, GpuAvailabilityMonitor, FakeClient]:
    client = FakeClient()
    it = iter(batches)
    monitor = GpuAvailabilityMonitor(
        GpuGateConfig(
            enabled=True, threshold_mib=THRESH, consecutive=1, grace_sec=grace
        ),
        lambda: next(it),
    )
    lc = Lifecycle(
        client,  # type: ignore[arg-type]
        30,
        120,
        tmp_path / "hb",
        cost_per_hour=0.0,
        gpu_monitor=monitor,
    )
    lc._status = WorkerStatus.IDLE  # as after start()
    lc.set_gpu_executor_probe(lambda: frozenset())
    return lc, monitor, client


class TestLifecycleIntegration:
    def test_availability_is_reported_without_touching_status(
        self, tmp_path: Path
    ) -> None:
        # The whole point of this change: the worker stays IDLE and keeps taking
        # CPU work while its GPU is reported as held.
        lc, _, client = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert client.statuses == []
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False

    def test_a_warm_gpu_executor_suppresses_the_reading(self, tmp_path: Path) -> None:
        # Reading the worker's own resident model as foreign must stay impossible.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc.set_gpu_executor_probe(lambda: None)
        lc._observe_gpu()
        assert monitor.snapshot() == {}
        assert monitor.live_snapshot() == {}

    def test_an_active_task_suppresses_the_reading(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc.set_busy("tsk-1")
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_no_registered_probe_is_unmeasurable(self, tmp_path: Path) -> None:
        # main.py registers the probe only after Runner exists; until then we
        # cannot know whether an executor is warm, so we must not measure.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._gpu_executor_probe = None
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_grace_window_suppresses_after_a_task(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}], grace=600)
        lc.set_busy("tsk-1")
        lc.set_idle("tsk-1")
        lc._observe_gpu()
        assert monitor.snapshot() == {}

    def test_a_finished_task_no_longer_wipes_availability(self, tmp_path: Path) -> None:
        # set_idle used to clear the gate, which is why admission once had to run
        # before set_busy. The latch now survives a task boundary.
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert monitor.snapshot()[GPU_A].available is False
        lc.set_busy("tsk-1")
        lc.set_idle("tsk-1")
        assert monitor.snapshot()[GPU_A].available is False

    def test_live_availability_hides_a_stale_latch(self, tmp_path: Path) -> None:
        # Refusing a task on a latch we cannot currently confirm would fail it
        # terminally; the advisory snapshot keeps it, the live view does not.
        lc, _, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}])
        lc._observe_gpu()
        assert lc.live_gpu_availability()[GPU_A].available is False
        lc.set_gpu_executor_probe(lambda: None)
        lc._observe_gpu()
        assert lc.live_gpu_availability() == {}
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False


class TestWarmExecutorGpuFlag:
    """The flag behind ``gpu_devices_in_use``.

    Reading GPU-ness off the executor class does not work: the default config
    wraps most executors in ``MPExecutor``, whose class carries no such
    attribute, and a transformers executor's device depends on the spec it ran.
    So the flag is recorded from each task's own spec.
    """

    def _runner(self, tmp_path: Path) -> Runner:
        return Runner(
            lifecycle=MagicMock(),
            task_stream=[],
            results_dir=tmp_path,
            hardware=MagicMock(),
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

    def _spec(self, *, gpu: bool, **message: Any) -> Any:
        if gpu:
            return make_worker_task_message(_gpu_spec(), **message)
        return make_worker_task_message(EchoSpecStrict(taskType=TaskType.ECHO))

    def test_no_executor_means_no_gpu_held(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._note_gpu_usage(self._spec(gpu=True))
        assert runner.gpu_devices_in_use() == frozenset()

    def test_a_gpu_task_marks_the_warm_executor(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        assert runner.gpu_devices_in_use() is None

    def test_a_cpu_task_alone_does_not(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=False))
        assert runner.gpu_devices_in_use() == frozenset()

    def test_a_later_cpu_task_does_not_clear_an_earlier_gpu_task(
        self, tmp_path: Path
    ) -> None:
        # The reuse hole: a CPU task following a GPU task on the same warm
        # executor must not make us forget the VRAM the GPU task allocated.
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        runner._note_gpu_usage(self._spec(gpu=False))
        assert runner.gpu_devices_in_use() is None

    def test_teardown_clears_the_flag(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True))
        runner._cleanup_active_executor()
        assert runner._active_executor_used_gpu is False
        assert runner.gpu_devices_in_use() == frozenset()

    def test_a_declared_gpu_alone_does_not_mark_the_executor(
        self, tmp_path: Path
    ) -> None:
        # A CPU executor kept warm after a task that only declared a GPU holds no GPU
        # memory; marking it would suppress every later reading for as long as it
        # stays warm.
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(
            make_worker_task_message(
                EchoSpecStrict(
                    taskType=TaskType.ECHO,
                    resources=ResourcesSpec(
                        hardware=HardwareRequirements(gpu=GPURequirements(count=2))
                    ),
                )
            )
        )
        assert runner.gpu_devices_in_use() == frozenset()

    def test_an_ssh_session_does_not_mark_the_executor(self, tmp_path: Path) -> None:
        # The session holds its devices only while it lives; the warm SSH executor
        # holds none afterwards.
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(
            make_worker_task_message(
                SSHSpecStrict(
                    taskType=TaskType.SSH,
                    resources=ResourcesSpec(
                        hardware=HardwareRequirements(gpu=GPURequirements(count=1))
                    ),
                )
            )
        )
        assert runner.gpu_devices_in_use() == frozenset()

    @pytest.mark.parametrize(
        "relay",
        [
            {"input_preparation": True},
            {"service_episode": ServiceLeafEpisodeDispatch(interface="chat")},
        ],
        ids=["input_preparation", "service_episode"],
    )
    def test_a_relaying_dispatch_does_not_mark_the_executor(
        self, tmp_path: Path, relay: dict[str, Any]
    ) -> None:
        # It loads no model, so the warm executor holds none of its memory.
        runner = self._runner(tmp_path)
        runner._active_executor = MagicMock()
        runner._note_gpu_usage(self._spec(gpu=True, **relay))
        assert runner.gpu_devices_in_use() == frozenset()


class TestAdmission:
    """``_refuse_if_gpu_is_held``: the worker's own veto on a held card."""

    def _runner(
        self,
        tmp_path: Path,
        availability: dict[str, DeviceAvailability],
        devices: int = 1,
    ) -> Runner:
        lifecycle = MagicMock()
        lifecycle.live_gpu_availability.return_value = availability
        hardware = WorkerHardware(
            cpu=CPUInfo(logical_cores=8, model="CPU"),
            memory=MemoryInfo(total_bytes=64 * 1024**3),
            gpu=GpuPlatformInfo(
                driver_version=None,
                cuda_version=None,
                devices=[
                    GpuInfo(
                        index=i,
                        name="NVIDIA RTX 6000 Ada Generation",
                        uuid=f"GPU-{i}",
                        memory_total_bytes=48 * 1024**3,
                    )
                    for i in range(devices)
                ],
            ),
            network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
        )
        return Runner(
            lifecycle=lifecycle,
            task_stream=[],
            results_dir=tmp_path,
            hardware=hardware,
            executors={},
            default_executor=MagicMock(),
            logger=MagicMock(),
        )

    def _gpu_spec(self, **message: Any) -> Any:
        return make_worker_task_message(_gpu_spec(), **message)

    def _ssh_spec(self, count: int | None) -> Any:
        return make_worker_task_message(
            SSHSpecStrict(
                taskType=TaskType.SSH,
                resources=ResourcesSpec(
                    hardware=HardwareRequirements(gpu=GPURequirements(count=count))
                ),
            )
        )

    def _held(self, *uuids: str) -> dict[str, DeviceAvailability]:
        return {u: DeviceAvailability(available=False, free_bytes=0) for u in uuids}

    def test_refuses_a_gpu_task_when_the_only_device_is_held(
        self, tmp_path: Path
    ) -> None:
        runner = self._runner(tmp_path, self._held("GPU-0"))
        with pytest.raises(ExecutionError) as excinfo:
            runner._refuse_if_gpu_is_held(self._gpu_spec())
        assert excinfo.value.retryable is True, "must reroute, not fail the task"

    def test_refuses_a_model_task_beside_a_free_device(self, tmp_path: Path) -> None:
        # The model would see every device, the held one included.
        runner = self._runner(tmp_path, self._held("GPU-0"), devices=2)
        with pytest.raises(ExecutionError) as excinfo:
            runner._refuse_if_gpu_is_held(self._gpu_spec())
        assert excinfo.value.retryable is True

    def test_admits_an_ssh_session_when_a_free_device_remains(
        self, tmp_path: Path
    ) -> None:
        runner = self._runner(tmp_path, self._held("GPU-0"), devices=2)
        runner._refuse_if_gpu_is_held(self._ssh_spec(count=1))

    def test_refuses_an_ssh_session_the_free_devices_fall_short_of(
        self, tmp_path: Path
    ) -> None:
        runner = self._runner(tmp_path, self._held("GPU-0"), devices=2)
        with pytest.raises(ExecutionError):
            runner._refuse_if_gpu_is_held(self._ssh_spec(count=2))

    def test_refuses_an_ssh_session_selecting_no_device_beside_a_free_one(
        self, tmp_path: Path
    ) -> None:
        # A gpu block naming nothing hands the session every device, the held one too.
        runner = self._runner(tmp_path, self._held("GPU-0"), devices=2)
        with pytest.raises(ExecutionError) as excinfo:
            runner._refuse_if_gpu_is_held(self._ssh_spec(count=None))
        assert excinfo.value.retryable is True

    def test_admits_a_cpu_task_onto_a_fully_held_worker(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path, self._held("GPU-0"))
        runner._refuse_if_gpu_is_held(
            make_worker_task_message(EchoSpecStrict(taskType=TaskType.ECHO))
        )

    def test_a_stale_latch_does_not_refuse(self, tmp_path: Path) -> None:
        # live_gpu_availability returns {} when the last reading was suppressed.
        runner = self._runner(tmp_path, {})
        runner._refuse_if_gpu_is_held(self._gpu_spec())

    def test_nothing_held_admits(self, tmp_path: Path) -> None:
        runner = self._runner(
            tmp_path, {"GPU-0": DeviceAvailability(available=True, free_bytes=1)}
        )
        runner._refuse_if_gpu_is_held(self._gpu_spec())

    @pytest.mark.parametrize(
        "relay",
        [
            {"input_preparation": True},
            {"service_episode": ServiceLeafEpisodeDispatch(interface="chat")},
        ],
        ids=["input_preparation", "service_episode"],
    )
    def test_admits_a_relaying_dispatch_onto_a_fully_held_worker(
        self, tmp_path: Path, relay: dict[str, Any]
    ) -> None:
        # Its spec would load a model on its own, but this dispatch loads none.
        runner = self._runner(tmp_path, self._held("GPU-0"))
        runner._refuse_if_gpu_is_held(self._gpu_spec(**relay))


class _Recording(Executor):
    name = "recording"

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.saw_gpu_executor: list[bool] = []
        self.runner: Runner | None = None

    def run(self, task: Any, out_dir: Path) -> BaseExecutorResult:
        self.ran.append(task.task_id)
        if self.runner is not None:
            self.saw_gpu_executor.append(
                self.runner.gpu_devices_in_use() != frozenset()
            )
        return BaseExecutorResult()

    def cancel(self, task_id: str) -> None:
        return None


class _Plane:
    def __init__(self, root: Path) -> None:
        self.store = SharedFilesystemObjectStore(root)

    def for_task(self, task_id: str) -> SharedFilesystemObjectStore:
        return self.store


class TestRefusalInTheTaskLoop:
    """The refusal runs where every controlled failure runs: inside the per-task
    ``try`` after ``set_busy``, so the failure is reported and ``set_idle`` follows."""

    def _run(
        self,
        tmp_path: Path,
        messages: list[Any] | None = None,
        held: bool = True,
        **message: Any,
    ) -> tuple[MagicMock, _Recording, MagicMock]:
        lifecycle = MagicMock()
        lifecycle.worker_id = "wrk-test"
        lifecycle.client.create_task_log_emitter.return_value = None
        lifecycle.client.iter_interrupts.return_value = []
        lifecycle.client.iter_stops.return_value = []
        lifecycle.client.next_mediated_op.side_effect = no_mediated_op
        lifecycle.content_plane = _Plane(tmp_path / "cas")
        lifecycle.live_gpu_availability.return_value = (
            {"GPU-0": DeviceAvailability(available=False, free_bytes=0)} if held else {}
        )
        executor = _Recording()
        runner = Runner(
            lifecycle=lifecycle,
            task_stream=messages
            or [
                make_worker_task_message(
                    _gpu_spec(),
                    task_type=TaskType.INFERENCE,
                    task_id="tsk-1",
                    **message,
                )
            ],
            results_dir=tmp_path / "out",
            hardware=make_worker_hardware(
                [GpuInfo(index=0, name="L4", uuid="GPU-0", memory_total_bytes=1 << 34)]
            ),
            executors={"vllm": executor, "service_leaf": executor, "echo": executor},
            default_executor=executor,
            logger=MagicMock(),
        )
        executor.runner = runner
        hydrator = MagicMock(wraps=runner._input_hydrator)
        runner._input_hydrator = hydrator
        runner.start()
        return lifecycle, executor, hydrator

    def test_a_gpu_dispatch_on_a_held_card_is_refused_and_the_worker_idles(
        self, tmp_path: Path
    ) -> None:
        lifecycle, executor, hydrator = self._run(tmp_path)
        assert executor.ran == []
        lifecycle.set_busy.assert_called_once_with("tsk-1")
        lifecycle.set_failed.assert_called_once()
        assert lifecycle.set_failed.call_args.kwargs["retryable"] is True
        lifecycle.set_idle.assert_called_once_with("tsk-1")
        # Refused on what the worker can see now, before it pays to read any input.
        hydrator.hydrate.assert_not_called()
        # Its start is reported as one that never ran, so it starts no serve TTL.
        assert lifecycle.notify_task_started.call_args.kwargs["executing"] is False

    def test_a_resident_service_episode_on_a_held_card_runs(
        self, tmp_path: Path
    ) -> None:
        lifecycle, executor, _ = self._run(
            tmp_path, service_episode=ServiceLeafEpisodeDispatch(interface="chat")
        )
        lifecycle.set_failed.assert_not_called()
        assert executor.ran == ["tsk-1"]

    def test_switching_executors_forgets_the_gpu_the_last_one_held(
        self, tmp_path: Path
    ) -> None:
        # The model task leaves its executor warm and GPU-bound; the CPU task after it
        # needs another executor, so the GPU-bound one is torn down first.
        lifecycle, executor, _ = self._run(
            tmp_path,
            messages=[
                make_worker_task_message(
                    _gpu_spec(), task_type=TaskType.INFERENCE, task_id="tsk-1"
                ),
                make_worker_task_message(
                    EchoSpecStrict(taskType=TaskType.ECHO),
                    task_type=TaskType.ECHO,
                    task_id="tsk-2",
                ),
            ],
            held=False,
        )
        lifecycle.set_failed.assert_not_called()
        assert executor.ran == ["tsk-1", "tsk-2"]
        assert executor.saw_gpu_executor == [True, False]


class TestClearingReachesTheServer:
    """A probe failure has to clear the server's copy, not just the worker's.

    The server latches what it was last told, so if the worker simply stopped
    mentioning a device the stale reading would sit there with nothing able to
    release it -- the worker would be held out of GPU placement indefinitely.
    """

    def test_an_empty_reading_is_still_reported(self, tmp_path: Path) -> None:
        lc, monitor, _ = _lifecycle(tmp_path, [{GPU_A: _reading(44_000)}, {}])
        lc._observe_gpu()
        assert lc._metrics()["gpu_availability"][GPU_A]["available"] is False
        lc._observe_gpu()  # probe fails
        assert lc._metrics()["gpu_availability"] == {}

    def test_a_worker_without_a_monitor_says_nothing(self, tmp_path: Path) -> None:
        # Absent key means "no opinion offered"; an empty map means "cleared".
        lc = Lifecycle(MagicMock(), 30, 120, tmp_path / "hb", cost_per_hour=0.0)
        assert "gpu_availability" not in lc._metrics()
