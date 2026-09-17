"""The worker GPU sampler: own nvmlInit(), never nvmlShutdown, silent CPU degrade."""

import inspect
import time

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource

from worker import gpu_sampler
from worker.gpu_sampler import build_gpu_sampler


class _FakeNvmlError(Exception):
    pass


class _FakePynvml:
    NVMLError = _FakeNvmlError
    NVML_TEMPERATURE_GPU = 0

    @staticmethod
    def nvmlInit() -> None:
        return None

    @staticmethod
    def nvmlDeviceGetCount() -> int:
        return 1

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    @staticmethod
    def nvmlDeviceGetUUID(handle: int) -> bytes:
        return b"GPU-0"

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle: int):
        class _Util:
            gpu = 42

        return _Util()

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle: int):
        class _Mem:
            used = 1024

        return _Mem()

    @staticmethod
    def nvmlDeviceGetPowerUsage(handle: int) -> int:
        return 50_000

    @staticmethod
    def nvmlDeviceGetTemperature(handle: int, sensor: int) -> int:
        return 65


class _CpuOnlyPynvml:
    NVMLError = _FakeNvmlError

    @staticmethod
    def nvmlInit() -> None:
        raise _FakeNvmlError("no GPU")


def _meter_with_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(resource=Resource.create({}), metric_readers=[reader])
    return reader, provider.get_meter("test")


def _datapoints(reader: InMemoryMetricReader) -> dict[str, list[tuple[float, dict]]]:
    out: dict[str, list[tuple[float, dict]]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out.setdefault(metric.name, [])
                for dp in metric.data.data_points:
                    out[metric.name].append((dp.value, dict(dp.attributes)))
    return out


def test_nvml_shutdown_is_never_called_anywhere_in_the_module():
    source = inspect.getsource(gpu_sampler)
    assert "nvmlShutdown(" not in source


def test_sample_once_emits_gauges_with_node_and_worker_id(monkeypatch):
    monkeypatch.setattr(gpu_sampler, "pynvml", _FakePynvml)
    reader, meter = _meter_with_reader()
    sampler = build_gpu_sampler(
        meter,
        node_id=lambda: "nde-1",
        worker_id=lambda: "wkr-1",
        interval_sec=1.0,
        enabled=True,
    )

    sampler._sample_once()

    points = _datapoints(reader)
    util_value, util_attrs = points["flowmesh.gpu.utilization_ratio"][0]
    assert util_value == 0.42
    assert util_attrs["flowmesh.node_id"] == "nde-1"
    assert util_attrs["flowmesh.worker_id"] == "wkr-1"
    assert util_attrs["flowmesh.gpu.index"] == "0"

    assert points["flowmesh.gpu.memory_used_bytes"][0][0] == 1024
    assert points["flowmesh.gpu.power_watts"][0][0] == 50.0
    assert points["flowmesh.gpu.temperature_celsius"][0][0] == 65.0


def test_cpu_only_worker_degrades_silently(monkeypatch, caplog):
    monkeypatch.setattr(gpu_sampler, "pynvml", _CpuOnlyPynvml)
    reader, meter = _meter_with_reader()
    sampler = build_gpu_sampler(
        meter,
        node_id=lambda: "nde-1",
        worker_id=lambda: "wkr-1",
        interval_sec=1.0,
        enabled=True,
    )

    with caplog.at_level("WARNING"):
        sampler._sample_once()
        sampler._sample_once()

    assert _datapoints(reader) == {}
    assert caplog.records == []


def test_disabled_sampler_never_starts_a_thread():
    _, meter = _meter_with_reader()
    sampler = build_gpu_sampler(
        meter,
        node_id=lambda: None,
        worker_id=lambda: None,
        interval_sec=1.0,
        enabled=False,
    )

    sampler.start()

    assert sampler.is_running is False


def test_enabled_sampler_starts_a_thread_and_shuts_down_without_shutdown_call(
    monkeypatch,
):
    monkeypatch.setattr(gpu_sampler, "pynvml", _FakePynvml)
    _, meter = _meter_with_reader()
    sampler = build_gpu_sampler(
        meter,
        node_id=lambda: "nde-1",
        worker_id=lambda: "wkr-1",
        interval_sec=0.01,
        enabled=True,
    )

    sampler.start()
    time.sleep(0.05)
    assert sampler.is_running is True

    sampler.shutdown()
    assert sampler.is_running is False
