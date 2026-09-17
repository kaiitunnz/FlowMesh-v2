"""Periodic worker-side sampling of GPU utilization, memory, power and temperature.

Runs on its own daemon thread, independent of ``collect_hw``'s one-shot hardware
probe (``hw.py``). NVML's init is refcounted, and the repo already calls
``nvmlInit()`` from two independent, uncoordinated sites — ``collect_hw`` and
``power.py``'s ``_ensure_nvml`` — with no shared init and no ``nvmlShutdown``
anywhere. This sampler follows the same pattern: it calls ``nvmlInit()`` itself and
never calls ``nvmlShutdown``, because pairing init/shutdown around a sampling pass
would tear NVML down underneath whichever of those other two call sites happens to be
mid-read on another thread — an intermittent, hardware-dependent failure with no
obvious link to telemetry.
"""

import logging
import threading
from collections.abc import Callable

import pynvml
from opentelemetry.metrics import Meter

from shared.telemetry.semconv import RESOURCE_NODE_ID, RESOURCE_WORKER_ID

__all__ = ["GpuSampler", "build_gpu_sampler"]

logger = logging.getLogger(__name__)

_GPU_INDEX_ATTR = "flowmesh.gpu.index"
_GPU_UUID_ATTR = "flowmesh.gpu.uuid"


class GpuSampler:
    """Periodic thread publishing GPU utilization/memory/power/temperature gauges.

    Degrades silently when NVML is unavailable or no GPU is present, matching the
    ``except pynvml.NVMLError: pass`` discipline ``collect_hw`` already uses — a
    CPU-only worker samples nothing and logs nothing alarming on every interval.
    """

    def __init__(
        self,
        meter: Meter,
        *,
        node_id: Callable[[], str | None],
        worker_id: Callable[[], str | None],
        interval_sec: float,
        enabled: bool,
    ) -> None:
        self._node_id = node_id
        self._worker_id = worker_id
        self._interval_sec = interval_sec
        self._enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_ready = False

        self._utilization = meter.create_gauge(
            "flowmesh.gpu.utilization_ratio",
            unit="1",
            description="GPU compute utilization, 0-1.",
        )
        self._memory_used = meter.create_gauge(
            "flowmesh.gpu.memory_used_bytes",
            unit="By",
            description="GPU memory in use.",
        )
        self._power = meter.create_gauge(
            "flowmesh.gpu.power_watts",
            unit="W",
            description="GPU power draw.",
        )
        self._temperature = meter.create_gauge(
            "flowmesh.gpu.temperature_celsius",
            unit="Cel",
            description="GPU die temperature.",
        )

    @property
    def is_running(self) -> bool:
        """Whether the sampling thread is active — the ``off`` gate asserts False."""
        return self._thread is not None

    def start(self) -> None:
        """Start the sampling thread. A no-op when disabled or already running."""
        if not self._enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="GpuMetricsSampler", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the sampling thread. Never touches NVML's init/shutdown state."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_sec + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval_sec)

    def _sample_once(self) -> None:
        try:
            if not self._nvml_ready:
                pynvml.nvmlInit()
                self._nvml_ready = True
            self._emit()
        except pynvml.NVMLError:
            self._nvml_ready = False

    def _emit(self) -> None:
        base_attrs = {
            RESOURCE_NODE_ID: self._node_id() or "",
            RESOURCE_WORKER_ID: self._worker_id() or "",
        }
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            attrs = dict(base_attrs)
            attrs[_GPU_INDEX_ATTR] = str(idx)
            try:
                uuid_raw = pynvml.nvmlDeviceGetUUID(handle)
                attrs[_GPU_UUID_ATTR] = (
                    uuid_raw.decode() if isinstance(uuid_raw, bytes) else uuid_raw
                )
            except pynvml.NVMLError:
                pass

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._utilization.set(util.gpu / 100.0, attrs)
            except pynvml.NVMLError:
                pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._memory_used.set(int(mem.used), attrs)
            except pynvml.NVMLError:
                pass

            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                self._power.set(power_mw / 1000.0, attrs)
            except pynvml.NVMLError:
                pass

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                self._temperature.set(float(temp), attrs)
            except pynvml.NVMLError:
                pass


def build_gpu_sampler(
    meter: Meter,
    *,
    node_id: Callable[[], str | None],
    worker_id: Callable[[], str | None],
    interval_sec: float,
    enabled: bool,
) -> GpuSampler:
    """Build the GPU sampler bound to the worker's meter and identity getters.

    ``node_id`` / ``worker_id`` are getters rather than plain values because both are
    resolved only after the supervisor registration handshake completes, after the
    sampler itself would typically be constructed.
    """
    return GpuSampler(
        meter,
        node_id=node_id,
        worker_id=worker_id,
        interval_sec=interval_sec,
        enabled=enabled,
    )
