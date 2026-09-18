"""Periodic root-event-loop sampling of resident fleet and residency gauges.

Publishes replica count, admission slots in use, and credit-bearing claim count — one
observation per service family, so a series joins a span on
``flowmesh.physical.service_family`` — plus the ready-queue depth. All of it runs as
an ``asyncio`` task on the root server's own event loop rather than a thread:
``ResidentCapacityControl``'s stores are guarded by an ``asyncio.Lock`` over plain
dicts, which excludes another coroutine on this loop but not another OS thread, so a
thread sampler reading them concurrently could raise ``RuntimeError: dictionary
changed size during iteration``. On the loop no lock is needed at all — this sampler
observes the stores under the same single-threaded discipline as every mutator.
"""

import asyncio
import logging
from collections.abc import Callable

from opentelemetry.metrics import Meter

from shared.telemetry.semconv import PHYSICAL_SERVICE_FAMILY, RESOURCE_NODE_ID

from ..resident.capacity import outstanding_slots
from ..resident.stores import ResidentStores
from ..task.runtime import TaskRuntime

__all__ = ["FleetResidencySampler", "build_fleet_sampler"]

logger = logging.getLogger(__name__)


class FleetResidencySampler:
    """Periodic ``asyncio`` task publishing fleet/residency gauges on the root loop.

    Every gauge is a store read plus a pure derivation (``outstanding_slots`` never
    touches the ``ServiceClaim`` authority), so sampling can never itself grant or
    release admission credit.
    """

    def __init__(
        self,
        meter: Meter,
        *,
        stores: ResidentStores | None,
        runtime: TaskRuntime,
        node_id: Callable[[], str | None],
        interval_sec: float,
        enabled: bool,
    ) -> None:
        self._stores = stores
        self._runtime = runtime
        self._node_id = node_id
        self._interval_sec = interval_sec
        self._enabled = enabled
        self._task: asyncio.Task[None] | None = None

        self._replica_count = meter.create_gauge(
            "flowmesh.resident.replica_count",
            unit="{replica}",
            description="Live replica incarnations, per service family.",
        )
        self._admission_slots = meter.create_gauge(
            "flowmesh.resident.admission_slots_in_use",
            unit="{slot}",
            description=(
                "Admission slots held by credit-bearing claims, per service family."
            ),
        )
        self._claim_credit = meter.create_gauge(
            "flowmesh.resident.claim_credit_held",
            unit="{claim}",
            description="Credit-bearing claims, per service family.",
        )
        self._queue_depth = meter.create_gauge(
            "flowmesh.resident.queue_depth",
            unit="{task}",
            description="Tasks waiting in the ready queue.",
        )

    @property
    def is_running(self) -> bool:
        """Whether the sampling task is active — the ``off`` gate asserts False."""
        return self._task is not None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the periodic sampling task on ``loop``.

        A no-op when disabled or already running, so a caller can invoke this
        unconditionally at startup without checking the telemetry config itself.
        """
        if not self._enabled or self._task is not None:
            return
        target_loop = loop if loop is not None else asyncio.get_event_loop()
        self._task = target_loop.create_task(self._run())

    def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        try:
            while True:
                try:
                    self._sample_once()
                except Exception:
                    # A sampling failure costs one interval's points; letting it end the
                    # task would stop the series with nothing said.
                    logger.exception("Fleet metric sampling failed")
                await asyncio.sleep(self._interval_sec)
        except asyncio.CancelledError:
            return

    def _sample_once(self) -> None:
        resource_attrs = {RESOURCE_NODE_ID: self._node_id() or ""}
        if self._stores is not None:
            for family in self._stores.families.all():
                attrs = dict(resource_attrs)
                attrs[PHYSICAL_SERVICE_FAMILY] = family.family
                replicas = self._stores.directory.live_by_family(family.family)
                self._replica_count.set(len(replicas), attrs)

                claims = [
                    claim
                    for replica in replicas
                    for claim in self._stores.claims.credit_bearing_for_replica(
                        replica.replica_id
                    )
                ]
                self._admission_slots.set(outstanding_slots(claims), attrs)
                self._claim_credit.set(len(claims), attrs)

        self._queue_depth.set(self._runtime.ready_queue_length(), resource_attrs)


def build_fleet_sampler(
    meter: Meter,
    *,
    stores: ResidentStores | None,
    runtime: TaskRuntime,
    node_id: Callable[[], str | None],
    interval_sec: float,
    enabled: bool,
) -> FleetResidencySampler:
    """Build the fleet/residency sampler for the root server.

    ``stores`` is ``None`` when resident-capacity control is disabled; the sampler
    then reports queue depth only. The caller starts it on the root event loop once
    it is running (``sampler.start(loop)``) and stops it at shutdown
    (``sampler.shutdown()``), the same lifecycle ``ResidentCapacityControl`` uses.
    """
    return FleetResidencySampler(
        meter,
        stores=stores,
        runtime=runtime,
        node_id=node_id,
        interval_sec=interval_sec,
        enabled=enabled,
    )
