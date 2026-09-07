"""Runs the worker's resident lanes on a dedicated asyncio loop.

The worker runtime is thread-based, but the resident origin driver and replica sidecar
are asyncio. This host owns one event loop on its own thread, builds the two lanes on
it, and marshals every control frame the interrupt-monitor thread delivers onto the
loop. Produced frames leave the worker as ``RESIDENT_FRAME`` events over the
authenticated attachment; the supervisor relays them opaquely to the peer worker.
"""

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from shared.network.relay_frame import RelayDirection, RelayFrame
from shared.outcome import FabricContentStore
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.reports import ResidentBootstrapAck, ResidentOpOutcome

from .engine import EngineOpen, HttpEngineDelivery
from .origin_driver import ResidentOriginDriver, ResidentOriginRequest
from .replica_sidecar import ResidentReplicaSidecar
from .transport import ResidentFrameSink

# Peeks the worker-private raw request for a captured resident boundary, or None.
RequestLookup = Callable[[str, str], str | None]
AckSink = Callable[[ResidentBootstrapAck], None]
OutcomeSink = Callable[[ResidentOpOutcome], None]


class _EventFrameSink:
    """Sends each produced relay frame up as a ``RESIDENT_FRAME`` attachment event."""

    def __init__(self, push_frame: Callable[[dict[str, Any]], None]) -> None:
        self._push_frame = push_frame

    async def send(self, frame: RelayFrame) -> None:
        self._push_frame(frame.to_wire())


class ResidentLaneHost:
    """Hosts the origin and replica resident lanes on one worker loop."""

    def __init__(
        self,
        *,
        push_frame: Callable[[dict[str, Any]], None],
        report_ack: AckSink,
        report_outcome: OutcomeSink,
        content_store: FabricContentStore | None,
        peek_request: RequestLookup,
        engine_open: EngineOpen | None = None,
        engine_timeout_sec: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._push_frame = push_frame
        self._report_ack = report_ack
        self._report_outcome = report_outcome
        self._content_store = content_store
        self._peek_request = peek_request
        self._engine_open = engine_open or HttpEngineDelivery(
            timeout_sec=engine_timeout_sec
        )
        self._logger = logger or logging.getLogger("resident-lane-host")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="flowmesh-resident", daemon=True
        )
        self._origin: ResidentOriginDriver | None = None
        self._replica: ResidentReplicaSidecar | None = None

    def start(self) -> None:
        """Start the loop thread and build both lanes on it."""
        self._thread.start()
        self._call(self._build).result()

    async def _build(self) -> None:
        sink: ResidentFrameSink = _EventFrameSink(self._push_frame)
        self._origin = ResidentOriginDriver(
            sink=sink,
            content_store=self._content_store,
            report_ack=self._report_ack,
            report_outcome=self._report_outcome,
            logger=self._logger,
        )
        self._replica = ResidentReplicaSidecar(
            sink=sink,
            engine_open=self._engine_open,
            logger=self._logger,
        )

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(
        self, coro_fn: Callable[[], Coroutine[Any, Any, Any]]
    ) -> "concurrent.futures.Future[Any]":
        return asyncio.run_coroutine_threadsafe(coro_fn(), self._loop)

    def route(self, frame_kind: str, frame: dict[str, Any]) -> bool:
        """Marshal one resident control frame onto the lane loop; return handled."""
        if frame_kind == "resident_handoff":
            self._loop.call_soon_threadsafe(self._begin, frame)
        elif frame_kind == "resident_authorization":
            self._loop.call_soon_threadsafe(self._authorize, frame)
        elif frame_kind == "resident_sidecar_bind":
            self._loop.call_soon_threadsafe(self._bind, frame)
        elif frame_kind == "resident_reap":
            self._loop.call_soon_threadsafe(self._reap, frame)
        elif frame_kind == "resident_frame":
            asyncio.run_coroutine_threadsafe(self._on_frame(frame), self._loop)
        else:
            return False
        return True

    def _begin(self, frame: dict[str, Any]) -> None:
        if self._origin is None:
            return
        handoff = AdmissionHandoff.model_validate(frame["handoff"])
        request = self._peek_request(frame["task_id"], frame["call_correlation"])
        self._origin.begin(
            ResidentOriginRequest(
                task_id=str(frame["task_id"]),
                call_correlation=str(frame["call_correlation"]),
                session_id=str(frame["session_id"]),
                handoff=handoff,
                request_payload=request,
            )
        )

    def _authorize(self, frame: dict[str, Any]) -> None:
        if self._origin is not None:
            self._origin.authorize(
                str(frame["call_correlation"]),
                RouteAuthorization.model_validate(frame["auth"]),
            )

    def _bind(self, frame: dict[str, Any]) -> None:
        if self._replica is None:
            return
        engine = frame["engine"]
        self._replica.bind(
            replica_id=str(frame["replica_id"]),
            incarnation=int(frame["incarnation"]),
            listener_generation=int(frame["listener_generation"]),
            endpoint=ReplicaEndpoint(
                base_url=str(engine["base_url"]),
                model=str(engine.get("model") or ""),
                api_key=engine.get("api_key"),
            ),
        )

    def _reap(self, frame: dict[str, Any]) -> None:
        if self._origin is not None:
            self._origin.reap(str(frame["call_correlation"]))

    async def _on_frame(self, frame: dict[str, Any]) -> None:
        relay = RelayFrame.from_wire(frame)
        # A frame's direction names the receiver's role: the origin receives
        # target-to-origin, the replica receives origin-to-target.
        if relay.direction is RelayDirection.TARGET_TO_ORIGIN:
            if self._origin is not None:
                await self._origin.on_frame(relay)
        elif self._replica is not None:
            await self._replica.on_frame(relay)

    def stop(self) -> None:
        """Reap the lanes and stop the loop thread."""
        if self._replica is not None:
            with contextlib.suppress(Exception):
                self._call(self._replica.aclose).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
