"""The replica worker's claim-gated resident sidecar lane.

A replica worker binds this lane for its serve task's incarnation. It admits data-plane
traffic only after the claim gate validates the fence it carries against the bound
incarnation and listener generation, then serves the co-located engine's response over
the invocation's windowed relay session. One engine runs per invocation under one
credit; a fresh-session re-drive supersedes its prior attempt, and a cancel reaps both
the session and its engine request. A resident allocation is reachable only through this
gate.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from shared.network.relay_frame import RelayFrame, RelayFrameKind
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.gate import LoadEvidence, SidecarClaimGate
from shared.resident.wire import (
    KIND_ACK,
    KIND_BOOTSTRAP,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_REJECT,
    KIND_STREAM,
)

from .engine import EngineOpen, EngineUnload, unload_adapter
from .session import ResidentRelaySession, ResidentSessionRole
from .transport import ResidentFrameSink

# Claim-tagged load evidence one admitted operation emits for control-plane accounting.
LoadSink = Callable[[LoadEvidence], None]


@dataclass
class _Binding:
    """One bound replica incarnation: its gate and co-located engine endpoint."""

    gate: SidecarClaimGate
    endpoint: ReplicaEndpoint


class ResidentReplicaSidecar:
    """Serves a replica incarnation's claim-gated resident invocations."""

    def __init__(
        self,
        *,
        sink: ResidentFrameSink,
        engine_open: EngineOpen,
        engine_unload: EngineUnload | None = None,
        window_bytes: int = 65536,
        stream_deadline_sec: float = 300.0,
        on_load: LoadSink | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sink = sink
        self._engine_open = engine_open
        self._engine_unload = engine_unload or unload_adapter
        self._window_bytes = window_bytes
        self._stream_deadline = stream_deadline_sec
        self._on_load = on_load or (lambda _ev: None)
        self._logger = logger or logging.getLogger("resident-replica-sidecar")
        self._bindings: dict[str, _Binding] = {}
        self._sessions: dict[str, ResidentRelaySession] = {}
        self._serves: dict[str, asyncio.Task[None]] = {}
        # The live serve task per invocation, so a fresh-session re-drive supersedes its
        # prior attempt and exactly one engine runs per invocation under one credit.
        self._inflight: dict[str, asyncio.Task[None]] = {}

    def bind(
        self,
        *,
        replica_id: str,
        incarnation: int,
        listener_generation: int,
        endpoint: ReplicaEndpoint,
    ) -> None:
        """Bind (or rebind) the claim gate and engine endpoint for one incarnation."""
        self._bindings[replica_id] = _Binding(
            gate=SidecarClaimGate(
                replica_id=replica_id,
                incarnation=incarnation,
                listener_generation=listener_generation,
            ),
            endpoint=endpoint,
        )

    def unbind(self, replica_id: str) -> None:
        """Drop a replica's binding; in-flight sessions run to their own terminal."""
        self._bindings.pop(replica_id, None)

    async def unload_adapter(self, replica_id: str, adapter_name: str) -> None:
        """Free an adapter's engine slot once its last credit-bearing claim released.

        Control decides the last holder released — gated on the same held-adapter set
        that arms admission — so this only frees the slot the server already reclaimed;
        it never unloads an adapter a peer still holds. Best effort and fail-loud: an
        unbound replica is a no-op, and a rare failed unload is logged and leaves the
        slot occupied until the replica is re-materialized — on a preempt, or an idle
        teardown when a retain window or serve TTL is configured (not the default).
        """
        binding = self._bindings.get(replica_id)
        if binding is None:
            return
        try:
            await self._engine_unload(binding.endpoint, adapter_name)
        except (httpx.HTTPError, OSError) as exc:
            self._logger.warning(
                "resident adapter unload failed (replica=%s adapter=%s): %s",
                replica_id,
                adapter_name,
                exc,
            )

    def reap_invocation(self, invocation_id: str) -> None:
        """Cancel the live serve task for an invocation on a fenced terminal or cancel.

        Cancelling the serve task runs its teardown — closing the engine request and
        reaping the session — so a cancelled or terminalized invocation stops the engine
        promptly rather than blocking on a receiver that stopped draining.
        """
        task = self._inflight.get(invocation_id)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def on_frame(self, frame: RelayFrame) -> None:
        """Route one inbound relay frame to its session, opening one on a bootstrap."""
        session = self._sessions.get(frame.session_id)
        if session is None:
            if frame.kind is not RelayFrameKind.DATA or frame.seq != 1:
                # A stray frame for a session this replica never opened (a late cancel
                # or window for a reaped session): nothing to route it to.
                return
            session = self._open_session(
                frame.session_id, frame.invocation_id, frame.idm
            )
        await session.on_frame(frame)
        if frame.kind is RelayFrameKind.CANCEL:
            self._reap(frame.session_id)

    def _open_session(
        self, session_id: str, invocation_id: str, idm: str
    ) -> ResidentRelaySession:
        session = ResidentRelaySession(
            session_id=session_id,
            invocation_id=invocation_id,
            idm=idm,
            role=ResidentSessionRole.REPLICA,
            sink=self._sink,
            window_bytes=self._window_bytes,
        )
        self._sessions[session_id] = session
        self._serves[session_id] = asyncio.ensure_future(
            self._serve(session_id, session)
        )
        return session

    async def _serve(self, session_id: str, session: ResidentRelaySession) -> None:
        engine_aclose = None
        try:
            opening = await session.recv_wire(self._stream_deadline)
            if opening is None or opening.get("kind") != KIND_BOOTSTRAP:
                return
            handoff = AdmissionHandoff.model_validate(opening["handoff"])
            binding = self._bindings.get(handoff.replica_id)
            if binding is None:
                # The bind frame has not arrived yet (a cold-start bootstrap/bind race):
                # a transient not-yet-bound condition, not a genuine fence rejection, so
                # signal a loss the origin holds and re-drives rather than a definite
                # reject that would release the credit and preempt a healthy replica.
                await session.send_wire(
                    KIND_FAILED, definite=False, reason="sidecar not bound"
                )
                return
            decision = binding.gate.check_bootstrap(handoff)
            if not decision.admitted:
                await session.send_wire(KIND_REJECT, reason=str(decision.rejection))
                return
            gate_session = binding.gate.session_for(handoff)
            self._supersede(handoff.invocation_id)
            self._on_load(binding.gate.load_evidence(handoff, "request"))
            # Open the engine request and acknowledge immediately: the ack marks engine
            # receipt, not completion, so control can authorize the response stream
            # before inference finishes.
            engine_task = asyncio.ensure_future(
                self._engine_open(
                    binding.endpoint,
                    opening.get("request"),
                    handoff.adapter_name,
                    handoff.adapter_source,
                )
            )
            try:
                await session.send_wire(KIND_ACK)
                follow = await session.recv_wire(self._stream_deadline)
                if follow is None:
                    return
                auth = RouteAuthorization.model_validate(follow["auth"])
                gate = binding.gate.check_stream(auth, gate_session)
                if not gate.admitted:
                    await session.send_wire(KIND_REJECT, reason=str(gate.rejection))
                    return
                if follow.get("kind") != KIND_STREAM:
                    return
                self._on_load(binding.gate.load_evidence(auth, "stream"))
                try:
                    engine = await engine_task
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    # A 4xx request error (bar 429) is a definite refusal that held no
                    # slot, so the origin releases the credit. A 429 or any 5xx is a
                    # transient engine condition carried as uncertain so the boundary
                    # holds the credit and re-drives.
                    definite = 400 <= status < 500 and status != 429
                    await session.send_wire(
                        KIND_FAILED, definite=definite, reason=f"engine {status}"
                    )
                    return
                engine_aclose = engine.aclose
                async for chunk in engine.chunks:
                    await session.send_wire(KIND_CHUNK, data=chunk)
                await session.send_wire(KIND_DONE)
            finally:
                if not engine_task.done():
                    engine_task.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await engine_task
        except (ValidationError, KeyError, httpx.HTTPError, OSError):
            # A malformed follow frame, a dropped engine connection, or a transport
            # error closes the session without a terminal; the origin reads the loss and
            # settles the boundary. It is not the caller's fence failure.
            pass
        finally:
            if engine_aclose is not None:
                with contextlib.suppress(Exception):
                    await engine_aclose()
            self._reap(session_id)

    def _supersede(self, invocation_id: str) -> None:
        current = asyncio.current_task()
        prior = self._inflight.get(invocation_id)
        if prior is not None and prior is not current and not prior.done():
            # A prior attempt for this invocation is still live (its engine running as
            # it awaits a stream that never comes); cancel it so no two engines run.
            prior.cancel()
        if current is not None:
            self._inflight[invocation_id] = current

    def _reap(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        task = self._serves.pop(session_id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        for inv, serve in list(self._inflight.items()):
            if serve is task:
                self._inflight.pop(inv, None)

    async def aclose(self) -> None:
        """Cancel every live serve task; for worker shutdown."""
        for task in list(self._serves.values()):
            if not task.done():
                task.cancel()
        pending = [t for t in self._serves.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._serves.clear()
        self._sessions.clear()
        self._inflight.clear()
