"""The remote external-tool egress sidecar: the enforced tool-fence boundary.

It binds in a worker node's supervisor process and performs the actual provider egress
off the server. A connection carries one operation: an operation frame opens it under a
``RemoteToolOperationEnvelope`` the sidecar validates before any egress — interface,
provider and target audience, policy, expiry, request-digest integrity, and result
budget. A failed fence returns a reject frame with no provider call and never demotes
reachability. On a valid fence the bounded provider egress runs off the event loop, so a
blocking search does not stall the node, and the typed outcome returns in one result
frame. The provider credential is read from this process's local environment and never
travels on the wire.
"""

import asyncio
import logging
import time

from . import tool_sidecar_wire as wire
from .tool_egress import (
    ExternalToolSidecar,
    RemoteToolOperationEnvelope,
    ToolOperationEnvelope,
    ToolRequest,
    tool_request_digest,
)


class ExternalToolSidecarServer:
    """Serves one bound external-tool target's fence-gated egress connections."""

    def __init__(
        self,
        *,
        sidecar: ExternalToolSidecar,
        target_id: str,
        target_generation: int,
        provider: str,
        interfaces: frozenset[str],
        policy_class: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:
        self._sidecar = sidecar
        self._target_id = target_id
        self._target_generation = target_generation
        self._provider = provider
        self._interfaces = interfaces
        self._policy_class = policy_class
        self._log = logger or logging.getLogger("external-tool-sidecar")

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await self._serve(reader, writer)
        except (
            ValueError,
            KeyError,
            ConnectionError,
            asyncio.IncompleteReadError,
            OSError,
        ):
            # A transport or decode error is not a fence failure; the origin reads the
            # close and leaves the durable boundary pending for its idm-* re-drive.
            pass
        finally:
            await _close(writer)

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        opening = await wire.read_msg(reader)
        if opening["kind"] != wire.KIND_OPERATION:
            return
        envelope = RemoteToolOperationEnvelope.model_validate(opening["envelope"])
        request = ToolRequest.model_validate(opening["request"])
        if (reason := self._fence_reject(envelope, request)) is not None:
            self._log.info(
                "tool fence rejected op target=%s reason=%s", self._target_id, reason
            )
            await wire.write_msg(writer, wire.KIND_REJECT, reason=reason)
            return
        colocated = ToolOperationEnvelope(
            interface=envelope.interface,
            idempotency_key=envelope.idempotency_key,
            max_results=envelope.max_results,
            timeout_sec=envelope.timeout_sec,
            result_char_cap=envelope.result_char_cap,
        )
        # The provider egress blocks; run it off the event loop so a slow search does
        # not stall the node's other sidecar connections or its relay attachment pump.
        outcome = await asyncio.get_running_loop().run_in_executor(
            None, self._sidecar.execute, colocated, request
        )
        await wire.write_msg(
            writer, wire.KIND_RESULT, outcome=outcome.model_dump(mode="json")
        )

    def _fence_reject(
        self, envelope: RemoteToolOperationEnvelope, request: ToolRequest
    ) -> str | None:
        if envelope.interface not in self._interfaces:
            return "interface"
        if envelope.provider != self._provider:
            return "provider"
        if envelope.target_id != self._target_id:
            return "audience"
        if envelope.target_generation != self._target_generation:
            return "generation"
        if envelope.policy_class != self._policy_class:
            return "policy"
        if time.time() > envelope.deadline_epoch:
            return "expired"
        if request.interface != envelope.interface:
            return "interface_mismatch"
        if request.max_results > envelope.max_results:
            return "budget"
        if (
            tool_request_digest(request.interface, request.query, request.max_results)
            != envelope.request_digest
        ):
            return "digest"
        return None


class ExternalToolSidecarListener:
    """A per-target TCP listener bound on the sidecar route."""

    def __init__(self, server: ExternalToolSidecarServer, *, route: str) -> None:
        self._server = server
        self._host, self._port = wire.split_host_port(route)
        self._tcp: asyncio.Server | None = None
        self._conns: set[asyncio.Task[None]] = set()

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conns.add(task)
        try:
            await self._server.handle(reader, writer)
        finally:
            if task is not None:
                self._conns.discard(task)

    async def start(self) -> tuple[str, int]:
        self._tcp = await asyncio.start_server(
            self._on_connection, self._host, self._port
        )
        host, port = self._tcp.sockets[0].getsockname()[:2]
        return host, port

    async def stop(self) -> None:
        if self._tcp is None:
            return
        self._tcp.close()
        for task in list(self._conns):
            task.cancel()
        if self._conns:
            await asyncio.gather(*self._conns, return_exceptions=True)
        try:
            await self._tcp.wait_closed()
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass
        self._tcp = None


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass


__all__ = ["ExternalToolSidecarListener", "ExternalToolSidecarServer"]
