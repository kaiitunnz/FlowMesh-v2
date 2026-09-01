"""Network-plane listeners for the feature-gated echo seam.

Each network-plane node runs a bounded echo sidecar (the resident-facing listener
stand-in) and a node-relay endpoint that uplinks to it; the root additionally runs a
control-relay endpoint reached as the controlled fallback. They exercise the transport
ladder and the bounded relay session without ever fronting a resident engine.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from shared.utils.ids import new_relay_session_id

from . import wire
from .relay import RelaySession

_ConnHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


def _split_host_port(endpoint: str) -> tuple[str, int]:
    host, _, port = endpoint.rpartition(":")
    return host or "127.0.0.1", int(port)


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.CancelledError):
        pass


class _TcpServer:
    """A minimal asyncio TCP server bound to a configured endpoint."""

    def __init__(
        self, endpoint: str, handler: _ConnHandler, *, logger: logging.Logger | None
    ) -> None:
        self._host, self._port = _split_host_port(endpoint)
        self._handler = handler
        self._logger = logger
        self._server: asyncio.Server | None = None

    async def start(self) -> tuple[str, int]:
        self._server = await asyncio.start_server(self._handler, self._host, self._port)
        host, port = self._server.sockets[0].getsockname()[:2]
        return host, port

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
            if self._logger is not None:
                self._logger.debug("Network listener close failed: %s", exc)
        self._server = None


async def _echo_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        payload = await wire.read_frame(reader)
        if payload == wire.APP_ERROR_SENTINEL:
            writer.write(wire.STATUS_APP_ERROR)
        else:
            writer.write(wire.STATUS_OK + payload)
        await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        pass
    finally:
        await _close(writer)


def _fixed_relay_connection(
    target: str, *, buffer_bytes: int, logger: logging.Logger | None
) -> _ConnHandler:
    target_host, target_port = _split_host_port(target)

    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            target_reader, target_writer = await asyncio.open_connection(
                target_host, target_port
            )
        except OSError:
            await _close(writer)
            return
        await RelaySession(
            new_relay_session_id(), buffer_bytes=buffer_bytes, logger=logger
        ).relay(reader, writer, target_reader, target_writer)

    return handler


def _control_relay_connection(
    *, buffer_bytes: int, logger: logging.Logger | None
) -> _ConnHandler:
    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            target = (await wire.read_frame(reader)).decode()
            target_host, target_port = _split_host_port(target)
            target_reader, target_writer = await asyncio.open_connection(
                target_host, target_port
            )
        except (OSError, ValueError, asyncio.IncompleteReadError):
            await _close(writer)
            return
        await RelaySession(
            new_relay_session_id(), buffer_bytes=buffer_bytes, logger=logger
        ).relay(reader, writer, target_reader, target_writer)

    return handler


class NetworkPlaneListeners:
    """The per-node echo sidecar and its node-relay uplink endpoint."""

    def __init__(
        self,
        *,
        sidecar_url: str,
        endpoint_url: str,
        buffer_bytes: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sidecar = _TcpServer(sidecar_url, _echo_connection, logger=logger)
        self._node_relay = _TcpServer(
            endpoint_url,
            _fixed_relay_connection(
                sidecar_url, buffer_bytes=buffer_bytes, logger=logger
            ),
            logger=logger,
        )

    async def start(self) -> None:
        await self._sidecar.start()
        await self._node_relay.start()

    async def stop(self) -> None:
        await self._node_relay.stop()
        await self._sidecar.stop()


class NetworkControlRelay:
    """The root's controlled-fallback relay, reached by ``control_relay`` routes."""

    def __init__(
        self,
        *,
        control_relay_url: str,
        buffer_bytes: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._server = _TcpServer(
            control_relay_url,
            _control_relay_connection(buffer_bytes=buffer_bytes, logger=logger),
            logger=logger,
        )

    async def start(self) -> None:
        await self._server.start()

    async def stop(self) -> None:
        await self._server.stop()
