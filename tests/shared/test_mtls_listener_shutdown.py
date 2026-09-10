"""The offload listener's teardown does not wait on a peer that never closes.

A legitimate offload connection idles between frames for as long as its invocation runs,
so a listener that waited for its reads to end on their own would hold a node's shutdown
open for as long as a dialer keeps its socket.
"""

import asyncio
import contextlib
import socket

from shared.network.frame_stream import write_relay_frame
from shared.network.mtls_listener import MutualTlsFrameListener
from shared.network.relay_frame import RelayDirection, RelayFrame, RelayFrameKind


class _Idle:
    """A handler that never closes the connection on its own."""

    def __init__(self, sink) -> None:
        self.sink = sink

    async def on_frame(self, frame: RelayFrame) -> None:
        return None

    def close(self) -> None:
        return None


def _frame() -> RelayFrame:
    return RelayFrame(
        kind=RelayFrameKind.DATA,
        session_id="rly-1",
        invocation_id="inv-1",
        idm="idm-1",
        direction=RelayDirection.ORIGIN_TO_TARGET,
        seq=1,
        payload=b"x",
    )


def test_stopping_releases_a_connection_whose_peer_keeps_it_open() -> None:
    async def run() -> None:
        listener = MutualTlsFrameListener(
            material=None,
            handler=_Idle,
            admits=lambda identities: True,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        await listener.start_on_socket(sock)

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await write_relay_frame(writer, _frame())
        for _ in range(100):
            if listener._open == 1:
                break
            await asyncio.sleep(0.02)
        assert listener._open == 1

        # The dialer holds its socket open, exactly as an in-flight invocation would.
        await asyncio.wait_for(listener.stop(), timeout=10.0)

        # The connection it was serving ended with it, cleanly or by reset.
        with contextlib.suppress(ConnectionError):
            assert await asyncio.wait_for(reader.read(), timeout=5.0) == b""
        assert listener._open == 0

        writer.close()

    asyncio.run(run())
