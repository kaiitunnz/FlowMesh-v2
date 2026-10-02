"""A worker serves the peer listener it advertises from registration on."""

import asyncio
import socket
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from shared.network.frame_stream import ProbeFrame, read_stream_frame, write_probe
from tests.worker.factories import make_worker_hardware, no_mediated_op
from worker.lifecycle import Lifecycle
from worker.resident.lane_host import ResidentLaneHost
from worker.runner import Runner


def _bound() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    sock.setblocking(False)
    return sock


async def _probe(port: int) -> Any:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await write_probe(writer, b"ping")
        return await asyncio.wait_for(read_stream_frame(reader), timeout=2.0)
    finally:
        writer.close()


def test_a_worker_answers_on_its_peer_listener_before_any_resident_frame(
    tmp_path: Path,
) -> None:
    sock = _bound()
    lifecycle = Lifecycle(MagicMock(), 5, 15, tmp_path / "hb", 0.0)
    client = cast(MagicMock, lifecycle.client)
    client.worker_id = "wkr-1"
    client.iter_interrupts.return_value = []
    client.iter_stops.return_value = []
    client.next_mediated_op.side_effect = no_mediated_op
    runner = Runner(
        lifecycle=lifecycle,
        task_stream=[],
        results_dir=tmp_path / "out",
        hardware=make_worker_hardware(),
        executors={},
        default_executor=cast(Any, MagicMock()),
        logger=MagicMock(),
        peer_enabled=True,
        peer_listener_sock=sock,
    )
    try:
        runner.start()
        answer = asyncio.run(_probe(sock.getsockname()[1]))
    finally:
        if (host := runner._resident_host) is not None:
            cast(ResidentLaneHost, host).stop(1.0)

    assert answer == ProbeFrame(b"ping")
