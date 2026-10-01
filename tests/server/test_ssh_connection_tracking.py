"""A relayed SSH connection is listed while it lasts, however it is cancelled."""

import asyncio
import logging
from typing import Any, cast

from server.schemas.ssh import SSHConnectionInfo
from server.services.ssh_connections import SshConnectionRegistry
from server.ssh import SshRelayTarget
from server.ssh.connections import tracked_ssh_connection

TARGET = SshRelayTarget(
    task_id="tsk-1", worker_id="wkr-1", node_id="nde-1", endpoint_id="ssn-1"
)


class _SlowRegistry:
    """Writes each record at once and finishes each call only after a delay, as a
    Redis call cancelled after the server applied it does."""

    def __init__(self) -> None:
        self.records: set[str] = set()
        self.registering = asyncio.Event()
        self.unregistering = asyncio.Event()

    async def register_connection(self, info: SSHConnectionInfo) -> None:
        self.records.add(info.connection_id)
        self.registering.set()
        await asyncio.sleep(0.05)

    async def unregister_connection(self, connection_id: str) -> None:
        self.unregistering.set()
        await asyncio.sleep(0.05)
        self.records.discard(connection_id)


async def _connect(registry: _SlowRegistry, held: asyncio.Event) -> None:
    async with tracked_ssh_connection(
        cast(SshConnectionRegistry, registry),
        "proxy",
        TARGET,
        "wfl-1",
        "flowmesh",
        (None, None),
        logging.getLogger("test.ssh_connection_tracking"),
    ):
        await held.wait()


async def _settle(task: "asyncio.Task[Any]") -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.2)


def test_a_cancel_while_the_connection_registers_leaves_no_record() -> None:
    async def run() -> None:
        registry = _SlowRegistry()
        task = asyncio.create_task(_connect(registry, asyncio.Event()))
        await registry.registering.wait()

        await _settle(task)

        assert registry.records == set()

    asyncio.run(run())


def test_a_cancel_while_the_connection_unregisters_leaves_no_record() -> None:
    async def run() -> None:
        registry = _SlowRegistry()
        held = asyncio.Event()
        task = asyncio.create_task(_connect(registry, held))
        await registry.registering.wait()
        await asyncio.sleep(0.1)
        held.set()
        await registry.unregistering.wait()

        await _settle(task)

        assert registry.records == set()

    asyncio.run(run())
