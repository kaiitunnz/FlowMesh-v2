"""A supervisor servicer over fakeredis with one external worker, for driving its RPCs
directly."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock

import fakeredis
import grpc

from server.clients.redis import SyncRedisClient
from server.hooks import PrincipalContext
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.external import (
    ExternalWorkerAdapter,
    ExternalWorkerConfig,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.grpc_server import SupervisorServicer
from server.supervisor.services.relay_service import RelayService
from shared.grpc.supervisor.v1 import supervisor_pb2
from tests.server.redis_helpers import fake_sync_client

_LOGGER = logging.getLogger("test.servicer")
TOKEN = "tok-1"
ALIAS = "worker-1"
NODE_ALIAS = "box"


def external_adapter(
    token: str, alias: str, cls: type[ExternalWorkerAdapter] = ExternalWorkerAdapter
) -> ExternalWorkerAdapter:
    """An external worker's adapter, admitted under ``token``."""
    return cls(
        cast(WorkerTokenType, token),
        alias,
        ExternalWorkerConfig(),
        MagicMock(spec=PrincipalContext),
    )


def supervisor_servicer(
    registry: WorkerRegistry,
    client: SyncRedisClient,
    node_id: str = "nod-1",
    node_alias: str = NODE_ALIAS,
    task_listener: Any = None,
    relay: RelayService | None = None,
) -> SupervisorServicer:
    """A servicer over ``registry`` and ``client``, its other collaborators mocked."""
    return SupervisorServicer(
        registry,
        client,
        node_id,
        node_alias,
        task_listener or MagicMock(),
        relay or cast(RelayService, MagicMock()),
        MagicMock(),
        _LOGGER,
    )


class RecordingRelay(RelayService):
    def __init__(self) -> None:
        super().__init__(cast(SyncRedisClient, None), _LOGGER)
        self.events: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []

    def add_event(self, event_data: Any) -> None:
        self.events.append(dict(event_data))

    def add_log(self, log_data: Any) -> None:
        self.logs.append(dict(log_data))

    def unregisters(self) -> list[str]:
        return [e["worker_id"] for e in self.events if e["type"] == "UNREGISTER"]


class Aborted(Exception):
    def __init__(self, code: grpc.StatusCode) -> None:
        self.code = code


class WorkerContext:
    def invocation_metadata(self) -> list[tuple[str, str]]:
        return [("authorization", f"Bearer {TOKEN}")]

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise Aborted(code)


class ServicerHarness:
    def __init__(
        self,
        server: fakeredis.FakeServer | None = None,
        node_alias: str = NODE_ALIAS,
    ) -> None:
        server = server or fakeredis.FakeServer()
        self.server = server
        self.rds = fakeredis.FakeRedis(server=server, decode_responses=True)
        self.relay = RecordingRelay()
        self.released: list[str] = []
        self.registry = WorkerRegistry(on_worker_id_released=self._released)
        self.adapter = external_adapter(TOKEN, ALIAS)
        self.registry.add(self.adapter)
        self.servicer = supervisor_servicer(
            self.registry,
            fake_sync_client(server),
            node_alias=node_alias,
            relay=self.relay,
        )

    def _released(self, worker_id: str) -> None:
        self.released.append(worker_id)
        self.servicer.worker_id_released(worker_id)

    async def register(self) -> str:
        response = await self.servicer.RegisterWorker(
            supervisor_pb2.RegisterRequest(), cast(Any, WorkerContext())
        )
        return response.worker_id
