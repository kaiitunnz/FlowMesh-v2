"""Shared stubs for the supervisor tests."""

import logging
import os
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock
from weakref import WeakSet

import fakeredis
from pydantic import SecretStr

from server.config import IdentityConfig
from server.registries.node import NodeRegistry
from server.supervisor.manager import WorkerManager
from server.supervisor.provisioning import (
    RunState,
    WorkerProvisioningStore,
    WorkerRecord,
)
from server.supervisor.registry import WorkerRegistry
from server.supervisor.services.lifecycle import Lifecycle

_LOGGER = logging.getLogger("test.supervisor")


def memory_store(identity: IdentityConfig | None = None) -> WorkerProvisioningStore:
    """A supervisor state store over an in-memory Redis."""
    return WorkerProvisioningStore(
        fakeredis.FakeRedis(decode_responses=True), identity or IdentityConfig()
    )


def worker_record(
    alias: str,
    provider: str = "docker",
    token: str = "tok",
    config: dict[str, Any] | None = None,
    **fields: Any,
) -> WorkerRecord:
    """A record of a running worker, unless ``fields`` say otherwise."""
    return WorkerRecord(
        alias=alias,
        provider=provider,
        config=config or {},
        token=SecretStr(token),
        **{"run_state": RunState.RUNNING, **fields},
    )


class StubRegistry(NodeRegistry):
    """NodeRegistry whose ``node_exists`` returns a fixed value and records the
    ids it was queried with."""

    def __init__(self, exists: bool) -> None:
        self._exists = exists
        self.exists_calls: list[str] = []

    def node_exists(self, node_id: str) -> bool:
        self.exists_calls.append(node_id)
        return self._exists


class StubLifecycle(Lifecycle):
    """Lifecycle with registration stubbed: re-register mints ``nde-2`` and each
    published event type is recorded instead of sent."""

    def __init__(self, node_registry: NodeRegistry, node_id: str) -> None:
        self._node_registry = node_registry
        self._node_id = node_id
        self.logger = _LOGGER
        self._unregister_published = True
        self._shutting_down = False
        self._on_reregister = None
        self.published_events: list[str] = []

    def _register(self) -> str:
        return "nde-2"

    def _publish_event(self, event_type: str, **extra: object) -> None:
        self.published_events.append(event_type)


class StubWorkerManager(WorkerManager):
    """WorkerManager in the started state, with no config file, providers, or
    Docker. The registry defaults to a ``MagicMock``."""

    def __init__(self, registry: WorkerRegistry | None = None) -> None:
        self.config_path = os.devnull
        self.logger = _LOGGER
        self._registry = registry if registry is not None else MagicMock()
        self._store = memory_store()
        self._records = {}
        self._records_lock = threading.RLock()
        self._unsaved = set()
        self._removing = {}
        self._awaiting = set()
        self._to_provision = []
        self._grace_deadline = None
        self._loop = None
        self._in_flight = Counter()
        self._tasks = set()
        self._is_started = True
        self._default_worker_config = {}
        self._capacity_change_callback: Callable[[], None] | None = None
        self._destroyed = WeakSet()
        self._providers = {}
