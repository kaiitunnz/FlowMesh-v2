"""The worker-side registry of harness-adapter bindings, keyed by backend.

A backend key selects one adapter binding; the module hosting it is imported lazily on
first use, so a binding's dependencies (a Codex app-server client, say) stay off the
import path of a worker that never selects it. A binding may also register a factory
directly, which tests use to install a double.
"""

import importlib
from collections.abc import Callable

from shared.harness import HarnessAdapter, HarnessBackendKey
from shared.tasks.worker_message import WorkerTaskMessage
from worker.config import WorkerConfig

type AdapterFactory = Callable[
    [HarnessBackendKey, WorkerTaskMessage, WorkerConfig], HarnessAdapter
]

_ADAPTER_MODULES: dict[str, tuple[str, str]] = {
    "scripted": (".scripted", "build_scripted_adapter"),
    "codex": (".codex", "build_codex_adapter"),
}

_REGISTERED: dict[str, AdapterFactory] = {}


class UnknownHarnessBackendError(RuntimeError):
    """Raised when a dispatch names a backend key no binding services."""


def register_adapter(backend: str, factory: AdapterFactory) -> None:
    """Register a factory for a backend, taking precedence over the module map."""
    _REGISTERED[backend] = factory


def build_adapter(
    backend: HarnessBackendKey, task: WorkerTaskMessage, config: WorkerConfig
) -> HarnessAdapter:
    """Instantiate the adapter for a backend key, importing its module on first use."""
    if (factory := _REGISTERED.get(backend.backend)) is not None:
        return factory(backend, task, config)
    if (entry := _ADAPTER_MODULES.get(backend.backend)) is None:
        raise UnknownHarnessBackendError(
            f"no harness binding for backend {backend.backend!r}"
        )
    module, factory_name = entry
    factory = getattr(importlib.import_module(module, __package__), factory_name)
    return factory(backend, task, config)
