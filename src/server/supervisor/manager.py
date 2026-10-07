import asyncio
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any, Self
from weakref import WeakSet

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

from ..hooks import PrincipalContext
from .adapters.base import ProviderSpec, WorkerAdapter, WorkerFactory, WorkerTokenType
from .adapters.docker import get_provider_spec as docker_provider_spec
from .adapters.external import get_provider_spec as external_provider_spec
from .adapters.external import verify_external_token
from .adapters.vastai import get_provider_spec as vastai_provider_spec
from .provisioning import (
    ProviderHandle,
    RecordState,
    Removal,
    RunState,
    WorkerProvisioningStore,
    WorkerRecord,
    recorded_config,
)
from .registry import WorkerRegistry
from .schemas import WorkerHardware, WorkerInfo, WorkerStatus

_MAX_PARALLELISM: int = 16
# How long a worker restored from its record has to register again before it is
# removed, counted from when the supervisor accepts registrations.
WORKER_RECONNECT_GRACE_SEC = 300.0
_STORE_RETRY_MAX_SEC = 60.0


def _is_live(worker: WorkerAdapter) -> bool:
    # The status reads STOPPED whenever the worker's event stream closes, while what the
    # adapter started may still run.
    return worker.status is not WorkerStatus.STOPPED or worker.holds_worker()


class ManagerNotStartedError(RuntimeError):
    """Raised when an operation arrives before the manager has started."""

    def __init__(self, message: str = "WorkerManager not started") -> None:
        super().__init__(message)


class ProviderUnavailableError(ValueError):
    """Raised when a create request names a provider this node does not have."""


class WorkerInitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="docker", description="Worker provider")
    init_on_start: bool = Field(
        default=True,
        description="Whether to start the worker immediately",
    )
    worker_token: WorkerTokenType | None = Field(
        default=None, description="Optional worker token (overrides registry token)"
    )
    worker_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific worker config"
    )

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in {"provider", "init_on_start", "worker_token", "worker_config"}
        }


class ServerWorkerConfig(BaseModel):
    default_worker_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration applied to all workers",
    )
    workers: list[WorkerInitConfig] = Field(
        default_factory=list,
        description="List of worker configurations",
    )

    @model_validator(mode="after")
    def require_worker_aliases(self) -> Self:
        if "worker_alias" in self.default_worker_config:
            raise ValueError("default_worker_config cannot set worker_alias")
        aliases: set[str] = set()
        for i, entry in enumerate(self.workers):
            config = self.default_worker_config | entry.worker_config
            alias = config.get("worker_alias")
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"workers[{i}] must set worker_config.worker_alias")
            if alias in aliases:
                raise ValueError(f"Worker alias '{alias}' is declared more than once")
            aliases.add(alias)
            label = config.get("label")
            if entry.provider.strip().lower() == "vastai" and label not in (
                None,
                alias,
            ):
                raise ValueError(
                    f"Worker '{alias}' sets a label other than its alias; a VastAI "
                    "worker takes its label as its alias"
                )
        return self


class WorkerManager:
    def __init__(
        self,
        system_principal: PrincipalContext,
        config_path: str,
        registry: WorkerRegistry,
        logger: logging.Logger,
        store: WorkerProvisioningStore,
        capacity_change_callback: Callable[[], None] | None = None,
        vast_api_key: SecretStr | None = None,
    ) -> None:
        self.config_path = config_path
        self.logger = logger

        self._registry = registry
        self._store = store
        self._default_worker_config: dict[str, Any] | None = None
        self._is_started: bool = False
        self._capacity_change_callback = capacity_change_callback
        # Workers already destroyed: a create's unwind and a shutdown can both reach
        # one, and a second destroy would free its GPUs twice.
        self._destroyed: WeakSet[WorkerAdapter] = WeakSet()
        # The record of each provisioned worker, by alias. A launching thread commits
        # a handle while the loop writes the rest.
        self._records: dict[str, WorkerRecord] = {}
        self._records_lock = threading.RLock()
        # Aliases whose latest record has not reached the store.
        self._unsaved: set[str] = set()
        # Adapters of workers being removed, out of the registry.
        self._removing: dict[str, WorkerAdapter] = {}
        # Restored workers expected to register again within the grace.
        self._awaiting: set[str] = set()
        self._to_provision: list[WorkerAdapter] = []
        self._grace_deadline: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Lifecycle operations under way, by alias; the heartbeat leaves those alone.
        self._in_flight: Counter[str] = Counter()
        self._tasks: set[asyncio.Task[Any]] = set()
        # External provider is always available.
        specs = [external_provider_spec(system_principal)]
        for label, build_spec in (
            (
                "Docker",
                lambda: docker_provider_spec(system_principal, self._alias_taken),
            ),
            (
                "Vast.ai",
                lambda: vastai_provider_spec(
                    system_principal, vast_api_key, self._alias_taken
                ),
            ),
        ):
            try:
                specs.append(build_spec())
            except Exception as exc:
                logger.warning(
                    "%s worker provider unavailable, continuing without it: %s",
                    label,
                    exc,
                )
        self._providers: dict[str, ProviderSpec] = {spec.name: spec for spec in specs}

    @property
    def is_started(self) -> bool:
        return self._is_started

    async def restore(self) -> list[str]:
        """Take back the workers this node's records name, before any is created.

        Return the worker ids their last registrations took.
        """
        for record in await self._load_records():
            self._records[record.alias] = record
            spec = self._providers.get(record.provider)
            try:
                if spec is None:
                    raise ValueError(f"provider '{record.provider}' is not available")
                worker = spec.factory.attach(
                    WorkerTokenType(record.token.get_secret_value()), record
                )
            except Exception as exc:
                self.logger.error("Failed to restore worker %s: %s", record.alias, exc)
                continue
            worker.on_handle = self._handle_committer(worker)
            if record.state is RecordState.REMOVING:
                self._removing[record.alias] = worker
                continue
            self._registry.add(worker)
            if record.handle is None and record.state is RecordState.PROVISIONING:
                found = await asyncio.to_thread(worker.recover_launch)
                if found is None:
                    self.logger.error(
                        "Worker %s was launching when the supervisor stopped and what "
                        "it launched cannot be told; resolve it by hand",
                        record.alias,
                    )
                    continue
                if not found:
                    self._update(record.alias, strict=False, state=RecordState.PRESENT)
            record = self._records[record.alias]
            if record.run_state is RunState.RUNNING:
                if record.handle is None:
                    self._to_provision.append(worker)
                else:
                    self._awaiting.add(record.alias)
        self._report_capacity_change()
        return [r.worker_id for r in self._records.values() if r.worker_id]

    async def _load_records(self) -> list[WorkerRecord]:
        delay = 1.0
        while True:
            try:
                return await asyncio.to_thread(self._store.load)
            except Exception as exc:
                self.logger.error(
                    "Failed to read the supervisor state store, retrying in %.0fs: %s",
                    delay,
                    exc,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _STORE_RETRY_MAX_SEC)

    async def start(self) -> None:
        if self.is_started:
            self.logger.warning("WorkerManager is already started.")
            return

        self._is_started = True
        self._default_worker_config = {}
        self._loop = asyncio.get_running_loop()
        to_start, self._to_provision = self._to_provision, []

        server_config = self._read_boot_config()
        if server_config is not None:
            self._default_worker_config = server_config.default_worker_config
        to_prepare: list[WorkerAdapter] = []
        for init_config in server_config.workers if server_config else []:
            alias = init_config.worker_config["worker_alias"]
            if self._alias_taken(alias):
                self.logger.info("Worker %s is already provisioned.", alias)
                continue
            try:
                worker = self._create_worker(init_config)
                worker_info = worker.get_info()
                self.logger.info(
                    "Created worker %s with provider '%s' (status=%s).",
                    worker_info.alias,
                    worker_info.provider,
                    worker_info.status,
                )
                if init_config.init_on_start:
                    to_start.append(worker)
                else:
                    to_prepare.append(worker)
            except Exception as exc:
                self.logger.error("Failed to register worker: %s", exc)

        await self._start_workers(to_start, to_prepare)

    def _read_boot_config(self) -> ServerWorkerConfig | None:
        if not os.path.isfile(self.config_path):
            self.logger.warning(
                "Worker config file '%s' does not exist. "
                "Skipping worker initialization.",
                self.config_path,
            )
            return None
        with open(self.config_path, encoding="utf-8") as f:
            raw = f.read()
        try:
            config_data = yaml.safe_load(raw) if raw.strip() else None
            if config_data is None:
                self.logger.info(
                    "Worker config file '%s' is empty. Skipping worker initialization.",
                    self.config_path,
                )
                return None
            return ServerWorkerConfig.model_validate(config_data)
        except (yaml.YAMLError, ValidationError) as exc:
            self.logger.error(
                "Worker config file '%s' is invalid; creating none of its workers: %s",
                self.config_path,
                exc,
            )
            return None

    async def _start_workers(
        self, to_start: list[WorkerAdapter], to_prepare: list[WorkerAdapter]
    ) -> None:
        if not (to_start or to_prepare):
            return

        max_parallel = min(len(to_start) + len(to_prepare), _MAX_PARALLELISM)
        sema = asyncio.Semaphore(max_parallel or 1)
        coros: list[Awaitable] = []
        if to_start:

            async def start_worker(worker: WorkerAdapter) -> None:
                async with sema:
                    try:
                        await self._start_worker(worker)
                    except Exception as exc:
                        self.logger.error(
                            "Failed to start worker %s: %s", worker.alias, exc
                        )

            coros.extend(start_worker(worker) for worker in to_start)

        if to_prepare:

            async def prepare_worker(worker: WorkerAdapter) -> None:
                async with sema:
                    try:
                        await worker.prepare()
                    except Exception as exc:
                        self.logger.error(
                            "Failed to prepare worker %s: %s", worker.alias, exc
                        )

            coros.extend(prepare_worker(worker) for worker in to_prepare)

        await asyncio.gather(*coros)
        self._report_capacity_change()

    async def stop(self) -> None:
        if not self.is_started:
            self.logger.warning("WorkerManager is not started.")
            return

        await self._stop_and_destroy_workers(self._registry.all_workers())
        self._report_capacity_change()
        for spec in self._providers.values():
            spec.factory.cleanup()
        self._registry.clear()
        self._default_worker_config = None
        self._is_started = False
        self.logger.info("Worker manager stopped")

    async def create_worker(self, init_config: WorkerInitConfig) -> WorkerInfo:
        if not self.is_started:
            raise ManagerNotStartedError()

        worker = self._create_worker(init_config)
        if init_config.init_on_start:
            # A create that fails or is cancelled (its command timing out) stops and
            # destroys its worker. The unwind outlives a second cancel, and the worker
            # keeps its alias until the unwind ends.
            try:
                if not await self._start_worker(worker):
                    raise RuntimeError(f"Failed to start worker '{worker.alias}'")
            except BaseException:
                unwind = asyncio.ensure_future(self._stop_and_destroy_worker(worker))
                unwind.add_done_callback(lambda _: self._registry.try_pop(worker.token))
                await asyncio.shield(unwind)
                raise
        self._report_capacity_change()
        return worker.get_info()

    async def admit_worker(self, token: WorkerTokenType) -> WorkerInfo | None:
        if not self.is_started:
            raise ManagerNotStartedError()

        if verify_external_token(token) is None:
            return None
        try:
            # init_on_start=False: an external worker is already running, so the
            # supervisor must not run its start lifecycle on it (that path gates
            # on STOPPED and would reject a worker born RUNNING).
            return await self.create_worker(
                WorkerInitConfig(
                    provider="external", worker_token=token, init_on_start=False
                )
            )
        except ValueError as exc:
            self.logger.warning("Failed to admit worker: %s", exc)
            return None

    def worker_registered(
        self, worker: WorkerAdapter, hardware: WorkerHardware | None
    ) -> None:
        """Apply what a worker reported when it registered."""
        self._awaiting.discard(worker.alias)
        if hardware is not None:
            worker.observe_reported_hardware(hardware)
        # A closing worker's destroy has released or is about to release its holds,
        # so a claim now would outlive it.
        if worker.closed:
            return
        if self._factory_for(worker).on_worker_registered(worker):
            self._report_capacity_change()

    def available_providers(self) -> list[str]:
        return sorted(self._providers)

    def list_workers(self) -> list[WorkerInfo]:
        if not self.is_started:
            return []
        return [worker.get_info() for worker in self._registry.all_workers()]

    def get_worker_info(self, alias: str) -> WorkerInfo | None:
        if not self.is_started:
            return None
        worker = self._registry.try_get_by_alias(alias)
        return None if worker is None else worker.get_info()

    async def start_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            raise ValueError(f"Worker '{alias}' does not exist")

        return await self._start_worker(worker)

    async def stop_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            raise ValueError(f"Worker '{alias}' does not exist")
        return await self._stop_worker(worker)

    async def destroy_worker(self, alias: str) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        worker = self._registry.try_get_by_alias(alias)
        if worker is None:
            return False

        # An accepted destroy completes even if its command is cancelled.
        destroying = asyncio.ensure_future(self._stop_and_destroy_worker(worker))
        destroying.add_done_callback(lambda _: self._forget_worker(alias))
        return await asyncio.shield(destroying)

    def _forget_worker(self, alias: str) -> None:
        self._registry.try_pop_by_alias(alias)
        self._report_capacity_change()

    async def destroy_workers(self, aliases: set[str] | None = None) -> None:
        if not self.is_started:
            raise ManagerNotStartedError()

        workers: list[WorkerAdapter]
        if aliases is None:
            workers = self._registry.all_workers()
        else:
            missing = [
                alias for alias in aliases if not self._registry.exists_by_alias(alias)
            ]
            if missing:
                raise ValueError(f"Workers not found: {', '.join(missing)}")
            workers = [self._registry.get_by_alias(alias) for alias in aliases]

        def forget(_: asyncio.Future[None]) -> None:
            for worker in workers:
                self._registry.discard(worker)
            self._report_capacity_change()

        destroying = asyncio.ensure_future(self._stop_and_destroy_workers(workers))
        destroying.add_done_callback(forget)
        await asyncio.shield(destroying)

    def _create_worker(self, init_config: WorkerInitConfig) -> WorkerAdapter:
        if not self.is_started:
            raise ManagerNotStartedError()

        token = init_config.worker_token or self._registry.new_token()
        provider = init_config.provider.strip().lower()
        worker_config = (self._default_worker_config or {}) | init_config.worker_config

        spec = self._providers.get(provider)
        if spec is None:
            raise ProviderUnavailableError(
                f"Worker provider '{provider}' is not available on this node; "
                f"available providers: {', '.join(sorted(self._providers))}"
            )
        config = spec.config_cls.model_validate(worker_config)
        worker = spec.factory.create_worker(token, config)

        try:
            if worker.alias in self._records:
                raise ValueError
            self._registry.add(worker)
        except ValueError:
            self._destroy_worker(worker)
            raise ValueError(f"Worker '{worker.alias}' already exists")
        if spec.provisioned:
            try:
                self._create_record(spec.name, worker, init_config.init_on_start)
            except BaseException:
                self._registry.discard(worker)
                self._destroy_worker(worker)
                raise
        return worker

    def _create_record(
        self, provider: str, worker: WorkerAdapter, init_on_start: bool
    ) -> None:
        record = WorkerRecord(
            alias=worker.alias,
            provider=provider,
            config=recorded_config(worker.config),
            token=SecretStr(worker.token),
            run_state=RunState.RUNNING if init_on_start else RunState.STOPPED,
        )
        with self._records_lock:
            if record.alias in self._records or not self._store.create(record):
                raise ValueError(f"Worker '{worker.alias}' already exists")
            self._records[record.alias] = record
        worker.on_handle = self._handle_committer(worker)

    def _alias_taken(self, alias: str) -> bool:
        return alias in self._records or self._registry.exists_by_alias(alias)

    def _update(self, alias: str, strict: bool = True, **changes: Any) -> None:
        """Change ``alias``'s record. When the store write fails, a strict update raises
        and changes nothing; any other keeps the change for the next heartbeat to
        save."""
        with self._records_lock:
            record = self._records.get(alias)
            if record is None:
                return
            updated = record.model_copy(update=changes)
            try:
                self._store.put(updated)
            except Exception:
                if strict:
                    raise
                self.logger.exception("Failed to save the record of worker %s", alias)
                self._unsaved.add(alias)
            else:
                self._unsaved.discard(alias)
            self._records[alias] = updated

    def _handle_committer(
        self, worker: WorkerAdapter
    ) -> Callable[[ProviderHandle | None], None]:
        def commit(handle: ProviderHandle | None) -> None:
            with self._records_lock:
                record = self._records.get(worker.alias)
                launched = (
                    handle is not None
                    and record is not None
                    and record.state is RecordState.PROVISIONING
                )
                state = RecordState.PRESENT if launched else None
                self._update(
                    worker.alias,
                    strict=False,
                    handle=handle,
                    **({"state": state} if state else {}),
                )

        return commit

    def commit_worker_id(self, worker: WorkerAdapter, worker_id: str) -> None:
        """Record the id a provisioned worker registered under; raise when the store
        write fails."""
        self._update(worker.alias, worker_id=worker_id)

    def grpc_ready(self) -> None:
        """Start the grace restored workers have to register again."""
        self._grace_deadline = time.monotonic() + WORKER_RECONNECT_GRACE_SEC

    def on_heartbeat(self) -> None:
        """Retry what the store or a provider left unfinished; called on the
        heartbeat thread."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._settle_records)

    def _settle_records(self) -> None:
        try:
            self._settle()
        except Exception:
            self.logger.exception("Failed to settle worker records")

    def _settle(self) -> None:
        for alias in list(self._unsaved):
            self._update(alias, strict=False)
        deadline = self._grace_deadline
        expired = deadline is not None and time.monotonic() >= deadline
        for alias, record in list(self._records.items()):
            if alias in self._in_flight:
                continue
            if record.state is RecordState.REMOVING:
                self._spawn(alias, self._finish_removal(alias))
            elif (
                alias in self._awaiting
                and expired
                and record.run_state is RunState.RUNNING
            ):
                self._spawn(alias, self._expire(alias))
            elif record.handle is not None and record.run_state is RunState.STOPPED:
                if (worker := self._registry.try_get_by_alias(alias)) is not None:
                    self._spawn(alias, self._stop_worker(worker))

    def _spawn(self, alias: str, work: Coroutine[Any, Any, Any]) -> None:
        self._enter(alias)
        task = asyncio.ensure_future(work)
        self._tasks.add(task)

        def done(_: asyncio.Task[Any]) -> None:
            self._leave(alias)
            self._tasks.discard(task)
            if not task.cancelled() and (exc := task.exception()) is not None:
                self.logger.error("Failed to settle worker %s: %r", alias, exc)

        task.add_done_callback(done)

    @contextmanager
    def _busy(self, alias: str) -> Iterator[None]:
        """Mark an operator's lifecycle operation on ``alias``, which ends the grace
        of a restored worker and keeps the heartbeat off it."""
        self._awaiting.discard(alias)
        self._enter(alias)
        try:
            yield
        finally:
            self._leave(alias)

    def _enter(self, alias: str) -> None:
        self._in_flight[alias] += 1

    def _leave(self, alias: str) -> None:
        self._in_flight[alias] -= 1
        if self._in_flight[alias] <= 0:
            del self._in_flight[alias]

    async def _expire(self, alias: str) -> None:
        # A registration may have landed since the heartbeat scheduled this.
        if alias not in self._awaiting:
            return
        self.logger.warning(
            "Worker %s did not register again within %.0fs; removing it",
            alias,
            WORKER_RECONNECT_GRACE_SEC,
        )
        self._update(alias, state=RecordState.REMOVING)
        self._awaiting.discard(alias)
        if (worker := self._registry.try_get_by_alias(alias)) is not None:
            worker.close()
            self._registry.discard(worker)
            self._removing[alias] = worker
        await self._finish_removal(alias)

    async def _finish_removal(self, alias: str) -> None:
        """Remove what a removing worker's record names, then the record."""
        record = self._records[alias]
        outcome = Removal.ABSENT
        if record.handle is not None:
            spec = self._providers.get(record.provider)
            if spec is None:
                outcome = Removal.UNKNOWN
            else:
                outcome = await asyncio.to_thread(spec.factory.remove, record.handle)
        if outcome is Removal.UNKNOWN:
            self.logger.warning(
                "Could not confirm the removal of worker %s; retrying", alias
            )
            return
        self._confirm_removal(alias)

    def _confirm_removal(self, alias: str, worker: WorkerAdapter | None = None) -> None:
        """Release a worker whose container or instance is gone, and forget it."""
        worker = self._removing.pop(alias, None) or worker
        self._awaiting.discard(alias)
        if worker is not None:
            self._destroy_worker(worker)
            self._report_capacity_change()
        with self._records_lock:
            try:
                self._store.delete(alias)
            except Exception as exc:
                self.logger.warning(
                    "Failed to delete the record of worker %s: %s", alias, exc
                )
                return
            self._records.pop(alias, None)
            self._unsaved.discard(alias)

    async def _start_worker(self, worker: WorkerAdapter) -> bool:
        if not self.is_started:
            raise ManagerNotStartedError()
        if worker.closed:
            raise ValueError(f"Worker '{worker.alias}' is being destroyed")
        if worker.status is not WorkerStatus.STOPPED or await worker.runs_held_worker():
            raise ValueError(
                f"Worker '{worker.alias}' is starting, running or stopping"
            )

        alias = worker.alias
        with self._busy(alias):
            self._update(
                alias, run_state=RunState.RUNNING, state=RecordState.PROVISIONING
            )
            # A cancelled start leaves its launch to commit its own handle, so only a
            # launch that ended marks the record as holding none.
            try:
                started = await worker.start()
            except Exception:
                self._launch_ended(worker)
                raise
            self._launch_ended(worker)
        if not started:
            self.logger.error("Worker %s failed to start", alias)
            return False
        return True

    def _launch_ended(self, worker: WorkerAdapter) -> None:
        if worker.handle() is None:
            self._update(worker.alias, strict=False, state=RecordState.PRESENT)

    def _destroy_worker(self, worker: WorkerAdapter) -> None:
        if worker in self._destroyed:
            return
        self._destroyed.add(worker)
        self._factory_for(worker).destroy_worker(worker)

    def _factory_for(self, worker: WorkerAdapter) -> WorkerFactory:
        for spec in self._providers.values():
            if isinstance(worker, spec.adapter_cls):
                return spec.factory
        raise ValueError(f"Unsupported worker type: {type(worker)}")

    def _report_capacity_change(self) -> None:
        callback = self._capacity_change_callback
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self.logger.debug("Failed to report capacity change: %s", exc)

    async def _stop_and_destroy_workers(self, workers: list[WorkerAdapter]) -> None:
        if not workers:
            return

        max_workers = min(len(workers), _MAX_PARALLELISM)
        sema = asyncio.Semaphore(max_workers or 1)

        async def stop_and_destroy(worker: WorkerAdapter) -> None:
            async with sema:
                try:
                    await self._stop_and_destroy_worker(worker)
                except Exception as exc:
                    # Its record stays, so the next start takes the worker back.
                    self.logger.error(
                        "Failed to destroy worker %s: %s", worker.alias, repr(exc)
                    )

        await asyncio.gather(*(stop_and_destroy(worker) for worker in workers))

    async def _stop_and_destroy_worker(self, worker: WorkerAdapter) -> bool:
        worker_alias = worker.alias
        recorded = worker_alias in self._records and worker not in self._destroyed
        with self._busy(worker_alias):
            if recorded:
                self._update(worker_alias, state=RecordState.REMOVING)
            return await self._destroy(worker, recorded)

    async def _destroy(self, worker: WorkerAdapter, recorded: bool) -> bool:
        worker_alias = worker.alias
        was_running = _is_live(worker)
        if was_running:
            self.logger.info("Stopping worker %s...", worker_alias)
        else:
            self.logger.info("Destroying worker %s that is not running.", worker_alias)
        # Closed first, so a start queued behind the stop creates nothing the destroy
        # would not remove.
        worker.close()
        # A worker mid-stop runs until that stop ends, whatever its status reads, so
        # a destroy joins the stop.
        try:
            success = await worker.stop()
        except Exception as exc:
            self.logger.error("Failed to stop worker %s: %s", worker_alias, repr(exc))
            success = False

        if recorded:
            if not success:
                # The heartbeat removes what the record names, then the record.
                self._removing[worker_alias] = worker
                return False
            self._confirm_removal(worker_alias, worker)

        try:
            self._destroy_worker(worker)
        except Exception as exc:
            self.logger.error(
                "Failed to destroy worker %s: %s", worker_alias, repr(exc)
            )
            success = False

        if success:
            outcome = "stopped" if was_running else "destroyed"
            self.logger.info("Worker %s %s.", worker_alias, outcome)

        return success

    async def _stop_worker(self, worker: WorkerAdapter) -> bool:
        worker_alias = worker.alias
        # A start queued behind another operation sets no status until it begins, and
        # the stop joins the chain behind it.
        if not (_is_live(worker) or worker.has_pending_start()):
            raise ValueError(f"Worker '{worker_alias}' is not starting or running")

        with self._busy(worker_alias):
            self._update(worker_alias, run_state=RunState.STOPPED)
            self.logger.info("Stopping worker %s...", worker_alias)
            try:
                success = await worker.stop()
            except Exception as exc:
                self.logger.error(
                    "Failed to stop worker %s: %s", worker_alias, repr(exc)
                )
                return False
            if not success:
                self.logger.error("Failed to stop worker %s", worker_alias)
                return False
            self._update(worker_alias, strict=False, handle=None)
            if not worker.has_event_stream:
                # A worker with no event stream open sends nothing that would mark
                # it stopped, so it is marked here and can be started again.
                worker.set_status(WorkerStatus.STOPPED)
            self.logger.info("Worker %s stopped.", worker_alias)
            return True
