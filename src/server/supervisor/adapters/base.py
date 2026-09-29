import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType

from pydantic import BaseModel, ConfigDict, SecretStr

from ... import env
from ...hooks import PrincipalContext
from ..schemas import WorkerInfo, WorkerStatus
from .utils import env_to_secret_str, to_env_str

logger = logging.getLogger("supervisor")


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    supervisor_grpc_target: str = f"{env.SERVER_LOCAL_HOST}:{env.SERVER_GRPC_PORT}"
    """Supervisor gRPC target"""
    results_dir: str = env.WORKER_RESULTS_DIR
    """Root directory for task outputs"""
    hb_interval: int = env.SERVER_HEARTBEAT_INTERVAL
    """Interval between heartbeats in seconds"""
    worker_alias: str | None = None
    """Optional worker alias"""
    tags: str = env.WORKER_TAGS
    """Comma-separated tags used by the scheduler"""
    hb_file: str | None = None
    """Path to the worker heartbeat file."""
    log_level: str = env.LOG_LEVEL
    """Logging level for the worker"""
    flowmesh_url: str = env.FLOWMESH_BASE_URL
    """FlowMesh HTTP base URL required to build artifact download links"""
    worker_cost_per_hour: float | None = None
    """Hourly cost in USD"""
    model_archive_use_pigz: bool | None = None
    """Whether to use pigz for model archive compression"""
    model_archive_compression_level: int | None = None
    """Gzip compression level (0-9)"""
    model_archive_pigz_threads: int | None = None
    """Number of threads for pigz compression"""
    model_archive_pigz_bin: str | None = None
    """Path to pigz binary"""
    model_archive_tar_bin: str | None = None
    """Path to tar binary"""
    network_bandwidth: float | None = None
    """Bandwidth in bytes per second to throttle HTTP uploads"""
    openai_api_key: SecretStr | None = env_to_secret_str("OPENAI_API_KEY")
    """OpenAI API key"""
    google_api_key: SecretStr | None = env_to_secret_str("GOOGLE_API_KEY")
    """Google API key"""
    hf_token: SecretStr | None = env_to_secret_str("HF_TOKEN")
    """Hugging Face API token"""
    hf_cache_dir: str | None = env.HF_CACHE_DIR
    """Hugging Face cache directory"""
    predownload_model_list: str = env.PREDOWNLOAD_MODEL_LIST
    """Comma-separated list of models to pre-download during worker startup"""
    nebula_api_token: SecretStr | None = env_to_secret_str("NEBULA_API_TOKEN")
    """Nebula API token"""
    upload_results: bool = env.WORKER_UPLOAD_RESULTS
    """Whether to always upload results to the server if spec.output.destination
    is unspecified."""
    executor_idle_cleanup_sec: float = env.WORKER_EXECUTOR_IDLE_CLEANUP_SEC
    """Seconds an executor may sit idle before the worker unloads it"""
    enable_dev_model: bool = env.WORKER_ENABLE_DEV_MODEL
    """Whether the worker advertises the GPU-free dev_model serving executor"""
    dev_model_forward_url: str = env.DEV_MODEL_FORWARD_URL
    """Upstream OpenAI-compatible base URL dev_model forwards to (empty = canned)"""
    dev_model_response_delay_sec: float = env.DEV_MODEL_RESPONSE_DELAY_SEC
    """Per-response delay the dev_model stand-in applies (a test seam; 0 = none)"""


WorkerTokenType = NewType("WorkerTokenType", str)


class WorkerAdapter(ABC):
    def __init__(
        self,
        token: WorkerTokenType,
        name: str,
        config: WorkerConfig,
        owner: PrincipalContext,
    ) -> None:
        self._worker_id: str | None = None
        self.token = token
        self.name = name
        self.config = config
        self.owner = owner
        self._starting: asyncio.Future[bool] | None = None
        self._stopping: asyncio.Future[bool] | None = None
        self._closed = False
        self._event_streams = 0

    @property
    @abstractmethod
    def status(self) -> WorkerStatus:
        pass

    @property
    def worker_id(self) -> str | None:
        return self._worker_id

    @abstractmethod
    def set_status(self, status: WorkerStatus) -> None:
        pass

    def set_worker_id(self, worker_id: str) -> None:
        if self._worker_id is not None:
            raise RuntimeError(f"Worker ID is already set to {self._worker_id}")
        self._worker_id = worker_id

    def clear_worker_id(self) -> None:
        if self._worker_id is None:
            raise RuntimeError("Worker ID is not set")
        self._worker_id = None

    @property
    def has_event_stream(self) -> bool:
        """Whether an event stream from the worker is open."""
        return self._event_streams > 0

    def attach_event_stream(self) -> None:
        self._event_streams += 1

    def detach_event_stream(self) -> None:
        self._event_streams -= 1

    @abstractmethod
    def get_info(self) -> WorkerInfo:
        pass

    async def start(self) -> bool:
        """Start worker. Returns whether the worker was successfully started.

        ``_start`` runs on a thread, which a cancel cannot stop; a stop waits for it
        first, so it finds whatever the start created. A start waits for every stop
        still running, so the stop never removes what the start creates, and a start
        while another runs waits for that one and returns its result. A start after
        :meth:`close` creates nothing, and a start behind a stop that left the worker
        running keeps that worker.
        """
        while (stopping := self._stopping) is not None and not stopping.done():
            await asyncio.wait({stopping})
        if self._closed:
            return False
        if (starting := self._starting) is None or starting.done():
            if self.holds_worker():
                return True
            self.set_status(WorkerStatus.STARTING)
            starting = self._starting = asyncio.ensure_future(
                asyncio.to_thread(self._start)
            )
        try:
            ok = await asyncio.shield(starting)
        except asyncio.CancelledError:
            starting.add_done_callback(self._on_abandoned_start)
            raise
        except Exception:
            self.set_status(WorkerStatus.STOPPED)
            raise
        if not ok:
            self.set_status(WorkerStatus.STOPPED)
        return ok

    def close(self) -> None:
        """Refuse every later start, as the adapter is about to be destroyed."""
        self._closed = True

    async def prepare(self) -> None:
        """Prepare worker (e.g., collecting hardware information) without starting
        it."""
        pass

    async def stop(self) -> bool:
        """Stop worker. Returns whether the worker was successfully stopped.

        ``_stop`` runs on a thread once a start still creating the worker finishes. A
        stop is in flight from the moment it is accepted, so a start after it waits
        for it. A stop while another runs waits for that one and returns its result,
        so no caller sees the worker stopped before it is; a cancel of a caller never
        stops the stop.
        """
        if (stopping := self._stopping) is None or stopping.done():
            stopping = self._stopping = asyncio.ensure_future(self._stop_after_start())
        return await asyncio.shield(stopping)

    async def _stop_after_start(self) -> bool:
        if (starting := self._starting) is not None and not starting.done():
            await asyncio.wait({starting})
        prev_status = self.status
        # The status reads STOPPED whenever the worker's event stream closes, so a
        # worker the adapter started is stopped whatever its status reads.
        if (
            prev_status in (WorkerStatus.STOPPING, WorkerStatus.STOPPED)
            and not self.holds_worker()
        ):
            return True
        self.set_status(WorkerStatus.STOPPING)
        return await self._stop_on_thread(prev_status)

    @abstractmethod
    def _start(self) -> bool:
        """Start the worker, blocking; returns whether it started."""
        pass

    @abstractmethod
    def _stop(self) -> bool:
        """Stop the worker, blocking; returns whether it stopped."""
        pass

    @abstractmethod
    def holds_worker(self) -> bool:
        """Whether this adapter started a worker it has not stopped."""
        pass

    def _on_abandoned_start(self, starting: asyncio.Future[bool]) -> None:
        if not starting.cancelled() and (exc := starting.exception()) is not None:
            logger.warning(
                "Worker %s failed to start after its start was cancelled: %r",
                self.name,
                exc,
            )
        if not self.holds_worker():
            self.set_status(WorkerStatus.STOPPED)

    async def _stop_on_thread(self, prev_status: WorkerStatus) -> bool:
        try:
            ok = await asyncio.to_thread(self._stop)
        except Exception:
            self.set_status(prev_status)
            raise
        if not ok:
            self.set_status(prev_status)
        return ok

    def _base_environment(self) -> dict[str, str]:
        config = self.config
        hb_file = config.hb_file or os.path.join(env.WORKER_HB_DIR, f"{self.token}.hb")
        return {
            "WORKER_TOKEN": self.token,  # type: ignore
            "SUPERVISOR_GRPC_TARGET": config.supervisor_grpc_target,
            "SUPERVISOR_GRPC_TLS_CA_B64": env.SERVER_GRPC_TLS_CA_B64,
            "NETWORK_PLANE_PEER_ENABLED": to_env_str(env.NETWORK_PLANE_PEER_ENABLED),
            "NETWORK_PLANE_PEER_DISABLE_MTLS": to_env_str(
                env.NETWORK_PLANE_PEER_DISABLE_MTLS
            ),
            "NETWORK_PLANE_PEER_TLS_CA_B64": env.NETWORK_PLANE_PEER_TLS_CA_B64,
            "NETWORK_PLANE_PEER_TLS_CERT_B64": env.NETWORK_PLANE_PEER_TLS_CERT_B64,
            "NETWORK_PLANE_PEER_TLS_KEY_B64": env.NETWORK_PLANE_PEER_TLS_KEY_B64,
            "RESULTS_DIR": config.results_dir,
            "WORKER_PRIVATE_STATE_DIR": env.WORKER_PRIVATE_STATE_DIR,
            "WORKER_CONTENT_DIR": env.WORKER_CONTENT_DIR,
            "CONTENT_HYDRATION_ENABLED": to_env_str(env.CONTENT_HYDRATION_ENABLED),
            "CONTENT_CACHE_TTL_SEC": to_env_str(env.CONTENT_CACHE_TTL_SEC),
            "CONTENT_CACHE_MAX_BYTES": to_env_str(env.CONTENT_CACHE_MAX_BYTES),
            # The store's address, but never its credential: a worker reaches content
            # with the access control mints for each task it runs.
            "CONTENT_STORE_BACKEND": env.CONTENT_STORE_BACKEND,
            "CONTENT_STORE_ENDPOINT_URL": env.CONTENT_STORE_ENDPOINT_URL,
            # The port too: with no endpoint named, both ends resolve the co-located
            # store's address themselves, and they have to resolve the same one.
            "CONTENT_STORE_PORT": env.CONTENT_STORE_PORT,
            "CONTENT_STORE_BUCKET": env.CONTENT_STORE_BUCKET,
            "CONTENT_STORE_PREFIX": env.CONTENT_STORE_PREFIX,
            "CONTENT_STORE_REGION": env.CONTENT_STORE_REGION,
            "CONTENT_STORE_FILESYSTEM_ROOT": env.CONTENT_STORE_FILESYSTEM_ROOT,
            "CONTENT_HOLDER_TTL_SEC": to_env_str(env.CONTENT_HOLDER_TTL_SEC),
            "CONTENT_TRANSFER_TIMEOUT_SEC": to_env_str(
                env.CONTENT_TRANSFER_TIMEOUT_SEC
            ),
            "HEARTBEAT_INTERVAL_SEC": to_env_str(config.hb_interval),
            "WORKER_HB_FILE": hb_file,
            "WORKER_NAMESPACE": env.NODE_NAMESPACE,
            "WORKER_CLUSTER": env.NODE_CLUSTER,
            "WORKER_ALIAS": config.worker_alias or "",
            "WORKER_TAGS": config.tags,
            "LOG_LEVEL": config.log_level,
            "WORKER_COST_PER_HOUR": to_env_str(config.worker_cost_per_hour),
            "FLOWMESH_BASE_URL": config.flowmesh_url,
            "MODEL_ARCHIVE_USE_PIGZ": to_env_str(config.model_archive_use_pigz),
            "MODEL_ARCHIVE_COMPRESSION_LEVEL": to_env_str(
                config.model_archive_compression_level
            ),
            "MODEL_ARCHIVE_PIGZ_THREADS": to_env_str(config.model_archive_pigz_threads),
            "MODEL_ARCHIVE_PIGZ_BIN": to_env_str(config.model_archive_pigz_bin),
            "MODEL_ARCHIVE_TAR_BIN": to_env_str(config.model_archive_tar_bin),
            "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC": to_env_str(
                config.network_bandwidth
            ),
            "WORKER_UPLOAD_RESULTS": to_env_str(config.upload_results),
            "WORKER_EXECUTOR_IDLE_CLEANUP_SEC": to_env_str(
                config.executor_idle_cleanup_sec
            ),
            "WORKER_ENABLE_DEV_MODEL": to_env_str(config.enable_dev_model),
            "DEV_MODEL_FORWARD_URL": config.dev_model_forward_url,
            "DEV_MODEL_RESPONSE_DELAY_SEC": to_env_str(
                config.dev_model_response_delay_sec
            ),
            "DOCKER_GPU_RUNTIME": to_env_str(env.DOCKER_GPU_RUNTIME),
            "FLOWMESH_API_KEY": to_env_str(env.FLOWMESH_API_KEY),
            "WORKER_OWNER_PRINCIPAL_JSON": self.owner.model_dump_json(),
            "OPENAI_API_KEY": to_env_str(config.openai_api_key),
            "GOOGLE_API_KEY": to_env_str(config.google_api_key),
            "WEB_SEARCH_PROVIDER": env.WEB_SEARCH_PROVIDER,
            "WEB_SEARCH_API_KEY": to_env_str(env.WEB_SEARCH_API_KEY),
            "AGENT_MODEL_API_KEY": to_env_str(env.AGENT_MODEL_API_KEY),
            "AGENT_MODEL_EGRESS_TIMEOUT_SEC": to_env_str(
                env.AGENT_MODEL_EGRESS_TIMEOUT_SEC
            ),
            "HF_TOKEN": to_env_str(config.hf_token),
            "PREDOWNLOAD_MODEL_LIST": config.predownload_model_list,
            "NEBULA_API_TOKEN": to_env_str(config.nebula_api_token),
            "NEBULA_API_BASE_URL": env.NEBULA_API_BASE_URL,
            "SERVER_METRICS_TELEMETRY_LEVEL": env.SERVER_METRICS_TELEMETRY_LEVEL,
            "SERVER_METRICS_TRACES_ENABLED": to_env_str(
                env.SERVER_METRICS_TRACES_ENABLED
            ),
            "SERVER_METRICS_METRICS_ENABLED": to_env_str(
                env.SERVER_METRICS_METRICS_ENABLED
            ),
            "SERVER_METRICS_TRACE_SAMPLE_RATIO": to_env_str(
                env.SERVER_METRICS_TRACE_SAMPLE_RATIO
            ),
            "SERVER_METRICS_OTLP_ENDPOINT": env.SERVER_METRICS_OTLP_ENDPOINT,
            "SERVER_METRICS_OTLP_TIMEOUT_SEC": to_env_str(
                env.SERVER_METRICS_OTLP_TIMEOUT_SEC
            ),
            "SERVER_METRICS_RESOURCE_SAMPLE_SEC": to_env_str(
                env.SERVER_METRICS_RESOURCE_SAMPLE_SEC
            ),
        }


class WorkerFactory(ABC):
    def __init__(self, system_principal: PrincipalContext) -> None:
        self.system_principal = system_principal

    @abstractmethod
    def create_worker(self, token: WorkerTokenType, *args, **kwargs) -> WorkerAdapter:
        pass

    @abstractmethod
    def destroy_worker(self, worker: WorkerAdapter) -> None:
        pass

    def cleanup(self) -> None:
        pass


@dataclass(frozen=True)
class ProviderSpec:
    """Per-provider dispatch entry consumed by `WorkerManager`.

    Each provider module (e.g. `adapters.docker`, `adapters.vastai`) exposes a
    `get_provider_spec(system_principal)` builder that returns one of these.
    """

    name: str
    config_cls: type[WorkerConfig]
    adapter_cls: type[WorkerAdapter]
    factory: WorkerFactory
