import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from docker import DockerClient
from docker.errors import APIError, NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest
from pydantic import Field

from shared.utils.docker import sanitize_container_name

from ... import env
from ...hooks import PrincipalContext
from ...utils.helpers import get_docker_client
from ..provisioning import DockerHandle, ProviderHandle, Removal, WorkerRecord
from ..resource_manager import GpuArch, ResourceManager
from ..schemas import WorkerHardware, WorkerInfo, WorkerStatus
from .base import (
    ProviderSpec,
    WorkerAdapter,
    WorkerConfig,
    WorkerFactory,
    WorkerTokenType,
)
from .ssh import SSHConfig
from .utils import get_worker_image_name

_STOP_TIMEOUT = 30  # seconds
_REMOVAL_IN_PROGRESS_TIMEOUT = 60  # seconds
_REMOVAL_IN_PROGRESS_POLL = 1.0  # seconds
_PROVIDER_NAME = "docker"
# vLLM's multi-GPU engines exchange messages through /dev/shm and refuse to start when
# Docker's 64 MiB default cannot hold their ring buffers. tmpfs takes memory only as it
# fills, so the size is a ceiling.
_GPU_WORKER_SHM_SIZE = "8g"
_SSH_OWNER_LABEL = "flowmesh.ssh.worker_id"
_SSH_MANAGED_LABEL = "flowmesh.ssh.managed"
_ssh_network_suffix = sanitize_container_name(env.NODE_ALIAS, maxlen=32)
_SSH_NETWORK_NAME = f"flowmesh_ssh_{_ssh_network_suffix or 'default'}"

logger = logging.getLogger("supervisor")


def _container_token(container: Container) -> str | None:
    for entry in (container.attrs.get("Config") or {}).get("Env") or []:
        name, _, value = entry.partition("=")
        if name == "WORKER_TOKEN":
            return value
    return None


def _remove_ssh_resources(client: DockerClient, container_name: str) -> None:
    """Remove the SSH session containers and staging volumes of a worker container."""
    # A volume is in use until the container mounting it is removed.
    _remove_ssh_containers(client, container_name)
    _remove_ssh_volumes(client, container_name)


def _remove_ssh_containers(client: DockerClient, container_name: str) -> None:
    try:
        containers = client.containers.list(
            all=True, filters={"label": f"{_SSH_OWNER_LABEL}={container_name}"}
        )
    except Exception as exc:
        logger.warning(
            "Failed to list SSH session containers for worker %s: %s",
            container_name,
            repr(exc),
        )
        return

    # The worker is stopped or gone, so its SSH containers are killed outright: a
    # staging container's shell ignores SIGTERM.
    for ssh_container in containers:
        try:
            ssh_container.remove(force=True)
        except Exception as exc:
            logger.warning(
                "Failed to remove SSH session container %s: %s",
                ssh_container.name,
                repr(exc),
            )


def _remove_ssh_volumes(client: DockerClient, container_name: str) -> None:
    try:
        volumes = client.volumes.list(
            filters={
                "label": [
                    f"{_SSH_OWNER_LABEL}={container_name}",
                    f"{_SSH_MANAGED_LABEL}=true",
                ]
            }
        )
    except Exception as exc:
        logger.warning(
            "Failed to list SSH staging volumes for worker %s: %s",
            container_name,
            repr(exc),
        )
        return

    for volume in volumes:
        try:
            volume.remove(force=True)
        except Exception as exc:
            logger.warning(
                "Failed to remove SSH staging volume %s: %s",
                volume.name,
                repr(exc),
            )


def _is_removal_in_progress(exc: Exception) -> bool:
    """Whether Docker refused a remove because one is already under way.

    Matched on ``explanation`` as well as status: 409 also reports conflicts
    that do not resolve on their own, such as a name already in use.
    """
    return (
        isinstance(exc, APIError)
        and exc.status_code == 409
        and "already in progress" in (exc.explanation or "")
    )


class _VolumeInitializer:
    VOLUME_INIT_IMAGE = "busybox:1.36.1"
    WORKER_UID = 10001
    WORKER_GID = 10001

    _locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()
    _initialized: set[str] = set()

    @classmethod
    def ensure(cls, client: DockerClient, volume_name: str, mount_path: str) -> None:
        if not volume_name:
            return
        with cls._locks_lock:
            lock = cls._locks.get(volume_name)
            if lock is None:
                lock = threading.Lock()
                cls._locks[volume_name] = lock
        with lock:
            if volume_name in cls._initialized:
                return
            cls._prepare_volume(client, volume_name, mount_path)
            cls._initialized.add(volume_name)

    @classmethod
    def _prepare_volume(
        cls, client: DockerClient, volume_name: str, mount_path: str
    ) -> None:
        try:
            client.containers.run(
                image=cls.VOLUME_INIT_IMAGE,
                command=[
                    "sh",
                    "-c",
                    (
                        f"mkdir -p {mount_path} && "
                        f"chown -R {cls.WORKER_UID}:{cls.WORKER_GID} {mount_path}"
                    ),
                ],
                volumes={volume_name: {"bind": mount_path, "mode": "rw"}},
                remove=True,
            )
        except Exception as exc:
            logger.warning(
                (
                    "Failed to prepare Docker volume %s: %s. "
                    "Workers may lack write access."
                ),
                volume_name,
                repr(exc),
            )


class WorkerType(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class DockerWorkerConfig(WorkerConfig):
    container_name: str | None = None
    """Optional Docker container name"""
    worker_type: WorkerType = WorkerType.CPU
    """Type of worker (cpu or gpu)"""
    cuda_devices: list[int] | None = None
    """List of CUDA devices to use (if any)"""
    gpu_count: int = 1
    """Number of GPUs to auto-pick when ``cuda_devices`` is unset
    (only consulted for GPU workers)."""
    docker_registry: str = env.FLOWMESH_REGISTRY
    """Docker registry to pull worker images from"""
    version: str = env.FLOWMESH_VERSION
    """Worker Docker image version tag"""
    enable_ssh: bool = env.ENABLE_SSH_BY_DEFAULT
    """Whether to enable support for SSH jobs"""
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    """Default SSH session configuration"""

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        if self.worker_type == WorkerType.GPU and (
            isinstance(self.cuda_devices, list) and len(self.cuda_devices) == 0
        ):
            raise ValueError("Expected at least one CUDA device for GPU worker.")


class DockerWorkerInfo(WorkerInfo):
    pass


class DockerWorkerAdapter(WorkerAdapter):
    CONTAINER_RESULTS_DIR: str = "/var/lib/flowmesh-results"
    CONTAINER_HF_CACHE_DIR: str = "/home/appuser/.cache/huggingface"
    HF_CACHE_VOLUME: str | None = "flowmesh_server_hf_cache"
    DOCKER_SOCKET_PATH: str = "/var/run/docker.sock"

    def __init__(
        self,
        token: WorkerTokenType,
        alias: str,
        container_name: str,
        cuda_devices: list[int] | None,
        gpu_arch: GpuArch | None,
        config: DockerWorkerConfig,
        docker_client: DockerClient,
        owner: PrincipalContext,
        held_gpus: list[int] | None = None,
        handle: DockerHandle | None = None,
    ) -> None:
        if config.worker_type == WorkerType.GPU and (
            cuda_devices is None or len(cuda_devices) == 0
        ):
            raise ValueError("Expected at least one CUDA device for GPU worker.")

        super().__init__(token, alias, config, owner)

        self.config: DockerWorkerConfig
        self.container_name = container_name
        self.cuda_devices = cuda_devices
        self.gpu_arch = gpu_arch
        # The devices this adapter holds in the resource manager.
        self.held_gpus = held_gpus

        self._docker = docker_client
        self._status: WorkerStatus = WorkerStatus.STOPPED
        self._hardware: dict[str, Any] | WorkerHardware | None = None
        self._container_id = handle.container_id if handle else None

    @property
    def status(self) -> WorkerStatus:
        return self._status

    def set_status(self, status: WorkerStatus) -> None:
        self._status = status

    def get_info(self) -> DockerWorkerInfo:
        hardware = self._hardware
        if isinstance(hardware, dict):
            hardware = WorkerHardware.model_validate(hardware)
            self._hardware = hardware
        return DockerWorkerInfo(
            id=self.worker_id,
            alias=self.alias,
            provider=_PROVIDER_NAME,
            status=self.status,
            hardware=hardware,
            held_gpus=(self.held_gpus or []).copy(),
            ssh_limits=self.config.ssh.to_limits() if self.config.enable_ssh else None,
        )

    async def prepare(self) -> None:
        self._hardware = await asyncio.to_thread(self._probe_hardware)

    def observe_reported_hardware(self, hardware: WorkerHardware) -> None:
        if self._hardware is None:
            self._hardware = hardware

    def handle(self) -> DockerHandle | None:
        if self._container_id is None:
            return None
        return DockerHandle(
            container_id=self._container_id, container_name=self.container_name
        )

    def recover_launch(self) -> bool | None:
        try:
            container = self._docker.containers.get(self.container_name)
        except NotFound:
            return False
        except Exception as exc:
            logger.warning(
                "Failed to inspect Docker container %s: %s", self.container_name, exc
            )
            return None
        if _container_token(container) != self.token:
            logger.error(
                "Container %s does not run worker %s; leaving it alone",
                self.container_name,
                self.alias,
            )
            return None
        self._container_id = container.id
        self._report_handle()
        return True

    def get_image_name(self) -> str:
        return get_worker_image_name(
            self.config.docker_registry, self.config.version, self.gpu_arch
        )

    def _remove_stale_container(self, container: Container) -> bool:
        try:
            container.remove(force=True)
        except NotFound:
            return True
        except Exception as exc:
            if _is_removal_in_progress(exc):
                logger.info(
                    "Container %s is already being removed; waiting for it to go",
                    self.container_name,
                )
                return self._wait_container_gone()
            logger.error(
                "Failed to remove stale container %s: %s", self.container_name, exc
            )
            return False
        logger.debug("Removed stale container %s", self.container_name)
        return True

    def _wait_container_gone(self) -> bool:
        """Block until this adapter's container no longer exists, or time out.

        Only ``NotFound`` confirms removal. An inspect that fails any other way
        leaves the outcome unknown, so the wait runs on to the deadline.
        """
        deadline = time.monotonic() + _REMOVAL_IN_PROGRESS_TIMEOUT
        inspect_error: Exception | None = None
        while True:
            try:
                self._docker.containers.get(self.container_name)
                inspect_error = None
            except NotFound:
                return True
            except Exception as exc:
                inspect_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(_REMOVAL_IN_PROGRESS_POLL)

        if inspect_error is None:
            logger.error(
                "Stale container %s is still present %ss after its removal "
                "was reported in progress",
                self.container_name,
                _REMOVAL_IN_PROGRESS_TIMEOUT,
            )
        else:
            logger.error(
                "Could not confirm removal of stale container %s within %ss: %s",
                self.container_name,
                _REMOVAL_IN_PROGRESS_TIMEOUT,
                inspect_error,
            )
        return False

    def _start(self) -> bool:
        existing: Container | None = None
        try:
            existing = self._docker.containers.get(self.container_name)
        except NotFound:
            pass
        except Exception as exc:
            logger.warning(
                "Failed to inspect Docker container %s: %s",
                self.container_name,
                exc,
            )
            return False

        if existing is not None:
            if existing.id != self._container_id:
                logger.error(
                    "Container %s is not worker %s's; remove it to start the worker",
                    self.container_name,
                    self.alias,
                )
                return False
            if existing.status == "running":
                logger.warning("Container %s is already running.", self.container_name)
                return True
            if not self._remove_stale_container(existing):
                return False
            self._container_id = None
            self._report_handle()

        environment: dict[str, str] = self._base_environment()
        labels: dict[str, str] = self._base_labels()
        volumes: list[str] = []
        self._mount_results(volumes)
        self._mount_hf_cache(volumes)
        self._mount_docker_socket(volumes)
        device_requests, runtime = self._apply_worker_type_settings(environment, labels)
        docker_gid = self._get_docker_socket_gid()

        try:
            run_kwargs: dict[str, Any] = {
                "image": self.get_image_name(),
                "name": self.container_name,
                "environment": environment,
                "labels": labels,
                "volumes": volumes,
                "network_mode": "host",
                "device_requests": device_requests,
                "restart_policy": {"Name": "unless-stopped"},
                "detach": True,
            }
            if runtime is not None:
                run_kwargs["runtime"] = runtime
            if self.config.worker_type is WorkerType.GPU:
                run_kwargs["shm_size"] = _GPU_WORKER_SHM_SIZE
            if docker_gid:
                run_kwargs["group_add"] = [docker_gid]
            container = self._docker.containers.run(**run_kwargs)
        except Exception as exc:
            logger.error(
                "Failed to start Docker container %s: %s",
                self.container_name,
                exc,
            )
            # A launch that raised may have created its container.
            self.recover_launch()
            return False
        self._container_id = container.id
        self._report_handle()

        if self._hardware is None:
            self._hardware = self._probe_hardware()

        return True

    def _probe_hardware(self) -> dict[str, Any] | None:
        logger.debug("Collecting hardware info for worker %s", self.container_name)
        container = self._get_running_container()
        output_prefix = "HW_PROBE_OUTPUT: "
        cmd = self._hardware_probe_cmd(output_prefix)
        output: bytes | None
        if container is None:
            # Probe hardware in a temporary container
            environment: dict[str, str] = self._base_environment()
            device_requests, runtime = self._apply_worker_type_settings(
                environment, None
            )
            try:
                run_kwargs: dict[str, Any] = {
                    "image": self.get_image_name(),
                    "command": cmd,
                    "environment": environment,
                    "network_mode": "host",
                    "device_requests": device_requests,
                    "remove": True,
                    "stdout": True,
                    "stderr": True,
                }
                if runtime is not None:
                    run_kwargs["runtime"] = runtime
                output = self._docker.containers.run(**run_kwargs)
            except Exception as exc:
                logger.warning(
                    "Failed to run hardware probe for worker %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return None
        else:
            # Probe hardware in the existing container
            try:
                result = container.exec_run(cmd, stdout=True, stderr=True)
            except Exception as exc:
                logger.warning(
                    "Failed to exec hardware probe for worker %s: %s",
                    self.container_name,
                    repr(exc),
                )
                return None
            exit_code = getattr(result, "exit_code", None)
            output = getattr(result, "output", None)
            if exit_code not in (0, None):
                logger.warning(
                    "Hardware probe exec failed for %s with exit code %s",
                    self.container_name,
                    exit_code,
                )
            if output is None:
                return None

        return self._parse_hardware_output(output, output_prefix)

    def holds_worker(self) -> bool:
        return self._container_id is not None

    def _held_worker_runs(self) -> bool:
        return self._get_running_container() is not None

    def _stop(self) -> bool:
        container_id = self._container_id
        if container_id is None:
            _remove_ssh_resources(self._docker, self.container_name)
            return True
        try:
            container = self._docker.containers.get(container_id)
        except NotFound:
            self._container_id = None
            logger.warning("Container %s not found.", self.container_name)
            _remove_ssh_resources(self._docker, self.container_name)
            return True
        except Exception as exc:
            self._log_failure("fetch", exc)
            return False

        # The worker stops first, so its shutdown gives up the SSH tasks it runs before
        # their containers go; what it leaves behind is removed after. A worker that
        # fails to stop keeps them. A container gone mid-stop has stopped.
        try:
            container.stop(timeout=_STOP_TIMEOUT)
        except NotFound:
            pass
        except Exception as exc:
            self._log_failure("stop", exc)
            return False
        try:
            container.remove()
        except NotFound:
            pass
        except Exception as exc:
            self._log_failure("remove", exc)
            return False
        finally:
            _remove_ssh_resources(self._docker, self.container_name)
        self._container_id = None
        return True

    def _log_failure(self, action: str, exc: Exception) -> None:
        logger.error(
            "Failed to %s Docker container %s: %s",
            action,
            self.container_name,
            repr(exc),
        )

    def _base_environment(self) -> dict[str, str]:
        environment = super()._base_environment()
        environment["RESULTS_DIR"] = self.CONTAINER_RESULTS_DIR
        environment["RESULTS_MOUNT_SOURCE"] = self.config.results_dir
        environment["FLOWMESH_REGISTRY"] = self.config.docker_registry
        environment["FLOWMESH_VERSION"] = self.config.version
        environment["WORKER_NETWORK_MODE"] = f"container:{self.container_name}"
        environment["WORKER_CONTAINER_NAME"] = self.container_name
        environment["SSH_NETWORK_NAME"] = _SSH_NETWORK_NAME
        environment.update(self.config.ssh.to_env(self.config.enable_ssh))
        return environment

    def _apply_worker_type_settings(
        self,
        environment: dict[str, str],
        labels: dict[str, str] | None,
    ) -> tuple[list[DeviceRequest] | None, str | None]:
        """Apply worker type specific settings to environment and labels.

        Returns:
            device_requests: list[DeviceRequest] | None
                Device requests for Docker container.
            runtime: str | None
                Runtime to use for Docker container.
        """
        device_requests: list[DeviceRequest] | None
        runtime: str | None
        match self.config.worker_type:
            case WorkerType.CPU:
                if labels is not None:
                    labels["flowmesh.worker.type"] = "cpu"
                device_requests = None
                runtime = None
            case WorkerType.GPU:
                assert self.cuda_devices is not None
                assert self.gpu_arch is not None
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in range(len(self.cuda_devices))
                )
                cuda_devices_str = [str(i) for i in self.cuda_devices]
                gpu_ids = ",".join(cuda_devices_str)
                environment["WORKER_HOST_GPU_ID"] = gpu_ids
                gpu_arch = self.gpu_arch.value
                environment["WORKER_HOST_GPU_ARCH"] = gpu_arch
                if labels is not None:
                    labels["flowmesh.worker.type"] = "gpu"
                    labels["flowmesh.worker.gpu_id"] = gpu_ids
                    labels["flowmesh.worker.gpu_arch"] = gpu_arch
                device_requests = [
                    DeviceRequest(device_ids=cuda_devices_str, capabilities=[["gpu"]])
                ]
                runtime = env.DOCKER_GPU_RUNTIME
            case _:
                raise ValueError(f"Unsupported worker type: {self.config.worker_type}")
        return device_requests, runtime

    def _base_labels(self) -> dict[str, str]:
        return {
            "flowmesh.role": "worker",
            "flowmesh.group": "server-workers",
        }

    def _ensure_volume_access(self, source: str, container_path: str) -> None:
        if not source or os.path.isabs(source):
            return
        _VolumeInitializer.ensure(self._docker, source, container_path)

    def _mount_results(self, volumes: list[str]) -> None:
        source = self.config.results_dir
        container_results_dir = self.CONTAINER_RESULTS_DIR
        self._ensure_volume_access(source, container_results_dir)
        result_mnt = f"{source}:{container_results_dir}"
        volumes.append(result_mnt)

    def _mount_hf_cache(self, volumes: list[str]) -> None:
        hf_cache_dir = self.config.hf_cache_dir
        container_cache_dir = self.CONTAINER_HF_CACHE_DIR
        if hf_cache_dir is not None:
            self._ensure_volume_access(hf_cache_dir, container_cache_dir)
            volumes.append(f"{hf_cache_dir}:{container_cache_dir}")
            return

        hf_cache_volume = self.HF_CACHE_VOLUME
        if hf_cache_volume is not None:
            # Create the volume if it doesn't exist
            found = self._docker.volumes.list(filters={"name": hf_cache_volume})
            if not found:
                self._docker.volumes.create(name=hf_cache_volume)
            self._ensure_volume_access(hf_cache_volume, container_cache_dir)
            volumes.append(f"{hf_cache_volume}:{container_cache_dir}")
            return

    @property
    def _needs_docker_socket(self) -> bool:
        return self.config.enable_ssh and self.config.ssh.uses_docker

    def _mount_docker_socket(self, volumes: list[str]) -> None:
        if self._needs_docker_socket:
            volumes.append(f"{self.DOCKER_SOCKET_PATH}:{self.DOCKER_SOCKET_PATH}")

    def _get_docker_socket_gid(self) -> int | None:
        if not self._needs_docker_socket:
            return None
        try:
            gid = os.stat(self.DOCKER_SOCKET_PATH).st_gid
        except OSError as exc:
            logger.warning(
                "Failed to inspect Docker socket %s for worker %s: %s",
                self.DOCKER_SOCKET_PATH,
                self.container_name,
                repr(exc),
            )
            return None
        return gid

    def _hardware_probe_cmd(self, prefix: str | None) -> list[str]:
        cmd = ["python", "-m", "worker.main", "--collect-hw"]
        bandwidth = self.config.network_bandwidth
        if bandwidth is not None:
            cmd.extend(["--bandwidth-bytes-per-sec", str(bandwidth)])
        if prefix:
            cmd.extend(["--collect-hw-prefix", prefix])
        return cmd

    def _get_running_container(self) -> Container | None:
        if self._container_id is None:
            return None
        try:
            container = self._docker.containers.get(self._container_id)
        except NotFound:
            return None
        except Exception as exc:
            logger.warning(
                "Failed to inspect Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return None
        try:
            container.reload()
        except Exception as exc:
            logger.warning(
                "Failed to reload Docker container %s: %s",
                self.container_name,
                repr(exc),
            )
            return None
        return container if container.status == "running" else None

    def _parse_hardware_output(
        self, output: bytes | str | None, prefix: str | None
    ) -> dict[str, Any] | None:
        if output is None:
            return None
        if isinstance(output, (bytes, bytearray)):
            output_text = output.decode("utf-8", errors="replace").strip()
        else:
            output_text = str(output).strip()
        if not output_text:
            logger.warning(
                "Hardware probe returned no output for %s", self.container_name
            )
            return None
        if prefix:
            # Find the output line and strip the prefix
            for line in output_text.splitlines():
                if line.startswith(prefix):
                    output_text = line.removeprefix(prefix)
                    break
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid hardware probe output for %s: %s",
                self.container_name,
                repr(exc),
            )
            logger.debug("Hardware probe output was: %s", output_text)
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "Hardware probe output for %s is not a JSON object", self.container_name
            )
            return None
        return payload


class DockerWorkerFactory(WorkerFactory):
    _CONTAINER_NAME_MAX_LEN = 128
    _CONTAINER_NAME_ALLOWED_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

    def __init__(
        self, system_principal: PrincipalContext, alias_taken: Callable[[str], bool]
    ) -> None:
        super().__init__(system_principal)
        self._alias_taken = alias_taken
        self._rm = ResourceManager.get_instance()
        self._docker = get_docker_client()
        self._worker_id_registry: Counter[str] = Counter()

    def create_worker(
        self, token: WorkerTokenType, config: DockerWorkerConfig
    ) -> DockerWorkerAdapter:
        cuda_devices: list[int] | None
        gpu_arch: GpuArch | None
        match config.worker_type:
            case WorkerType.CPU:
                cuda_devices = gpu_arch = None
            case WorkerType.GPU:
                cuda_devices, gpu_arch = self._rm.reserve_gpus(
                    devices=config.cuda_devices,
                    n=config.gpu_count if config.cuda_devices is None else None,
                )

        alias = self._resolve_worker_alias(config)
        container_name = (
            config.container_name
            if config.container_name
            else self._sanitize_container_name(alias, config)
        )
        # The resolved name and devices are part of what a record keeps.
        config = config.model_copy(
            update={"container_name": container_name, "cuda_devices": cuda_devices}
        )
        return DockerWorkerAdapter(
            token=token,
            alias=alias,
            container_name=container_name,
            cuda_devices=cuda_devices,
            gpu_arch=gpu_arch,
            config=config,
            docker_client=self._docker,
            owner=self.system_principal,
            held_gpus=cuda_devices,
        )

    def attach(
        self, token: WorkerTokenType, record: WorkerRecord
    ) -> DockerWorkerAdapter:
        config = DockerWorkerConfig.model_validate(record.config)
        handle = record.handle
        if handle is not None and not isinstance(handle, DockerHandle):
            raise TypeError(f"Not a Docker handle: {handle!r}")
        if config.container_name is None:
            raise ValueError(f"the record of worker {record.alias} names no container")
        held: list[int] | None = None
        gpu_arch: GpuArch | None = None
        if config.cuda_devices:
            try:
                held, gpu_arch = self._rm.reserve_gpus(devices=config.cuda_devices)
            except ValueError as exc:
                logger.error(
                    "Could not hold GPUs %s of worker %s again: %s",
                    config.cuda_devices,
                    record.alias,
                    exc,
                )
        return DockerWorkerAdapter(
            token=token,
            alias=record.alias,
            container_name=config.container_name,
            cuda_devices=config.cuda_devices,
            gpu_arch=gpu_arch,
            config=config,
            docker_client=self._docker,
            owner=self.system_principal,
            held_gpus=held,
            handle=handle,
        )

    def remove(self, handle: ProviderHandle) -> Removal:
        if not isinstance(handle, DockerHandle):
            raise TypeError(f"Not a Docker handle: {handle!r}")
        try:
            self._docker.containers.get(handle.container_id).remove(force=True)
            outcome = Removal.REMOVED
        except NotFound:
            outcome = Removal.ABSENT
        except Exception as exc:
            logger.warning(
                "Failed to remove Docker container %s: %s",
                handle.container_name,
                repr(exc),
            )
            return Removal.UNKNOWN
        _remove_ssh_resources(self._docker, handle.container_name)
        return outcome

    def destroy_worker(self, worker: WorkerAdapter) -> None:
        if not isinstance(worker, DockerWorkerAdapter):
            raise ValueError("Invalid worker type")

        # Taken, not read: a repeated destroy must not drop another holder's hold.
        devices, worker.held_gpus = worker.held_gpus, None
        worker.cuda_devices = None
        if devices:
            self._rm.deallocate_gpus(devices)

    def cleanup(self) -> None:
        self._remove_ssh_network()

    def _remove_ssh_network(self) -> None:
        network_name = _SSH_NETWORK_NAME
        try:
            existing = self._docker.networks.list(names=[network_name])
            for net in existing:
                labels = net.attrs.get("Labels") or {}
                if (
                    net.name == network_name
                    and labels.get(_SSH_MANAGED_LABEL) == "true"
                ):
                    net.remove()
                    logger.info("Removed SSH network %s", network_name)
        except Exception:
            pass

    def _get_next_worker_id(self, prefix: str) -> int:
        containers: list[Container] = self._docker.containers.list(
            all=True, filters={"name": prefix}
        )
        max_id = -1
        for container in containers:
            name = container.name
            assert isinstance(name, str)
            try:
                cur_id = int(name.rsplit("_", maxsplit=1)[-1])
            except ValueError:
                continue
            if cur_id > max_id:
                max_id = cur_id

        registered_id = self._worker_id_registry[prefix]
        if registered_id > max_id:
            container_id = registered_id
        else:
            container_id = max_id + 1
        self._worker_id_registry[prefix] = container_id + 1
        return container_id

    def _resolve_worker_alias(self, config: DockerWorkerConfig) -> str:
        return config.worker_alias or self._get_next_worker_alias(config.worker_type)

    def _sanitize_container_name(self, value: str, config: DockerWorkerConfig) -> str:
        raw = str(value or "").strip()
        if not raw:
            return self._get_next_worker_alias(config.worker_type)
        sanitized = sanitize_container_name(raw, self._CONTAINER_NAME_MAX_LEN)
        if not sanitized or not self._CONTAINER_NAME_ALLOWED_RE.fullmatch(sanitized):
            return self._get_next_worker_alias(config.worker_type)
        return sanitized

    def _get_next_worker_alias(self, worker_type: WorkerType) -> str:
        match worker_type:
            case WorkerType.CPU:
                prefix = "flowmesh_server_worker_cpu_"
            case WorkerType.GPU:
                prefix = "flowmesh_server_worker_gpu_"
            case _:
                raise ValueError(f"Unsupported worker type: {worker_type}")
        while True:
            alias = f"{prefix}{self._get_next_worker_id(prefix)}"
            if not self._alias_taken(alias):
                return alias


def get_provider_spec(
    system_principal: PrincipalContext, alias_taken: Callable[[str], bool]
) -> ProviderSpec:
    return ProviderSpec(
        name=_PROVIDER_NAME,
        config_cls=DockerWorkerConfig,
        adapter_cls=DockerWorkerAdapter,
        factory=DockerWorkerFactory(system_principal, alias_taken),
    )
