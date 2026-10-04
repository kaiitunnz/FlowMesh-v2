"""vLLM OpenAI-compatible serving executor.

Starts a persistent vLLM API server for a single model, emits a TASK_UPDATE
with the endpoint details, and blocks until the TTL expires or a stop command
arrives.
"""

import collections
import json
import logging
import os
import secrets
import signal
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any, NoReturn

import httpx

from shared.resident.contracts import LOCAL_ENGINE_ORIGIN
from shared.schemas.result import ServeResult
from shared.tasks.specs.serve import ServeSpecStrict, engine_env_vars
from shared.tasks.task_type import TaskType
from worker.config import WorkerConfig
from worker.hw import cuda_device_env
from worker.resident.local_engines import EngineSocketPathTooLong, LocalEngine

from ..utils.process import signal_process_group
from .base_executor import ExecutionError, Executor, ExecutorTask, RunSignals
from .utils.collective import loopback_collective_env
from .utils.serve_ttl import serve_deadline

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL_SEC = 2.0
# 600s default: cold-start includes model download, engine init, and CUDA graph capture
_DEFAULT_READINESS_TIMEOUT_SEC = 600.0
_POLL_INTERVAL_SEC = 5.0
_STOP_TIMEOUT_SEC = 15.0
_TAIL_MAX_LINES = 200
_TAIL_SNIPPET_BYTES = 4096


def _drain_to_log(
    proc: subprocess.Popen[str],
    tail: collections.deque[str],
    eof_event: threading.Event,
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        logger.info("[vllm] %s", line)
        tail.append(line)
    eof_event.set()


def _raise_with_tail(message: str, tail: collections.deque[str]) -> NoReturn:
    snippet = "\n".join(tail)
    raw = snippet.encode("utf-8", errors="replace")
    if len(raw) > _TAIL_SNIPPET_BYTES:
        snippet = "...\n" + raw[-_TAIL_SNIPPET_BYTES:].decode("utf-8", errors="replace")
    raise ExecutionError(
        message + (f"\n--- last vLLM output ---\n{snippet}" if snippet else "")
    )


class VLLMServeExecutor(Executor):
    name = "vllm_serve"
    supported_task_types = frozenset({TaskType.SERVE})
    runs_on_visible_gpus = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signals = RunSignals()
        self._proc: subprocess.Popen[str] | None = None
        self._devices: tuple[str, ...] | None = None

    @property
    def binds_devices(self) -> bool:
        return self.runs_on_visible_gpus

    def bind_devices(self, devices: tuple[str, ...] | None) -> None:
        # Each run launches its own engine, so the next launch takes the binding.
        self._devices = devices

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except Exception:
            return False

    def run(self, task: ExecutorTask, out_dir: Path) -> ServeResult:
        with self._signals.running(task.task_id):
            return self._serve(task, out_dir)

    def _serve(self, task: ExecutorTask, out_dir: Path) -> ServeResult:
        spec = self.require_spec(task, ServeSpecStrict)

        model_id = spec.model_name
        if model_id is None:
            raise ExecutionError("Serve spec is missing model.source.identifier")

        deadline = serve_deadline(
            spec.ttlSeconds,
            task.serve_elapsed_sec,
            self._config.serve_default_ttl_sec,
            self._config.serve_max_ttl_sec,
        )
        if deadline <= time.time():
            logger.info("Serve task %s TTL elapsed; not starting vLLM", task.task_id)
            return ServeResult(model=model_id)
        readiness_timeout = (
            spec.readinessTimeoutSeconds or _DEFAULT_READINESS_TIMEOUT_SEC
        )
        try:
            with self._local_engines().socket_path() as socket_path:
                return self._launch(
                    task,
                    spec,
                    model_id,
                    deadline,
                    readiness_timeout,
                    socket_path,
                    out_dir,
                )
        except EngineSocketPathTooLong as exc:
            raise ExecutionError(str(exc)) from exc

    def _launch(
        self,
        task: ExecutorTask,
        spec: ServeSpecStrict,
        model_id: str,
        deadline: float,
        readiness_timeout: float,
        socket_path: Path,
        out_dir: Path,
    ) -> ServeResult:
        """Run the engine on ``socket_path`` until its TTL elapses or it is stopped.

        The engine listens only on the worker-private socket, so only this worker's
        claim-gated sidecar reaches it, presenting the internally generated key it
        resolves inside the worker. External access is only through the gated task-ID
        serve route.
        """
        local_engines = self._local_engines()
        api_key = secrets.token_hex(32)

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        vllm_kwargs = dict(spec.model.vllm or {}) if spec.model is not None else {}
        try:
            env_vars = engine_env_vars(vllm_kwargs.pop("env_vars", None))
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc
        rendered_flags: set[str] = set()
        for k, v in vllm_kwargs.items():
            if v is None:
                continue
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
                    rendered_flags.add(flag)
            else:
                value = json.dumps(v) if isinstance(v, dict | list) else str(v)
                cmd.extend([flag, value])
                rendered_flags.add(flag)

        if spec.model_trust_remote_code and "--trust-remote-code" not in rendered_flags:
            cmd.append("--trust-remote-code")
        # The executor's own options come last: vLLM keeps the last value of a repeated
        # option.
        cmd.extend(["--model", model_id, "--uds", socket_path.as_posix()])
        if revision := spec.model_revision:
            cmd.extend(["--revision", revision])

        env = dict(os.environ)
        env.update(loopback_collective_env())
        # A spec setting its own CUDA variables runs unbound, so no binding replaces
        # them.
        env.update(env_vars)
        if self._devices is not None:
            env.update(cuda_device_env(self._devices))
        # The key rides the engine's environment: argv is readable by every local user.
        env["VLLM_API_KEY"] = api_key
        env.setdefault("VLLM_CONFIGURE_LOGGING", "0")
        env["PYTHONUNBUFFERED"] = "1"
        if "--enable-lora" in rendered_flags:
            # A LoRA-enabled serve accepts runtime adapter loads so a resident consumer
            # can load its adapter into a slot on demand.
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

        logger.info(
            "Starting vLLM server for model %s "
            "(task=%s ttl=%.0fs readiness_timeout=%.0fs)",
            model_id,
            task.task_id,
            deadline - time.time(),
            readiness_timeout,
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        if self._signals.raise_if_cancelled():
            logger.info("Serve task %s stopped before vLLM launch", task.task_id)
            return ServeResult(model=model_id)

        tail: collections.deque[str] = collections.deque(maxlen=_TAIL_MAX_LINES)
        try:
            proc = subprocess.Popen(  # nosec B603 - argv list, no shell=True, absolute path via sys.executable
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise ExecutionError(f"Failed to start vLLM server: {exc}") from exc

        eof_event = threading.Event()
        drain_thread = threading.Thread(
            target=_drain_to_log, args=(proc, tail, eof_event), daemon=True
        )
        drain_thread.start()

        self._proc = proc
        try:
            self._poll_health(
                proc, socket_path, task.task_id, readiness_timeout, tail, eof_event
            )
            if self._signals.raise_if_cancelled():
                logger.info("Serve task stop requested before vLLM became ready")
                return ServeResult(model=model_id)
            # Worker-private endpoint facts ("_"-prefixed so task metadata never
            # discloses them); the resident endpoint probe reads them to bind the
            # claim-gated sidecar in front of the engine.
            interface = (
                "embedding" if vllm_kwargs.get("runner") == "pooling" else "chat"
            )
            update_payload: dict[str, Any] = {
                "serve": {
                    "model": model_id,
                    "interface": interface,
                    "_socket": socket_path.as_posix(),
                }
            }
            local_engines.publish(
                task.task_id, LocalEngine(socket_path.as_posix(), api_key)
            )
            self.emit_update(task.task_id, update_payload)
            logger.info("vLLM server ready (task=%s)", task.task_id)
            self._wait_for_serve(proc, deadline)
        finally:
            local_engines.withdraw(task.task_id)
            self._proc = None
            self._terminate_process_group(proc)
            drain_thread.join(timeout=5.0)

        return ServeResult(model=model_id)

    def _poll_health(
        self,
        proc: subprocess.Popen[str],
        socket_path: Path,
        task_id: str,
        timeout_sec: float,
        tail: collections.deque[str],
        eof_event: threading.Event | None = None,
    ) -> None:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path.as_posix()), timeout=2.0
        ) as client:
            self._poll_health_over(client, proc, task_id, timeout_sec, tail, eof_event)

    def _poll_health_over(
        self,
        client: httpx.Client,
        proc: subprocess.Popen[str],
        task_id: str,
        timeout_sec: float,
        tail: collections.deque[str],
        eof_event: threading.Event | None,
    ) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._signals.raise_if_cancelled():
                return
            if proc.poll() is not None:
                # A stop or cancel terminates the process it waits on.
                if self._signals.raise_if_cancelled():
                    return
                _raise_with_tail(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})",
                    tail,
                )
            if eof_event is not None and eof_event.is_set():
                # Stdout pipe closed: the whole vLLM process tree exited.
                # proc.poll() may lag slightly behind pipe close; wait briefly.
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                if self._signals.raise_if_cancelled():
                    return
                _raise_with_tail(
                    f"vLLM server process exited (code={proc.returncode}) "
                    f"before becoming ready (task={task_id})",
                    tail,
                )
            try:
                if client.get(f"{LOCAL_ENGINE_ORIGIN}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL_SEC)
        _raise_with_tail(
            f"vLLM server did not become ready within {timeout_sec:.0f}s "
            f"(task={task_id})",
            tail,
        )

    def _wait_for_serve(self, proc: subprocess.Popen[str], deadline: float) -> None:
        while time.time() < deadline:
            if self._signals.raise_if_cancelled():
                logger.info("Serve task stop requested; terminating vLLM server")
                return
            if proc.poll() is not None:
                if self._signals.raise_if_cancelled():
                    return
                raise ExecutionError(
                    f"vLLM server process exited unexpectedly (code={proc.returncode})"
                )
            time.sleep(_POLL_INTERVAL_SEC)
        logger.info("Serve task TTL reached; terminating vLLM server")

    def _terminate_process_group(self, proc: subprocess.Popen[str]) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
        if pgid is not None:
            try:
                signal_process_group(pgid, signal.SIGTERM)
            except (ProcessLookupError, ChildProcessError, OSError):
                pass
            try:
                proc.wait(timeout=_STOP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                try:
                    signal_process_group(pgid, signal.SIGKILL)
                except (ProcessLookupError, ChildProcessError, OSError):
                    pass
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            pass

    def cancel(self, task_id: str) -> None:
        if not self._signals.cancel(task_id):
            return
        proc = self._proc
        if proc is not None:
            self._terminate_process_group(proc)

    def stop(self, task_id: str) -> None:
        if not self._signals.stop(task_id):
            return
        proc = self._proc
        if proc is not None:
            self._terminate_process_group(proc)
