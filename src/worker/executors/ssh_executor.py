"""SSH session executor.

Supports two modes:

**Interactive** (default): Creates an ephemeral session running sshd, emits a
TASK_UPDATE event with connection info, and blocks until the session ends (TTL,
idle timeout, or worker shutdown).

**Non-interactive** (``interactive=false``): Runs a user-provided container
image with an optional custom entrypoint/command.

A session backend (``worker.executors.ssh_session``) supplies the sandbox the
session runs in.
"""

import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared.schemas.result import SSHResult
from shared.tasks.specs.ssh import RELAYED_SSH_ACCESS_MODES, SSHSpecStrict
from shared.tasks.task_type import TaskType
from shared.utils import new_ssh_session_id
from shared.utils.manifest import ARTIFACTS_DIR, prepare_output_dir
from worker.config import WorkerConfig
from worker.executors.ssh_session import (
    SessionGone,
    SessionInterrupted,
    SessionRequest,
    SSHConfig,
    SSHSession,
    SSHSessionBackend,
    select_backend_cls,
)
from worker.executors.ssh_session.config import output_limit, raise_if_exceeded
from worker.executors.ssh_session.inputs import resolve_inputs
from worker.executors.utils.checkpoints import maybe_upload_artifacts
from worker.gpu_binding import FreeGpus

from .base_executor import ExecutionError, Executor, ExecutorTask, RunSignals

logger = logging.getLogger(__name__)

_SESSION_READY_TIMEOUT_SEC = 30.0


class SSHExecutor(Executor):
    """Executor for SSH tasks (interactive sessions and non-interactive jobs)."""

    name = "ssh"
    supported_task_types = frozenset({TaskType.SSH})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self._config
        self._owner = config.container_name or config.alias
        self._signals = RunSignals()
        self._current_session: SSHSession | None = None
        self._backend = self._make_backend(config)
        # A worker that died with a session up left it behind.
        self._backend.reap_stale()

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        return select_backend_cls(config) is not None

    @property
    def backend(self) -> SSHSessionBackend:
        return self._backend

    def _make_backend(self, config: WorkerConfig) -> SSHSessionBackend:
        backend_cls = select_backend_cls(config)
        if backend_cls is None:
            raise ExecutionError(
                "No SSH session backend is available on this worker "
                f"(SSH_SESSION_BACKEND={config.ssh_session_backend})"
            )
        logger.info("SSH sessions use the %s backend", backend_cls.name)
        return backend_cls(config, self._hardware)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        self._backend.prepare()

    def teardown(self) -> None:
        """Stop all sessions owned by this worker."""
        self._backend.teardown(self._owner)

    # ------------------------------------------------------------------ #
    # Main execution
    # ------------------------------------------------------------------ #

    def _free_gpus(self) -> FreeGpus | None:
        if self._lifecycle is None or self._hardware is None:
            return None
        return FreeGpus.read(self._lifecycle, self._hardware.gpu.devices)

    def run(self, task: ExecutorTask, out_dir: Path) -> SSHResult:
        with self._signals.running(task.task_id):
            return self._run_session(task, out_dir)

    def _run_session(self, task: ExecutorTask, out_dir: Path) -> SSHResult:
        spec = self.require_spec(task, SSHSpecStrict)
        # A session holds its devices for as long as it lives, so it is never handed
        # one another tenant is on.
        cfg = SSHConfig.from_spec(spec, self._config, self._hardware, self._free_gpus())
        access_mode = cfg.access_mode
        interactive = cfg.interactive

        if interactive and access_mode not in ("direct", "proxy", "forward"):
            raise ExecutionError(f"accessMode '{access_mode}' is not supported")

        self.prepare()
        session_id = new_ssh_session_id()

        prepare_output_dir(out_dir)  # Ensure output dir exists before mounting
        if self._signals.interrupted:
            return self._interrupted_before_start(task, session_id)
        request = SessionRequest(
            task_id=task.task_id,
            session_id=session_id,
            owner=self._owner,
            cfg=cfg,
            out_dir=out_dir,
            resolved_inputs=resolve_inputs(task, cfg, self._config.results_dir),
            signals=self._signals,
        )

        session_kind = "SSH session" if interactive else "non-interactive"
        if interactive:
            logger.info(
                "Starting %s (task=%s session=%s mode=%s ttl=%ds idle=%ds)",
                session_kind,
                task.task_id,
                session_id,
                access_mode,
                cfg.ttl_sec,
                cfg.idle_sec,
            )
        else:
            logger.info(
                "Starting %s session (task=%s session=%s ttl=%ds cmd=%s)",
                session_kind,
                task.task_id,
                session_id,
                cfg.ttl_sec,
                cfg.command,
            )

        # Other start failures propagate; staging and a daemon call may succeed on a
        # retry.
        try:
            session = self._backend.start_session(request)
        except SessionInterrupted:
            return self._interrupted_before_start(task, session_id)

        self._current_session = session
        if self._signals.interrupted:
            # The request landed before the session existed, so it stopped nothing.
            self._interrupt_session(session)
        log_thread: threading.Thread | None = None
        if not interactive:
            log_thread = threading.Thread(
                target=session.drain_logs,
                daemon=True,
                name=f"flowmesh-session-logs-{task.task_id[:8]}",
            )
            log_thread.start()
        exit_code = 0
        try:
            session_info = (
                self._wait_session_ready(session, session_id, task, cfg)
                if interactive
                else {}
            )
            exit_code = self._wait_for_session(session, cfg)
            self._signals.raise_if_cancelled()
            # Output written after the last poll is checked once the session ends,
            # before the session's logs are saved under it.
            max_bytes = output_limit(cfg.output)
            if max_bytes is not None and (size := session.output_size_bytes()):
                raise_if_exceeded(size, max_bytes)
            result = SSHResult(session_id=session_id, exit_code=exit_code)
            if interactive:
                for key, value in session_info.items():
                    setattr(result, key, value)
            else:
                # Keep as fallback — captures any output the streaming thread missed.
                session.save_logs(out_dir)
                if cfg.command is not None:
                    result.command = cfg.command
                if cfg.entrypoint is not None:
                    result.entrypoint = cfg.entrypoint
            session.collect_output(out_dir / ARTIFACTS_DIR, max_bytes)
            maybe_upload_artifacts(task, out_dir, logger=logger, skip_errors=True)
        finally:
            # Its relayed connections end first, so none outlives the session's sshd.
            self.withdraw_ssh_endpoint(session_id)
            # A log stream ends only once its session stops. A cancel or stop stops
            # it at once, inside the worker's own stop timeout.
            session.stop(1 if self._signals.interrupted else cfg.stop_timeout_sec)
            if log_thread is not None:
                # Wait for the thread to drain remaining output before tearing down
                # the session.
                log_thread.join(timeout=30.0)
            self._current_session = None
            session.cleanup()

        if not (interactive or exit_code == 0):
            raise ExecutionError(
                f"{session_kind.capitalize()} session exited with code {exit_code}"
            )

        return result

    def _interrupted_before_start(
        self, task: ExecutorTask, session_id: str
    ) -> SSHResult:
        """End a task a cancel or stop reached before its session started: a cancel
        raises, and a stop succeeds."""
        self._signals.raise_if_cancelled()
        logger.info("SSH task %s stopped before its session started", task.task_id)
        return SSHResult(session_id=session_id, exit_code=0)

    def cancel(self, task_id: str) -> None:
        if self._signals.cancel(task_id):
            self._interrupt_session(self._current_session)

    def stop(self, task_id: str) -> None:
        if self._signals.stop(task_id):
            self._interrupt_session(self._current_session)

    @staticmethod
    def _interrupt_session(session: SSHSession | None) -> None:
        if session is None:
            return
        try:
            session.stop(1)
        except Exception:
            logger.debug("Failed to stop SSH session", exc_info=True)

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def _wait_session_ready(
        self, session: SSHSession, session_id: str, task: ExecutorTask, cfg: SSHConfig
    ) -> dict[str, Any]:
        access_mode = cfg.access_mode
        expires_at = self._iso_offset(cfg.ttl_sec)
        host_port = session.wait_ready(_SESSION_READY_TIMEOUT_SEC)
        if host_port is None:
            return {}
        host_name = self._backend.session_address(access_mode)
        ssh_info: dict[str, Any] = {
            "session_id": session_id,
            "mode": access_mode,
            "username": session.login_user(),
            "expires_at": expires_at,
            "host": host_name,
            "port": host_port,
        }
        if access_mode in RELAYED_SSH_ACCESS_MODES:
            # The root relays to the session by this id; the lane resolves its port.
            self.publish_ssh_endpoint(session_id, host_port)
            # The session's own address, for a client on a host that can reach it;
            # the server may rewrite `host` and `port` to its route, never these.
            ssh_info["directHost"] = host_name
            ssh_info["directPort"] = host_port
            ssh_info["directScope"] = self._backend.session_scope(access_mode)
            ssh_info["workerId"] = task.assigned_worker
            logger.info(
                "SSH %s session ready: host=%s port=%s (task=%s)",
                access_mode,
                host_name,
                host_port,
                task.task_id,
            )
        else:
            logger.info(
                "SSH session ready: host=%s port=%s (task=%s)",
                host_name,
                host_port,
                task.task_id,
            )
        self.emit_update(task.task_id, {"ssh": ssh_info})
        return {"expires_at": expires_at, "host": host_name, "port": host_port}

    def _wait_for_session(self, session: SSHSession, cfg: SSHConfig) -> int:
        """Block until the session exits or its TTL or idle timeout fires, stopping it
        at its TTL so nothing it writes lands after its output is collected; return
        the session's exit code."""
        deadline = time.time() + cfg.ttl_sec
        idle_enabled = cfg.interactive and cfg.idle_sec > 0
        last_active = time.time()
        idle_unobservable_logged = False
        while time.time() < deadline:
            if self._signals.raise_if_cancelled() or session.finish_requested():
                logger.info("SSH session finish requested; stopping session")
                session.stop(1)
                return 0
            try:
                exit_code = session.poll()
            except SessionGone as exc:
                # A stop or cancel stops the session it waits on; nothing else
                # removes it while it runs.
                if self._signals.raise_if_cancelled():
                    return 0
                raise ExecutionError(str(exc)) from exc
            if exit_code is not None:
                if self._signals.raise_if_cancelled():
                    return 0
                return exit_code
            self._enforce_output_limit(session, cfg)
            if idle_enabled:
                connections = session.established_connections()
                if connections is None:
                    if not idle_unobservable_logged:
                        logger.warning(
                            "SSH idle timeout cannot be enforced: this session's "
                            "connection state is not observable"
                        )
                        idle_unobservable_logged = True
                    last_active = time.time()
                elif connections > 0:
                    last_active = time.time()
                elif time.time() - last_active >= cfg.idle_sec:
                    logger.info(
                        "SSH session idle for %ds; stopping session", cfg.idle_sec
                    )
                    session.stop(1)
                    return 0
            time.sleep(cfg.poll_interval_sec)

        logger.info("SSH session TTL reached; stopping session")
        session.stop(cfg.stop_timeout_sec)
        return 0

    @staticmethod
    def _enforce_output_limit(session: SSHSession, cfg: SSHConfig) -> None:
        if (max_bytes := output_limit(cfg.output)) is None:
            return
        size = session.output_size_bytes()
        if size is None or size <= max_bytes:
            return
        session.stop(1)
        raise_if_exceeded(size, max_bytes)

    @staticmethod
    def _iso_offset(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()
