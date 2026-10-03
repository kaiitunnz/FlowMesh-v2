import logging
import os
import queue
import threading
from collections.abc import Callable, Iterable
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from typing import Any

import grpc

from shared.grpc.supervisor.v1 import supervisor_pb2, supervisor_pb2_grpc
from shared.utils.time import now_iso


class PrivateRotatingFileHandler(RotatingFileHandler):
    """A ``RotatingFileHandler`` that keeps its file and backups at mode 0600,
    including ones left by an earlier run, since the worker log records every
    task's details."""

    def __init__(self, filename: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(filename, *args, **kwargs)
        for index in range(1, self.backupCount + 1):
            try:
                os.chmod(f"{self.baseFilename}.{index}", 0o600)
            except FileNotFoundError:
                continue

    def _open(self) -> TextIOWrapper:
        append = "a" in self.mode
        flags = os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC
        flags |= os.O_APPEND if append else os.O_TRUNC
        fd = os.open(self.baseFilename, flags, 0o600)
        os.fchmod(fd, 0o600)
        if append:
            return open(fd, "a", encoding=self.encoding, errors=self.errors)
        return open(fd, "w", encoding=self.encoding, errors=self.errors)


def get_logger(
    name: str = "flowmesh_worker",
    log_file: str = "worker.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    level: str = "INFO",
) -> logging.Logger:
    """Return a configured logger with a rotating file handler and console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # File handler (rotating)
    fh = PrivateRotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger


def configure_hf_library_logging() -> None:
    """Disable HF default handlers and enable propagation to root.

    Safe to call multiple times; no-ops if the libraries are not installed.
    """

    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.disable_default_handler()
        hf_logging.enable_propagation()
    except Exception:
        pass

    try:
        from diffusers.utils import logging as diff_logging  # type: ignore

        diff_logging.disable_default_handler()
        diff_logging.enable_propagation()
    except Exception:
        pass


class _GrpcLogStream:
    _SENTINEL = object()

    def __init__(
        self,
        stub: supervisor_pb2_grpc.SupervisorStub,
        metadata: tuple[tuple[str, str], ...],
        struct_from_payload: Callable[[dict[str, Any]], Any],
        logger: logging.Logger,
    ) -> None:
        self._stub = stub
        self._metadata = metadata
        self._struct_from_payload = struct_from_payload
        self._logger = logger

        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=10_000)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ServerLogStream",
            daemon=True,
        )
        self._thread.start()

    def send(self, payload: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            self._q.put(payload, timeout=0.1)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._q.put(self._SENTINEL)
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _messages(self) -> Iterable[supervisor_pb2.LogMessage]:
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            if isinstance(item, dict):
                yield supervisor_pb2.LogMessage(payload=self._struct_from_payload(item))

    def _run(self) -> None:
        try:
            self._stub.PushLogs(self._messages(), metadata=self._metadata)
        except grpc.RpcError as exc:
            self._logger.debug("Server log stream error: %s", exc)
        except Exception as exc:
            self._logger.debug("Server log stream crashed: %s", exc)


class TaskLogEmitter(logging.Handler):
    """Per-task log handler that emits Python logging records to the server."""

    _traceback_formatter = logging.Formatter()

    def __init__(
        self,
        stub: supervisor_pb2_grpc.SupervisorStub,
        metadata: tuple[tuple[str, str], ...],
        struct_from_payload: Callable[[dict[str, Any]], Any],
        logger: logging.Logger,
        task_id: str,
        workflow_id: str,
        owner_id: str,
        worker_id: str,
        task_refs: list[dict[str, str]] | None = None,
        scrub: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(level=logging.NOTSET)
        self._logger = logger
        self._scrub = scrub
        self._task_id = task_id
        self._workflow_id = workflow_id
        self._owner_id = owner_id
        self._worker_id = worker_id
        self._task_refs = (
            [{"task_id": task_id, "workflow_id": workflow_id}]
            if task_refs is None
            else task_refs
        )
        self._stream = _GrpcLogStream(
            stub=stub,
            metadata=metadata,
            struct_from_payload=struct_from_payload,
            logger=logger,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", ""))
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self._traceback_formatter.formatException(
                    record.exc_info
                )
            if exc_text := record.exc_text:
                message = f"{message}\n{exc_text}" if message else exc_text
        if not message:
            return
        if self._scrub is not None:
            message = self._scrub(message)

        stream = getattr(record, "flowmesh_stream", None)
        if stream not in ("stdout", "stderr", "system"):
            stream = "system"

        payload: dict[str, Any] = {
            "type": "TASK_LOG",
            "ts": now_iso(),
            "workflow_id": self._workflow_id,
            "task_id": self._task_id,
            "task_refs": self._task_refs,
            "owner_id": self._owner_id,
            "worker_id": self._worker_id,
            "level": record.levelname,
            "stream": stream,
            "logger": record.name,
            "message": message,
        }
        self._stream.send(payload)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            super().close()

    def emit_warning_only(self, message: str) -> None:
        """Send a single warning log line and do not attach the handler."""
        payload: dict[str, Any] = {
            "type": "TASK_LOG",
            "ts": now_iso(),
            "workflow_id": self._workflow_id,
            "task_id": self._task_id,
            "task_refs": self._task_refs,
            "owner_id": self._owner_id,
            "worker_id": self._worker_id,
            "level": "WARNING",
            "stream": "system",
            "logger": "task_log_emitter",
            "message": message,
        }
        try:
            self._stream.send(payload)
        finally:
            self._stream.close()
