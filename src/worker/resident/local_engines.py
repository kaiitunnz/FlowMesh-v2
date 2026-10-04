"""The serve engines this worker launched, reached only from inside the worker."""

import contextlib
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_SOCKET_NAME = "engine.sock"
_DIR_PREFIX = "flowmesh-engine-"
# ``sun_path`` holds 108 bytes, including the terminating NUL.
_MAX_SOCKET_PATH_BYTES = 107


class EngineSocketPathTooLong(ValueError):
    """An engine's socket path exceeds the Unix socket path limit."""


@dataclass(frozen=True)
class LocalEngine:
    """A serve engine this worker launched, listening on its worker-private socket.

    ``api_key`` is the key the engine accepts; a keyless stand-in has none, and presents
    the key its sidecar bind carries to its own upstream.
    """

    socket_path: str
    api_key: str | None = None


class LocalEngineRegistry:
    """The live engine of each serve task this worker runs, keyed by its task id.

    Each engine listens on a Unix socket in its own ``0700`` directory, created under
    ``parent`` (the temp directory by default), so only the worker's replica sidecar
    reaches it. A serve executor publishes its engine once it is ready and withdraws it
    when it stops, and the sidecar resolves the engine by serve task id when it binds.
    """

    def __init__(self, parent: Path | None = None) -> None:
        self._parent = parent
        self._engines: dict[str, LocalEngine] = {}
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def socket_path(self) -> Iterator[Path]:
        """Yield a fresh socket path for one engine, removed with its directory.

        Raises :class:`EngineSocketPathTooLong` when the path would not fit a Unix
        socket address.
        """
        directory = Path(tempfile.mkdtemp(prefix=_DIR_PREFIX, dir=self._parent))
        try:
            path = directory / _SOCKET_NAME
            if (size := len(os.fsencode(path))) > _MAX_SOCKET_PATH_BYTES:
                raise EngineSocketPathTooLong(
                    f"the engine socket path {path} is {size} bytes, past the "
                    f"{_MAX_SOCKET_PATH_BYTES}-byte Unix socket limit; point TMPDIR "
                    "at a shorter directory"
                )
            yield path
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def publish(self, serve_task_id: str, engine: LocalEngine) -> None:
        with self._lock:
            self._engines[serve_task_id] = engine

    def withdraw(self, serve_task_id: str) -> None:
        with self._lock:
            self._engines.pop(serve_task_id, None)

    def lookup(self, serve_task_id: str) -> LocalEngine | None:
        """Return the live engine this worker launched for a serve task, if any."""
        with self._lock:
            return self._engines.get(serve_task_id)
