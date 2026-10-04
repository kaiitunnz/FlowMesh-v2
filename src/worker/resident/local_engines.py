"""The serve engines this worker launched, reached only from inside the worker."""

import contextlib
import shutil
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_SOCKET_NAME = "engine.sock"
_ROOT_PREFIX = "flowmesh-engines-"


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

    Each engine listens on a Unix socket in its own ``0700`` directory under a root
    only this worker's user can open, so only the worker's replica sidecar reaches it.
    A serve executor publishes its engine once it is ready and withdraws it when it
    stops, and the sidecar resolves the engine by serve task id when it binds.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._engines: dict[str, LocalEngine] = {}
        self._lock = threading.Lock()

    def _ensure_root(self) -> Path:
        with self._lock:
            if self._root is None:
                self._root = Path(tempfile.mkdtemp(prefix=_ROOT_PREFIX))
            else:
                self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
                self._root.chmod(0o700)
            return self._root

    @contextlib.contextmanager
    def socket_path(self) -> Iterator[Path]:
        """Yield a fresh socket path for one engine, removed with its directory."""
        directory = Path(tempfile.mkdtemp(prefix="engine-", dir=self._ensure_root()))
        try:
            yield directory / _SOCKET_NAME
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
