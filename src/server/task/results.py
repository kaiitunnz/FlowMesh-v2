"""How the control plane reads a task's result back from the shared content store.

A task's result is the envelope its worker stored before reporting success, named by
the reference that success bound. Reading it hydrates that reference and verifies the
bytes against it, so a result that is missing, corrupt, or not the envelope it names is
refused rather than read as some other value. Nothing here is persisted: the store is
the only place a result lives, and this keeps just a bounded in-memory copy of what it
recently verified, which an immutable reference makes safe to reuse.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from shared.content import ContentReference, ContentStoreError, FabricObjectStore
from shared.schemas.result import BaseExecutorResult, ResultEnvelope

_CACHE_MAX_BYTES = 64 * 1024 * 1024


class ResultUnreadable(RuntimeError):
    """A bound result could not be read from the store or is not a result envelope."""


@dataclass(frozen=True)
class ResultBinding:
    """What a settled task's result resolves to: a stored envelope or a skip."""

    task_id: str
    reference: ContentReference | None = None
    skip: dict[str, Any] | None = None
    settled_at: str | None = None


def skip_envelope(binding: ResultBinding) -> ResultEnvelope:
    """The envelope a task that settled without running reads as."""
    envelope = ResultEnvelope(
        task_id=binding.task_id, result=BaseExecutorResult(), metadata=binding.skip
    )
    if binding.settled_at is not None:
        envelope.received_at = binding.settled_at
    return envelope


class ResultReader:
    """Reads stored result envelopes, verified against the reference naming them."""

    def __init__(
        self, store: FabricObjectStore, *, cache_max_bytes: int = _CACHE_MAX_BYTES
    ) -> None:
        self._store = store
        self._cache_max_bytes = cache_max_bytes
        self._cache: OrderedDict[ContentReference, bytes] = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.Lock()

    def read_bytes(self, binding: ResultBinding) -> bytes:
        """The bytes a binding reads as: the stored envelope, or a synthesized skip."""
        if binding.reference is None:
            if binding.skip is None:
                raise ResultUnreadable(f"task {binding.task_id} has no bound result")
            return skip_envelope(binding).model_dump_json(indent=2).encode("utf-8")
        return self._hydrate(binding.reference)

    def read(self, binding: ResultBinding) -> ResultEnvelope:
        """The envelope a binding reads as, validated as a result envelope."""
        if binding.reference is None and binding.skip is not None:
            return skip_envelope(binding)
        data = self.read_bytes(binding)
        try:
            return ResultEnvelope.model_validate_json(data)
        except ValidationError as exc:
            raise ResultUnreadable(
                f"the stored result of task {binding.task_id} is not a result "
                f"envelope: {exc}"
            ) from exc

    def _hydrate(self, reference: ContentReference) -> bytes:
        key = reference
        with self._lock:
            if (cached := self._cache.get(key)) is not None:
                self._cache.move_to_end(key)
                return cached
        try:
            data = self._store.hydrate(reference)
        except ContentStoreError as exc:
            raise ResultUnreadable(str(exc)) from exc
        self._remember(key, data)
        return data

    def _remember(self, key: ContentReference, data: bytes) -> None:
        if len(data) > self._cache_max_bytes:
            return
        with self._lock:
            if key in self._cache:
                return
            self._cache[key] = data
            self._cached_bytes += len(data)
            while self._cached_bytes > self._cache_max_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= len(evicted)


__all__ = [
    "ResultBinding",
    "ResultReader",
    "ResultUnreadable",
    "skip_envelope",
]
