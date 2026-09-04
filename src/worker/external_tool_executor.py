"""The worker-hosted external-tool executor: the enforced tool-fence egress boundary.

A fabric-served external-tool operation reaches the worker over its existing supervisor
attachment as an opaque frame; this executor validates the operation fence and one-use
delivery nonce, performs the provider egress, and returns one opaque outcome frame back
over the attachment. It runs in a normal worker process, distinct from the
network-fenced harness sandbox: the harness never egresses, and the provider credential
is read only from this process's local environment and never travels on the wire.

A failed fence returns a reject frame with no provider call and never demotes
reachability. A lost or crashed egress sends no reply, so the origin's carriage leaves
the durable boundary pending for a same-``idempotency_key`` re-drive; the executor never
manufactures a terminal outcome for a delivery whose provider reply it did not decode.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from shared.outcome import FabricContentStore, OutcomeManifest
from shared.tools.contract import (
    RemoteToolOperationEnvelope,
    ToolOperationEnvelope,
    ToolOutcome,
    ToolOutcomeStatus,
)
from shared.tools.search.egress import ExternalToolSidecar
from shared.tools.search.providers import LazySearchProvider
from shared.tools.search.schema import (
    SEARCH_INTERFACE,
    ToolRequest,
    tool_request_digest,
)
from shared.tools.wire import (
    FRAME_CANCEL,
    FRAME_OPERATION,
    FRAME_REAP,
    FRAME_REPLY,
    KIND_MANIFEST,
    KIND_OPERATION,
    KIND_REJECT,
    KIND_RESULT,
    decode_msg,
    encode_msg,
)

# A sink the executor calls to return one attachment frame to the supervisor: the
# session id, the ToolEgressFrame.kind (FRAME_REPLY / FRAME_REAP), and payload bytes.
EgressResultSink = Callable[[str, str, bytes], None]


@dataclass(frozen=True)
class _ProviderBinding:
    """The provider name and optional key the local worker environment provisions."""

    provider: str
    api_key: str | None


class WorkerExternalToolExecutor:
    """Serves fence-gated external-tool operations delivered over the attachment."""

    def __init__(
        self,
        *,
        worker_id: str,
        generation: int,
        provider: str,
        api_key: str | None,
        result_sink: EgressResultSink,
        content_store: FabricContentStore | None = None,
        policy_class: str = "default",
        interfaces: frozenset[str] = frozenset({SEARCH_INTERFACE}),
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._generation = generation
        self._provider = provider
        self._content_store = content_store
        self._policy_class = policy_class
        self._interfaces = interfaces
        self._sink = result_sink
        self._log = logger or logging.getLogger("worker-external-tool")
        self._sidecar = ExternalToolSidecar(
            LazySearchProvider(_ProviderBinding(provider, api_key)), self._log
        )
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tool-egress"
        )
        self._lock = threading.Lock()
        # Consumed one-use delivery nonces -> fence deadline. Process-scoped to this
        # worker incarnation: a restart drops the set, but a stale delivery is rejected
        # on the rotated generation before its nonce is checked.
        self._nonces: dict[str, float] = {}
        # In-flight egress futures and cancelled sessions, keyed by session id.
        self._inflight: dict[str, Future[None]] = {}
        self._cancelled: set[str] = set()

    def submit(self, session_id: str, kind: str, payload: bytes) -> None:
        """Route one attachment frame: an operation to egress, or a cancel to reap."""
        if kind == FRAME_OPERATION:
            with self._lock:
                if session_id in self._inflight:
                    return
                fut = self._pool.submit(self._run, session_id, payload)
                self._inflight[session_id] = fut
        elif kind == FRAME_CANCEL:
            self._cancel(session_id)

    def stop(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _cancel(self, session_id: str) -> None:
        with self._lock:
            fut = self._inflight.get(session_id)
            if fut is not None and fut.cancel():
                # Cancelled before it started: _run never fires, so drop its in-flight
                # entry now and record nothing to suppress.
                self._inflight.pop(session_id, None)
            elif fut is not None:
                # Already running: suppress its late reply when _run finishes; _run
                # discards the marker, so the cancelled set never accumulates.
                self._cancelled.add(session_id)
        # Ack the reap so the supervisor can retire the routing record; a late reply
        # from an already-running egress is fenced by the cancelled set in _run.
        self._sink(session_id, FRAME_REAP, b"")

    def _run(self, session_id: str, payload: bytes) -> None:
        try:
            reply = self._egress(payload)
        except Exception as exc:  # noqa: BLE001 - a crashed egress leaves it ambiguous
            # No reply frame: the origin's carriage times out into an AmbiguousDelivery
            # and re-drives under the same idempotency key with a fresh fence and nonce.
            self._log.warning("tool egress raised, leaving it ambiguous: %s", exc)
            return
        finally:
            with self._lock:
                self._inflight.pop(session_id, None)
                cancelled = session_id in self._cancelled
                self._cancelled.discard(session_id)
        if not cancelled:
            self._sink(session_id, FRAME_REPLY, reply)

    def _egress(self, payload: bytes) -> bytes:
        opening = decode_msg(payload)
        if opening["kind"] != KIND_OPERATION:
            raise ValueError("expected an operation frame")
        envelope = RemoteToolOperationEnvelope.model_validate(opening["envelope"])
        request = ToolRequest.model_validate(opening["request"])
        if (reason := self._fence_reject(envelope, request)) is not None:
            self._log.info("tool fence rejected op reason=%s", reason)
            return encode_msg(KIND_REJECT, reason=reason)
        if not self._consume_nonce(envelope.delivery_nonce, envelope.deadline_epoch):
            self._log.info("tool fence rejected op reason=replay")
            return encode_msg(KIND_REJECT, reason="replay")
        if (prior := self._prior_manifest(envelope)) is not None:
            # A prior materialization under this idempotency key stands: return it
            # without sampling the provider again.
            return encode_msg(KIND_MANIFEST, manifest=prior.model_dump(mode="json"))
        colocated = ToolOperationEnvelope(
            interface=envelope.interface,
            idempotency_key=envelope.idempotency_key,
            max_results=envelope.max_results,
            timeout_sec=envelope.timeout_sec,
            result_char_cap=envelope.result_char_cap,
        )
        self._log.info(
            "tool egress in worker=%s interface=%s provider=%s",
            self._worker_id,
            envelope.interface,
            self._provider,
        )
        try:
            outcome = self._sidecar.execute(colocated, request)
        except ValueError as exc:
            # A misprovisioned provider (unknown backend, or a keyed provider with no
            # key) is a deterministic fault: a same-idm re-drive to this worker repeats
            # it. Return a terminal typed control datum inline, not a no-reply ambiguous
            # loss that re-drives until the retry budget exhausts.
            self._log.warning("tool provider unavailable: %s", exc)
            return encode_msg(
                KIND_RESULT,
                outcome=ToolOutcome(
                    status=ToolOutcomeStatus.UNAVAILABLE,
                    value="the external-tool provider is unavailable",
                ).model_dump(mode="json"),
            )
        # Materialize a provider result by reference before it leaves the worker; a
        # store failure raises here, out of _run's guard, leaving no reply so the origin
        # holds the boundary pending and re-drives under the same idempotency key.
        return self._deliver_outcome(envelope, outcome)

    def _deliver_outcome(
        self, envelope: RemoteToolOperationEnvelope, outcome: ToolOutcome
    ) -> bytes:
        """A manifest frame for a materialized provider result, else an inline datum.

        A successful result materializes through the content store so no result body
        crosses the origin. A typed control status is a bounded inline datum. With no
        store or idempotency key the result inlines as the compatibility fallback.
        """
        if (
            self._content_store is None
            or outcome.status is not ToolOutcomeStatus.SUCCESS
            or envelope.idempotency_key is None
        ):
            return encode_msg(KIND_RESULT, outcome=outcome.model_dump(mode="json"))
        manifest = self._content_store.materialize(
            envelope.tenant,
            envelope.idempotency_key,
            outcome.model_dump_json().encode(),
            media_type="application/json",
            provenance=f"worker:{self._worker_id}:{envelope.interface}",
        )
        return encode_msg(KIND_MANIFEST, manifest=manifest.model_dump(mode="json"))

    def _prior_manifest(
        self, envelope: RemoteToolOperationEnvelope
    ) -> OutcomeManifest | None:
        """The manifest already materialized for this operation's idempotency key."""
        if self._content_store is None or envelope.idempotency_key is None:
            return None
        return self._content_store.find(envelope.tenant, envelope.idempotency_key)

    def _fence_reject(
        self, envelope: RemoteToolOperationEnvelope, request: ToolRequest
    ) -> str | None:
        if envelope.interface not in self._interfaces:
            return "interface"
        if envelope.provider != self._provider:
            return "provider"
        if envelope.target_id != self._worker_id:
            return "audience"
        if envelope.target_generation != self._generation:
            return "generation"
        if envelope.policy_class != self._policy_class:
            return "policy"
        if time.time() > envelope.deadline_epoch:
            return "expired"
        if request.interface != envelope.interface:
            return "interface_mismatch"
        if request.max_results > envelope.max_results:
            return "budget"
        if (
            tool_request_digest(request.interface, request.query, request.max_results)
            != envelope.request_digest
        ):
            return "digest"
        return None

    def _consume_nonce(self, nonce: str, deadline_epoch: float) -> bool:
        """Atomically consume a one-use delivery nonce; ``False`` on an exact replay."""
        now = time.time()
        with self._lock:
            for expired in [n for n, exp in self._nonces.items() if exp <= now]:
                self._nonces.pop(expired, None)
            if nonce in self._nonces:
                return False
            self._nonces[nonce] = deadline_epoch
            return True


__all__ = ["EgressResultSink", "WorkerExternalToolExecutor"]
