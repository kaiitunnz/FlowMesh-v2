import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

import grpc
import grpc.aio
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from shared.grpc.supervisor.v1 import (
    supervisor_pb2,
    supervisor_pb2_grpc,
)
from shared.network.relay_frame import RelayFrame
from shared.utils import new_worker_id
from shared.utils.ids import PREFIX_WORKER

from ... import env
from ...clients.redis import (
    WORKER_ID_SEQ_KEY,
    WORKERS_SET_KEY,
    SyncRedisClient,
    worker_key,
)
from ...network.worker_bridge import RelayWorkerBridge
from ..adapters.base import WorkerAdapter, WorkerTokenType
from ..manager import WorkerManager
from ..registry import WorkerRegistry
from ..schemas import WorkerStatus
from ..services.relay_service import RelayService
from ..services.task_listener import TaskListener

# Rewrite node_id for each worker key that still exists, atomically. KEYS are
# worker keys; ARGV[1] is the new node id. Returns the count actually rewritten.
# KEYS: [worker id counter, workers set]
# ARGV: [worker id prefix, worker id bound on this supervisor, ...]
# Returns the next counter value whose worker id neither a recorded worker nor a
# binding here holds, and records that id, so a counter the store lost never hands out
# an id still in use.
_ALLOCATE_WORKER_LUA = """
local bound = {}
for i = 2, #ARGV do bound[ARGV[i]] = true end
while true do
  local seq = redis.call('INCR', KEYS[1])
  local id = ARGV[1] .. seq
  if not bound[id] and redis.call('SADD', KEYS[2], id) == 1 then
    return seq
  end
end
"""

# Longer than a worker waits before reconnecting a closed event stream.
_REATTACH_GRACE_SEC = 10.0

_REHOME_LUA = """
local rehomed = 0
for _, key in ipairs(KEYS) do
  if redis.call('EXISTS', key) == 1 then
    redis.call('HSET', key, 'node_id', ARGV[1])
    rehomed = rehomed + 1
  end
end
return rehomed
"""


def _token_from_context(context: grpc.aio.ServicerContext) -> WorkerTokenType | None:
    if metadata := context.invocation_metadata():
        for key, value in metadata:
            key_l = key.lower()
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", errors="ignore")
            elif isinstance(value, memoryview):
                value = value.tobytes().decode("utf-8", errors="ignore")
            if key_l == "authorization" and value.startswith("Bearer "):
                return WorkerTokenType(value[7:])
            if key_l == "x-worker-token":
                return WorkerTokenType(value)
    return None


def _load_tls_credentials() -> grpc.ServerCredentials | None:
    cert_file = env.SERVER_GRPC_TLS_CERT_FILE
    key_file = env.SERVER_GRPC_TLS_KEY_FILE
    if not (cert_file or key_file):
        return None
    if not (cert_file and key_file):
        raise RuntimeError(
            "SERVER_GRPC_TLS_CERT_FILE and SERVER_GRPC_TLS_KEY_FILE are required"
        )
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    try:
        cert_bytes = cert_path.read_bytes()
        key_bytes = key_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Failed to read server TLS files: {exc}") from exc
    return grpc.ssl_server_credentials([(key_bytes, cert_bytes)])


def _struct_from_payload(payload: dict) -> Struct:
    struct = Struct()
    struct.update(payload)
    return struct


def _payload_from_struct(struct: Struct) -> dict:
    return MessageToDict(struct, preserving_proto_field_name=True)


class SupervisorServicer(supervisor_pb2_grpc.SupervisorServicer):
    def __init__(
        self,
        registry: WorkerRegistry,
        redis: SyncRedisClient,
        node_id: str,
        node_alias: str,
        task_listener: TaskListener,
        relay_service: RelayService,
        worker_manager: WorkerManager,
        logger: logging.Logger,
        relay_bridges: dict[str, RelayWorkerBridge] | None = None,
    ) -> None:
        self._registry = registry
        self._task_listener = task_listener
        self._relay_service = relay_service
        self._redis = redis
        self._node_id = node_id
        self._node_alias = node_alias
        self._worker_manager = worker_manager
        # Each relay namespace's bridge, by the event type its frames push up as.
        self._relay_bridges = relay_bridges or {}
        self._logger = logger
        # Guards _node_id and the registry-vs-rehome window against concurrent
        # RegisterWorker (grpc loop thread) and rebind_node (heartbeat thread).
        self._lock = Lock()
        # Worker ids whose unregister already reached the root, so their release sends
        # none.
        self._unregistered: set[str] = set()
        self._unregistered_lock = Lock()
        self._pending_unregisters: set[asyncio.Task[None]] = set()

    def reconcile_workers(self) -> None:
        """Release every binding whose worker the root no longer records, so the worker
        registers again rather than running unseen."""
        with self._lock:
            recorded = self._redis.set_members(WORKERS_SET_KEY)
            gone = [
                worker_id
                for worker_id in self._registry.bound_worker_ids()
                if worker_id not in recorded
            ]
            released = sum(self._registry.retire(worker_id) for worker_id in gone)
        if released:
            self._logger.warning(
                "Released %d worker(s) the root no longer records: %s",
                released,
                ", ".join(gone),
            )

    def worker_id_released(self, worker_id: str) -> None:
        """Tell the root a worker id ended, unless its own unregister already did."""
        with self._unregistered_lock:
            if worker_id in self._unregistered:
                self._unregistered.discard(worker_id)
                return
            self._unregistered.add(worker_id)
        self._relay_service.add_unregister(worker_id)

    def _note_unregistered(self, worker_id: str) -> bool:
        """Record that the root heard ``worker_id`` unregister; return whether it is the
        first to."""
        with self._unregistered_lock:
            if worker_id in self._unregistered:
                return False
            self._unregistered.add(worker_id)
            return True

    def rebind_node(self, node_id: str) -> None:
        """Re-home this node's workers under a new node id.

        Future registrations stamp the new id, and every already-registered
        worker's ``node_id`` field is rewritten in Redis so the dispatcher
        routes tasks to them on the node's new dispatch channel. A worker whose
        record no longer exists (e.g. Redis was wiped) is skipped rather than
        resurrected as a partial record.
        """
        with self._lock:
            if node_id == self._node_id:
                return
            old_node_id = self._node_id
            self._node_id = node_id
            worker_ids = [
                worker_id
                for worker in self._registry.all_workers()
                if (worker_id := self._registry.get_worker_id(worker.token)) is not None
            ]
            rehomed, skipped = self._rehome_workers(worker_ids, node_id)
            self._logger.info(
                "Re-homed %d worker(s) (%d skipped: no record) from node %s to %s",
                rehomed,
                skipped,
                old_node_id,
                node_id,
            )

    def _rehome_workers(self, worker_ids: list[str], node_id: str) -> tuple[int, int]:
        """Rewrite node_id only for workers whose record still exists, atomically
        so a worker deleted mid-rebind is skipped rather than resurrected as a
        partial record. Returns (rehomed, skipped)."""
        if not worker_ids:
            return 0, 0
        keys = [worker_key(worker_id) for worker_id in worker_ids]
        rehomed = int(self._redis.eval(_REHOME_LUA, len(keys), *keys, node_id))
        return rehomed, len(worker_ids) - rehomed

    async def RegisterWorker(
        self,
        request: supervisor_pb2.RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> supervisor_pb2.RegisterResponse:
        token = _token_from_context(context)
        if not token:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")
        worker = self._registry.try_get(token)
        if worker is None:
            # Unknown token: try admitting an external worker.
            await self._worker_manager.admit_worker(token)
            worker = self._registry.try_get(token)
        if worker is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")
        worker_meta = _payload_from_struct(request.meta)
        reported_alias = worker_meta.get("alias")
        if reported_alias != worker.alias:
            if reported_alias:
                self._logger.warning(
                    "Worker %s reported alias %r; check WORKER_ALIAS against the "
                    "alias the supervisor assigned or the worker's token",
                    worker.alias,
                    reported_alias,
                )
            worker_meta["alias"] = worker.alias
        # Stamp node_id, persist the record and set the worker id as one unit so a
        # concurrent rebind_node either sees this worker in its snapshot or stamps
        # it with the new id.
        with self._lock:
            incarnation = int(
                self._redis.eval(
                    _ALLOCATE_WORKER_LUA,
                    2,
                    WORKER_ID_SEQ_KEY,
                    WORKERS_SET_KEY,
                    f"{PREFIX_WORKER}-",
                    *self._registry.bound_worker_ids(),
                )
            )
            worker_id = new_worker_id(incarnation)
            worker_meta["id"] = worker_id
            worker_meta["incarnation"] = incarnation
            worker_meta["node_alias"] = self._node_alias
            worker_meta["node_id"] = self._node_id
            self._redis.hash_set(worker_key(worker_id), worker_meta)
            self._registry.set_worker_id(token, worker_id)
        self._task_listener.add_worker(worker_id)
        try:
            worker.set_worker_id(worker_id)
        except RuntimeError as exc:
            self._logger.warning(exc)
        self._logger.info("Registered worker %s", worker_id)
        return supervisor_pb2.RegisterResponse(
            worker_id=worker_id, incarnation=incarnation
        )

    async def StreamTasks(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[supervisor_pb2.DispatchMessage]:
        worker_id = self._get_worker_id_from_context(context)
        if worker_id is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        stream = self._task_listener.attach_stream(worker_id)
        if stream is None:
            self._logger.warning("No dispatch queue for worker %s", worker_id)
            return
        self._relay_service.add_attached(worker_id)
        try:
            while True:
                try:
                    event = await stream.next()
                except asyncio.CancelledError:
                    break
                if event is None:
                    break
                if event.get("kind") == "interrupt":
                    yield supervisor_pb2.DispatchMessage(
                        interrupt=supervisor_pb2.InterruptMessage(
                            task_id=str(event["task_id"]),
                            reason=str(event["reason"]),
                            dispatch_id=str(event.get("dispatch_id") or ""),
                        )
                    )
                elif event.get("kind") == "stop":
                    yield supervisor_pb2.DispatchMessage(
                        stop=supervisor_pb2.StopMessage(
                            task_id=str(event["task_id"]),
                            reason=str(event["reason"]),
                            dispatch_id=str(event.get("dispatch_id") or ""),
                        )
                    )
                elif event.get("kind") == "mediated_op":
                    yield supervisor_pb2.DispatchMessage(
                        mediated_op=supervisor_pb2.MediatedOperationFrame(
                            kind=str(event["frame_kind"]),
                            payload=_struct_from_payload(event["payload"]),
                        )
                    )
                else:
                    yield supervisor_pb2.DispatchMessage(
                        task=supervisor_pb2.TaskMessage(
                            payload=_struct_from_payload(event)
                        )
                    )
        finally:
            self._task_listener.detach_stream(stream)
        self._logger.info("Task stream closed for worker %s", worker_id)

    async def PushEvents(
        self,
        request_iterator: AsyncIterator[supervisor_pb2.EventMessage],
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        worker = self._get_worker_from_context(context)
        if worker is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        worker_id = self._registry.get_worker_id(worker.token)
        if worker_id is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Worker not registered"
            )

        worker.attach_event_stream()
        try:
            return await self._relay_events(
                worker, worker_id, request_iterator, context
            )
        finally:
            worker.detach_event_stream()

    async def _relay_events(
        self,
        worker: WorkerAdapter,
        worker_id: str,
        request_iterator: AsyncIterator[supervisor_pb2.EventMessage],
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        registered: bool = False
        unregistered: bool = False
        async for message in request_iterator:
            if self._registry.get_worker_id(worker.token) != worker_id:
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED, "Worker registration ended"
                )
            payload = _payload_from_struct(message.payload)
            event_type = payload.get("type")
            if (bridge := self._relay_bridges.get(str(event_type))) is not None:
                # A relay frame publishes up to the root bridge opaquely, which
                # carries it to the session's other end.
                await bridge.publish_up(
                    RelayFrame.from_wire(payload["payload"]["frame"])
                )
                continue
            # Control attributes an event to the worker this stream authenticated.
            payload["worker_id"] = worker_id
            # Trap register/unregister events
            match event_type:
                case "REGISTER":
                    registered = True
                    worker.set_status(WorkerStatus.RUNNING)
                case "UNREGISTER":
                    unregistered = True
                    if not self._note_unregistered(worker_id):
                        continue
            self._relay_service.add_event(payload)
        self._logger.info("Event stream closed for worker %s", worker_id)
        if registered and not unregistered:
            task = asyncio.ensure_future(
                self._unregister_unless_reattached(worker, worker_id)
            )
            self._pending_unregisters.add(task)
            task.add_done_callback(self._pending_unregisters.discard)
        try:
            worker.clear_worker_id()
        except RuntimeError as exc:
            self._logger.warning(exc)
        worker.set_status(WorkerStatus.STOPPED)
        return Empty()

    async def _unregister_unless_reattached(
        self, worker: WorkerAdapter, worker_id: str
    ) -> None:
        """Unregister a worker whose event stream closed without it unregistering,
        unless it re-attaches first, as a worker reconnecting after a blip does."""
        await asyncio.sleep(_REATTACH_GRACE_SEC)
        if worker.has_event_stream:
            return
        if self._registry.get_worker_id(worker.token) != worker_id:
            return
        if self._note_unregistered(worker_id):
            self._relay_service.add_unregister(worker_id)

    async def PushLogs(
        self,
        request_iterator: AsyncIterator[supervisor_pb2.LogMessage],
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        worker_id = self._get_worker_id_from_context(context)
        if worker_id is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid worker token")

        async for msg in request_iterator:
            payload = _payload_from_struct(msg.payload)
            if isinstance(payload, dict):
                payload.setdefault("worker_id", worker_id)
            self._relay_service.add_log(payload)
        self._logger.debug("Log stream closed for worker %s", worker_id)
        return Empty()

    def _get_worker_from_context(
        self, context: grpc.aio.ServicerContext
    ) -> WorkerAdapter | None:
        if token := _token_from_context(context):
            return self._registry.try_get(token)
        return None

    def _get_worker_id_from_context(
        self, context: grpc.aio.ServicerContext
    ) -> str | None:
        if token := _token_from_context(context):
            return self._registry.get_worker_id(token)
        return None


_GRPC_MAX_MSG_BYTES = 1024 * 1024 * 1024  # 1 GB
# The server pings each worker connection, so a half-open stream ends and its worker
# reattaches within about the sum of these, rather than swallowing frames sent to it.
_GRPC_KEEPALIVE_TIME_MS = 20_000
_GRPC_KEEPALIVE_TIMEOUT_MS = 10_000


class GrpcServer:
    def __init__(
        self,
        host: str,
        port: int,
        registry: WorkerRegistry,
        redis: SyncRedisClient,
        node_id: str,
        node_alias: str,
        task_listener: TaskListener,
        relay_service: RelayService,
        worker_manager: WorkerManager,
        logger: logging.Logger,
        relay_bridges: dict[str, RelayWorkerBridge] | None = None,
    ) -> None:
        self._logger = logger
        self._server: grpc.aio.Server | None = None
        self._servicer = SupervisorServicer(
            registry,
            redis,
            node_id,
            node_alias,
            task_listener,
            relay_service,
            worker_manager,
            logger,
            relay_bridges=relay_bridges,
        )
        self._listen_addr = f"{host}:{port}"

    async def start(self) -> None:
        if self._server is not None:
            self._logger.warning("Server gRPC server already started")
            return
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", _GRPC_MAX_MSG_BYTES),
                ("grpc.max_send_message_length", _GRPC_MAX_MSG_BYTES),
                ("grpc.keepalive_time_ms", _GRPC_KEEPALIVE_TIME_MS),
                ("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS),
                (
                    "grpc.keepalive_permit_without_calls",
                    int(env.SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS),
                ),
                (
                    "grpc.http2.min_recv_ping_interval_without_data_ms",
                    env.SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS,
                ),
            ]
        )
        supervisor_pb2_grpc.add_SupervisorServicer_to_server(
            self._servicer, self._server
        )
        creds = (
            None if env.SUPERVISOR_GRPC_DISABLE_SERVER_TLS else _load_tls_credentials()
        )
        if creds is None:
            bound_port = self._server.add_insecure_port(self._listen_addr)
            self._logger.warning(
                "Server gRPC TLS disabled; running insecure on %s",
                self._listen_addr,
            )
        else:
            bound_port = self._server.add_secure_port(self._listen_addr, creds)
            self._logger.info(
                "Server gRPC TLS enabled; running secure on %s", self._listen_addr
            )
        if not bound_port:
            raise RuntimeError(f"Failed to bind gRPC server to {self._listen_addr}")
        await self._server.start()
        self._logger.info("Server gRPC server started on %s", self._listen_addr)

    async def stop(self, grace: float = 5.0) -> None:
        if self._server is None:
            return
        await self._server.stop(grace)
        self._server = None
        self._logger.info("Server gRPC server stopped")

    def rebind_node(self, node_id: str) -> None:
        """Re-home registered workers under a new node id."""
        self._servicer.rebind_node(node_id)

    def reconcile_workers(self) -> None:
        """Release the workers the root no longer records."""
        self._servicer.reconcile_workers()

    def worker_id_released(self, worker_id: str) -> None:
        """Tell the root a worker id's binding ended."""
        self._servicer.worker_id_released(worker_id)
