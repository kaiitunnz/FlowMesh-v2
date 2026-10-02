import json
import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from shared.schemas.command import (
    InterruptMessage,
    MediatedOpMessage,
    RevokeMessage,
    StopMessage,
    TaskMessage,
)
from shared.schemas.worker import SSHLimits, WorkerCapabilities
from shared.tasks import TaskEnvelope
from shared.tasks.specs import SSHSpecStrict, SSHSpecTemplate
from shared.tasks.worker_message import (
    WorkerHardware,
    WorkerStatus,
    WorkerTaskMessage,
)
from shared.utils import new_worker_id, now_iso, parse_mem_to_bytes
from shared.utils.hardware import gpu_meets_requirements, gpus_fit_dispatch
from shared.utils.ids import PREFIX_WORKER

from ..clients.redis import (
    WORKER_EVENT_CHANNEL,
    WORKER_ID_SEQ_KEY,
    WORKERS_SET_KEY,
    RedisClient,
    node_dispatch_channel,
    worker_hb_key,
    worker_key,
)

logger = logging.getLogger(__name__)

# KEYS: [worker id counter, workers set]
# ARGV: [worker id prefix, worker id to skip, ...]
# Returns the next counter value whose worker id neither a recorded worker holds nor
# ARGV skips, and records that id, so a counter the store lost never hands out an id
# still in use.
_ALLOCATE_WORKER_LUA = """
local skip = {}
for i = 2, #ARGV do skip[ARGV[i]] = true end
while true do
  local seq = redis.call('INCR', KEYS[1])
  local id = ARGV[1] .. seq
  if not skip[id] and redis.call('SADD', KEYS[2], id) == 1 then
    return seq
  end
end
"""


# A write for a worker that is no longer a set member must not recreate a partial
# record. A read-then-write cannot promise that, since the watchdog can reap
# between the two calls; these run the membership test and the write as one
# atomic Redis call and report whether the write landed.
#
# The worker is the authority on its own status, and the dispatcher's reservation
# fences it: reserving a worker for a dispatch records that dispatch, and an IDLE the
# worker reports applies only when it names the reserved dispatch, which it clears.
# An IDLE for an earlier dispatch is fenced and returns the reservation, so it can
# never free a worker a dispatch is still on its way to. A BUSY always applies and
# never moves the reservation. The status last reported is kept, fenced or not, so a
# release returns the worker to it.
_REPORT_STATUS_IF_REGISTERED = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
    return {0}
end
if ARGV[3] ~= '' then
    redis.call('SETEX', KEYS[3], ARGV[3], ARGV[2])
end
redis.call('HSET', KEYS[2], 'last_seen', ARGV[2])
if ARGV[4] == '' then
    return {1}
end
redis.call('HSET', KEYS[2], 'reported_status', ARGV[4])
if ARGV[4] == 'IDLE' then
    local reserved = redis.call('HGET', KEYS[2], 'reserved_dispatch')
    if reserved and reserved ~= ARGV[5] then
        return {2, reserved, redis.call('HGET', KEYS[2], 'reserved_task') or ''}
    end
    redis.call('HDEL', KEYS[2], 'reserved_dispatch', 'reserved_task')
end
redis.call('HSET', KEYS[2], 'status', ARGV[4])
if #ARGV > 5 then
    redis.call('HSET', KEYS[2], unpack(ARGV, 6))
end
return {1}
"""

_RESERVE_IF_REGISTERED = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
    return 0
end
redis.call('HSET', KEYS[2], 'status', 'BUSY', 'last_seen', ARGV[2],
    'reserved_dispatch', ARGV[3], 'reserved_task', ARGV[4])
return 1
"""

_RELEASE_IF_RESERVED = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
    return ''
end
if redis.call('HGET', KEYS[2], 'reserved_dispatch') ~= ARGV[3] then
    return ''
end
local status = redis.call('HGET', KEYS[2], 'reported_status') or 'IDLE'
redis.call('HSET', KEYS[2], 'status', status, 'last_seen', ARGV[2])
redis.call('HDEL', KEYS[2], 'reserved_dispatch', 'reserved_task')
return status
"""

_SET_FIELDS_IF_REGISTERED = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 0 then
    return 0
end
redis.call('HSET', KEYS[2], unpack(ARGV, 2))
return 1
"""

# A heartbeat key with no TTL left (missing, or never given one) is a stale worker.
_REAP_IF_STALE = """
if redis.call('TTL', KEYS[3]) >= 0 then
    return 0
end
redis.call('SREM', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[2], KEYS[3])
return 1
"""

# A record another node wrote under the same id is that node's worker; only that node's
# unregister deletes it.
_UNREGISTER_IF_NODE = """
local alias = redis.call('HGET', KEYS[2], 'node_alias')
if alias and alias ~= ARGV[2] then
    return 0
end
redis.call('SREM', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[2], KEYS[3])
return 1
"""


class ReportOutcome(StrEnum):
    """How the registry took a worker's report of its status."""

    APPLIED = "applied"
    UNKNOWN = "unknown"
    FENCED = "fenced"


class StatusReport(NamedTuple):
    """A worker's status report as the registry took it; a fenced IDLE names the
    dispatch the worker is reserved for."""

    outcome: ReportOutcome
    reserved_task: str | None = None
    reserved_dispatch: str | None = None


class Reservation(NamedTuple):
    """A worker reserved for a dispatch of a task."""

    worker_id: str
    task_id: str
    dispatch_id: str


def _merge_gpu_availability(
    hardware: WorkerHardware | None, availability: dict[str, Any]
) -> None:
    """Apply a worker's reported per-device availability onto its device list.

    ``hardware_json`` is the worker's registration-time description of itself and is
    never rewritten, while availability changes every heartbeat, so the two live in
    separate fields and are joined by UUID here. A device the worker said nothing
    about keeps ``None`` and schedules as it would unread.
    """
    if hardware is None or not availability:
        return
    for device in hardware.gpu.devices:
        reported = availability.get(device.uuid)
        if not isinstance(reported, dict):
            continue
        available = reported.get("available")
        if isinstance(available, bool):
            device.gpu_available = available
        free_bytes = reported.get("free_bytes")
        if isinstance(free_bytes, int):
            device.memory_free_bytes = free_bytes


def _flatten_fields(mapping: dict[str, str]) -> list[str]:
    """Flatten a hash mapping into the field/value ARGV tail HSET expects."""
    flat: list[str] = []
    for field, value in mapping.items():
        flat.append(field)
        flat.append(value)
    return flat


class Worker(BaseModel):
    id: str = Field(description="Worker identifier.")
    alias: str | None = Field(
        default=None, description="Optional human-readable alias."
    )
    namespace: str = Field(description="Worker namespace.")
    cluster: str = Field(description="Worker cluster.")
    node_id: str = Field(description="Owning node identifier.")
    node_alias: str = Field(description="Owning node alias.")
    version: str | None = Field(default=None, description="Worker version.")
    status: WorkerStatus = Field(
        default=WorkerStatus.UNKNOWN, description="Worker status."
    )
    started_at: str | None = Field(default=None, description="Start timestamp.")
    pid: int | None = Field(default=None, description="Worker process ID.")
    incarnation: int = Field(
        default=0, description="Monotonic registration incarnation."
    )
    env: dict[str, Any] = Field(default_factory=dict, description="Runtime metadata.")
    hardware: WorkerHardware | None = Field(
        default=None, description="Hardware metadata."
    )
    capabilities: WorkerCapabilities = Field(
        default_factory=WorkerCapabilities,
        description="Task capabilities the worker advertises.",
    )
    ssh_limits: SSHLimits | None = Field(
        default=None, description="Configured ceiling on SSH session resources."
    )
    tags: list[str] = Field(default_factory=list, description="Worker tags.")
    last_seen: str | None = Field(default=None, description="Last heartbeat timestamp.")
    cached_models: list[str] = Field(
        default_factory=list, description="Cached model identifiers."
    )
    cached_datasets: list[str] = Field(
        default_factory=list, description="Cached dataset identifiers."
    )
    cache_updated_ts: str | None = Field(
        default=None, description="Cache metadata update timestamp."
    )
    cost_per_hour: float | None = Field(
        default=None, description="Estimated hourly cost."
    )


class WorkerInfo(Worker):
    stale: bool = Field(description="Whether the worker heartbeat is stale.")


class WorkerRegistry:
    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    # ------------------------------------------------------------------ #
    # Worker lifecycle helpers
    # ------------------------------------------------------------------ #

    def allocate_worker_seq(self, skip: Iterable[str] = ()) -> int:
        """Record a fresh worker id and return its sequence number."""
        return int(
            self._rds.sync.eval(
                _ALLOCATE_WORKER_LUA,
                2,
                WORKER_ID_SEQ_KEY,
                WORKERS_SET_KEY,
                f"{PREFIX_WORKER}-",
                *skip,
            )
        )

    async def allocate_worker_seq_async(self, skip: Iterable[str] = ()) -> int:
        """Record a fresh worker id and return its sequence number."""
        return int(
            await self._rds.asyncio.eval(
                _ALLOCATE_WORKER_LUA,
                2,
                WORKER_ID_SEQ_KEY,
                WORKERS_SET_KEY,
                f"{PREFIX_WORKER}-",
                *skip,
            )
        )

    def register_worker(
        self,
        node_id: str,
        node_alias: str,
        worker_meta: dict[str, Any],
        skip: Iterable[str] = (),
    ) -> str:
        """Record a worker under a fresh id that no recorded worker holds and ``skip``
        leaves out, stamping its id, incarnation and node into ``worker_meta``."""
        seq = self.allocate_worker_seq(skip)
        worker_id = new_worker_id(seq)
        worker_meta["id"] = worker_id
        worker_meta["incarnation"] = seq
        worker_meta["node_id"] = node_id
        worker_meta["node_alias"] = node_alias
        self._rds.sync.hash_set(worker_key(worker_id), worker_meta)
        return worker_id

    async def register_worker_async(
        self,
        node_id: str,
        node_alias: str,
        worker_meta: dict[str, Any],
    ) -> str:
        seq = await self.allocate_worker_seq_async()
        worker_id = new_worker_id(seq)
        worker_meta["id"] = worker_id
        worker_meta["incarnation"] = seq
        worker_meta["node_id"] = node_id
        worker_meta["node_alias"] = node_alias
        await self._rds.asyncio.hash_set(worker_key(worker_id), worker_meta)
        return worker_id

    def update_worker_hb(
        self,
        worker_id: str,
        ts: str,
        ttl_sec: int,
        status: WorkerStatus | None = None,
        dispatch_id: str | None = None,
    ) -> StatusReport:
        """Record a worker's heartbeat, and the status it carries when it has one."""
        return self._report_status(worker_id, ts, str(ttl_sec), status, dispatch_id, {})

    def set_worker_status(
        self,
        worker_id: str,
        status: WorkerStatus,
        ts: str,
        extra: dict[str, Any] | None,
        dispatch_id: str | None = None,
    ) -> StatusReport:
        """Record the status a worker reports, with the dispatch it concerns."""
        fields = {f"extra_{k}": str(v) for k, v in (extra or {}).items()}
        return self._report_status(worker_id, ts, "", status, dispatch_id, fields)

    def _report_status(
        self,
        worker_id: str,
        ts: str,
        ttl: str,
        status: WorkerStatus | None,
        dispatch_id: str | None,
        fields: dict[str, str],
    ) -> StatusReport:
        reply = self._rds.sync.eval(
            _REPORT_STATUS_IF_REGISTERED,
            3,
            WORKERS_SET_KEY,
            worker_key(worker_id),
            worker_hb_key(worker_id),
            worker_id,
            ts,
            ttl,
            status.value if status is not None else "",
            dispatch_id or "",
            *_flatten_fields(fields),
        )
        match int(reply[0]):
            case 0:
                return StatusReport(ReportOutcome.UNKNOWN)
            case 2:
                return StatusReport(
                    ReportOutcome.FENCED, _text(reply[2]) or None, _text(reply[1])
                )
        return StatusReport(ReportOutcome.APPLIED)

    def reserve_worker(self, worker_id: str, task_id: str, dispatch_id: str) -> bool:
        """Mark a worker BUSY for a dispatch about to be published to it; returns
        whether the worker is registered."""
        ts = now_iso()
        reserved = self._rds.sync.eval(
            _RESERVE_IF_REGISTERED,
            2,
            WORKERS_SET_KEY,
            worker_key(worker_id),
            worker_id,
            ts,
            dispatch_id,
            task_id,
        )
        if not int(reserved):
            return False
        self._announce_status(worker_id, WorkerStatus.BUSY, ts)
        return True

    def release_worker(self, worker_id: str, dispatch_id: str) -> bool:
        """Return a worker still reserved for ``dispatch_id`` to the status it last
        reported, IDLE if none; returns whether it was reserved for it."""
        ts = now_iso()
        released = _text(
            self._rds.sync.eval(
                _RELEASE_IF_RESERVED,
                2,
                WORKERS_SET_KEY,
                worker_key(worker_id),
                worker_id,
                ts,
                dispatch_id,
            )
        )
        if not released:
            return False
        self._announce_status(worker_id, WorkerStatus(released), ts)
        return True

    def _announce_status(self, worker_id: str, status: WorkerStatus, ts: str) -> None:
        """Announce a status the registry stored; a failed announcement leaves the
        stored status as it is."""
        payload = {
            "type": "STATUS",
            "worker_id": worker_id,
            "status": status.value,
            "ts": ts,
            "origin": "server",
        }
        try:
            self._rds.sync.publish_telemetry(
                WORKER_EVENT_CHANNEL, json.dumps(payload, ensure_ascii=False)
            )
        except Exception as exc:
            logger.warning(
                "Failed to announce worker %s as %s: %s", worker_id, status, exc
            )

    def reservation(self, worker_id: str) -> Reservation | None:
        """The dispatch a worker is reserved for, if any."""
        task_id, dispatch_id = self._rds.sync.hash_mget(
            worker_key(worker_id), ["reserved_task", "reserved_dispatch"]
        )
        if not task_id or not dispatch_id:
            return None
        return Reservation(worker_id, _text(task_id), _text(dispatch_id))

    def reservations(self) -> list[Reservation]:
        """Every registered worker's reservation."""
        worker_ids = sorted(self.get_worker_ids())
        with self._rds.sync.control_pipeline() as pipe:
            for worker_id in worker_ids:
                pipe.hmget(
                    worker_key(worker_id), ["reserved_task", "reserved_dispatch"]
                )
            replies = pipe.execute()
        return [
            Reservation(worker_id, _text(task_id), _text(dispatch_id))
            for worker_id, (task_id, dispatch_id) in zip(worker_ids, replies)
            if task_id and dispatch_id
        ]

    def reap_stale_worker(self, worker_id: str) -> bool:
        """Delete a worker's record while its heartbeat is stale; returns whether it
        was deleted."""
        reaped = self._rds.sync.eval(
            _REAP_IF_STALE,
            3,
            WORKERS_SET_KEY,
            worker_key(worker_id),
            worker_hb_key(worker_id),
            worker_id,
        )
        return bool(int(reaped))

    def unregister_workers(self, *worker_ids: str) -> None:
        with self._rds.sync.control_pipeline() as pipe:
            pipe.srem(WORKERS_SET_KEY, *worker_ids)
            pipe.delete(*(worker_key(worker_id) for worker_id in worker_ids))
            pipe.delete(*(worker_hb_key(worker_id) for worker_id in worker_ids))
            pipe.execute()

    def unregister_node_worker(self, worker_id: str, node_alias: str) -> bool:
        """Delete a worker's record unless another node wrote it; returns False when
        another node holds the id."""
        return bool(
            self._rds.sync.eval(
                _UNREGISTER_IF_NODE,
                3,
                WORKERS_SET_KEY,
                worker_key(worker_id),
                worker_hb_key(worker_id),
                worker_id,
                node_alias,
            )
        )

    async def unregister_node_worker_async(
        self, worker_id: str, node_alias: str
    ) -> bool:
        """Delete a worker's record unless another node wrote it; returns False when
        another node holds the id."""
        return bool(
            await self._rds.asyncio.eval(
                _UNREGISTER_IF_NODE,
                3,
                WORKERS_SET_KEY,
                worker_key(worker_id),
                worker_hb_key(worker_id),
                worker_id,
                node_alias,
            )
        )

    async def unregister_workers_async(self, *worker_ids: str) -> None:
        async with self._rds.asyncio.control_pipeline() as pipe:
            pipe.srem(WORKERS_SET_KEY, *worker_ids)
            pipe.delete(*(worker_key(worker_id) for worker_id in worker_ids))
            pipe.delete(*(worker_hb_key(worker_id) for worker_id in worker_ids))
            await pipe.execute()

    # ------------------------------------------------------------------ #
    # Worker query helpers
    # ------------------------------------------------------------------ #

    def get_worker_ids(self) -> set[str]:
        return self._rds.sync.set_members(WORKERS_SET_KEY)

    async def get_worker_ids_async(self) -> set[str]:
        return await self._rds.asyncio.set_members(WORKERS_SET_KEY)

    def get_worker(self, worker_id: str) -> Worker | None:
        raw = self._rds.sync.hash_getall(worker_key(worker_id))
        return _parse_worker_from_redis(worker_id, raw)

    async def get_worker_async(self, worker_id: str) -> Worker | None:
        raw = await self._rds.asyncio.hash_getall(worker_key(worker_id))
        return _parse_worker_from_redis(worker_id, raw)

    def get_workers(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        with self._rds.sync.control_pipeline() as pipe:
            for worker_id in worker_ids:
                pipe.hgetall(worker_key(worker_id))
            raws: list[dict] = pipe.execute()
        return [
            _parse_worker_from_redis(worker_id, raw)
            for worker_id, raw in zip(worker_ids, raws)
        ]

    async def get_workers_async(self, worker_ids: Sequence[str]) -> list[Worker | None]:
        async with self._rds.asyncio.control_pipeline() as pipe:
            for worker_id in worker_ids:
                pipe.hgetall(worker_key(worker_id))
            raws: list[dict] = await pipe.execute()
        return [
            _parse_worker_from_redis(worker_id, raw)
            for worker_id, raw in zip(worker_ids, raws)
        ]

    def is_worker_stale(self, worker_id: str) -> bool:
        ttl = self._rds.sync.ttl(worker_hb_key(worker_id))
        return ttl is None or ttl < 0

    async def is_worker_stale_async(self, worker_id: str) -> bool:
        ttl = await self._rds.asyncio.ttl(worker_hb_key(worker_id))
        return ttl is None or ttl < 0

    def get_worker_heartbeat(self, worker_id: str) -> str | None:
        return self._rds.sync.get(worker_hb_key(worker_id))

    async def get_worker_heartbeat_async(self, worker_id: str) -> str | None:
        return await self._rds.asyncio.get(worker_hb_key(worker_id))

    def worker_exists(self, worker_id: str) -> bool:
        return self._rds.sync.exists(worker_key(worker_id))

    async def worker_exists_async(self, worker_id: str) -> bool:
        return await self._rds.asyncio.exists(worker_key(worker_id))

    def list_workers(self) -> list[WorkerInfo]:
        results: list[WorkerInfo] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker:
                continue
            stale = self.is_worker_stale(worker_id)
            if not worker.last_seen:
                hb_ts = self.get_worker_heartbeat(worker_id)
                if hb_ts:
                    worker.last_seen = hb_ts
            results.append(
                WorkerInfo(
                    **worker.model_dump(),
                    stale=stale,
                )
            )
        return results

    async def list_workers_async(self) -> list[WorkerInfo]:
        results: list[WorkerInfo] = []
        for worker_id in await self.get_worker_ids_async():
            worker = await self.get_worker_async(worker_id)
            if not worker:
                continue
            stale = await self.is_worker_stale_async(worker_id)
            if not worker.last_seen:
                hb_ts = await self.get_worker_heartbeat_async(worker_id)
                if hb_ts:
                    worker.last_seen = hb_ts
            results.append(
                WorkerInfo(
                    **worker.model_dump(),
                    stale=stale,
                )
            )
        return results

    def record_worker_cache(
        self,
        worker_id: str,
        models: Iterable[str] | None = None,
        datasets: Iterable[str] | None = None,
    ) -> None:
        models_list = [v for v in (models or []) if _normalize_cache_value(v)]
        datasets_list = [v for v in (datasets or []) if _normalize_cache_value(v)]
        if not models_list and not datasets_list:
            return

        try:
            current_models_json, current_datasets_json = self._rds.sync.hash_mget(
                worker_key(worker_id), ["cache_models_json", "cache_datasets_json"]
            )
        except Exception:
            return

        try:
            current_models = (
                json.loads(current_models_json) if current_models_json else []
            )
            if not isinstance(current_models, list):
                current_models = []
        except Exception:
            current_models = []

        try:
            current_datasets = (
                json.loads(current_datasets_json) if current_datasets_json else []
            )
            if not isinstance(current_datasets, list):
                current_datasets = []
        except Exception:
            current_datasets = []

        updated_models = _merge_unique(current_models, models_list)
        updated_datasets = _merge_unique(current_datasets, datasets_list)

        mapping: dict[str, str] = {}
        if updated_models != current_models:
            mapping["cache_models_json"] = json.dumps(
                updated_models, ensure_ascii=False
            )
        if updated_datasets != current_datasets:
            mapping["cache_datasets_json"] = json.dumps(
                updated_datasets, ensure_ascii=False
            )
        if mapping:
            mapping["cache_updated_ts"] = now_iso()
        if not mapping:
            return

        try:
            self._set_worker_fields(worker_id, mapping)
        except Exception:
            return

    def idle_satisfying_pool(
        self, task: TaskEnvelope, relays_only: bool
    ) -> list[Worker]:
        """Idle, non-stale workers that can run a dispatch of ``task`` now: those
        ``satisfying_workers`` returns, less any whose free GPUs fall short."""
        available: list[Worker] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker or worker.status is not WorkerStatus.IDLE:
                continue
            if self.is_worker_stale(worker.id):
                continue
            if (
                hw_satisfies(worker, task)
                and capability_satisfies(worker, task)
                and gpu_available_for(worker, task, relays_only)
            ):
                available.append(worker)
        return self.sort_workers(available)

    def satisfying_workers(self, task: TaskEnvelope) -> list[Worker]:
        """Non-stale workers whose hardware and capabilities satisfy the task."""
        available: list[Worker] = []
        for worker_id in self.get_worker_ids():
            worker = self.get_worker(worker_id)
            if not worker or self.is_worker_stale(worker.id):
                continue
            if hw_satisfies(worker, task) and capability_satisfies(worker, task):
                available.append(worker)
        return self.sort_workers(available)

    def sort_workers(self, workers: list[Worker]) -> list[Worker]:
        decorated: list[tuple[Worker, int, int, int, int]] = []
        for worker in workers:
            hardware = worker.hardware
            gpu_entries = [] if hardware is None else hardware.gpu.devices
            gpu_count = len(gpu_entries)
            total_vram = dedicated_gpu_memory_total_bytes(hardware)
            sys_ram = 0 if hardware is None else (hardware.memory.total_bytes or 0)
            cpu_cores = 0 if hardware is None else hardware.cpu.logical_cores
            decorated.append((worker, gpu_count, total_vram, sys_ram, cpu_cores))

        decorated.sort(key=lambda item: item[0].id)
        decorated.sort(
            key=lambda item: (
                item[1] > 0,
                item[2],
                item[3],
                item[4],
            ),
            reverse=True,
        )
        return [item[0] for item in decorated]

    def publish_task(self, worker: Worker, payload: WorkerTaskMessage) -> int:
        payload_json = payload.model_dump(mode="json", exclude_none=True, by_alias=True)
        message = TaskMessage(
            worker_id=worker.id,
            payload=payload_json,
        ).model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_task_async(
        self, worker: Worker, payload: WorkerTaskMessage
    ) -> int:
        payload_json = payload.model_dump(mode="json", exclude_none=True, by_alias=True)
        message = TaskMessage(
            worker_id=worker.id,
            payload=payload_json,
        ).model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_interrupt(self, worker: Worker, payload: InterruptMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_interrupt_async(
        self, worker: Worker, payload: InterruptMessage
    ) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_revoke(self, node_id: str, payload: RevokeMessage) -> int:
        message = payload.model_dump_json()
        return self._rds.sync.publish_control(node_dispatch_channel(node_id), message)

    async def publish_revoke_async(self, node_id: str, payload: RevokeMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_stop(self, worker: Worker, payload: StopMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_stop_async(self, worker: Worker, payload: StopMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def publish_mediated_op(self, worker: Worker, payload: MediatedOpMessage) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return self._rds.sync.publish_control(channel, message)

    async def publish_mediated_op_async(
        self, worker: Worker, payload: MediatedOpMessage
    ) -> int:
        message = payload.model_dump_json()
        channel = node_dispatch_channel(worker.node_id)
        return await self._rds.asyncio.publish_control(channel, message)

    def record_gpu_availability(
        self, worker_id: str, availability: dict[str, Any]
    ) -> bool:
        """Store a heartbeat's per-device availability, latched until the worker
        reports again; an empty map clears it."""
        return self._set_worker_fields(
            worker_id,
            {"gpu_availability_json": json.dumps(availability, ensure_ascii=False)},
        )

    def _set_worker_fields(self, worker_id: str, mapping: dict[str, str]) -> bool:
        wrote = self._rds.sync.eval(
            _SET_FIELDS_IF_REGISTERED,
            2,
            WORKERS_SET_KEY,
            worker_key(worker_id),
            worker_id,
            *_flatten_fields(mapping),
        )
        return bool(int(wrote))


# --- Helper functions --- #


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _normalize_cache_value(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _merge_unique(existing: list[str], additions: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in existing:
        normalized = _normalize_cache_value(value)
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        merged.append(normalized)
    for value in additions:
        normalized = _normalize_cache_value(value)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged


def hw_satisfies(worker: Worker, task: TaskEnvelope) -> bool:
    resources = task.spec.resources
    if resources is None:
        return True
    requirements = resources.hardware
    if requirements is None:
        return True

    hw = worker.hardware
    cpu_needed = requirements.cpu
    mem_needed = requirements.memory
    gpu_req = requirements.gpu

    # Consider SSH hardware limits for SSH tasks.
    ssh_caps = (
        worker.ssh_limits
        if isinstance(task.spec, (SSHSpecStrict, SSHSpecTemplate))
        and worker.ssh_limits is not None
        else None
    )

    if cpu_needed is not None:
        cpu_cores: float | None = None if hw is None else hw.cpu.logical_cores
        if (
            ssh_caps is not None
            and ssh_caps.max_cpu_cores is not None
            and cpu_cores is not None
        ):
            cpu_cores = min(cpu_cores, ssh_caps.max_cpu_cores)
        if cpu_cores is None or cpu_cores < cpu_needed:
            return False

    if mem_needed:
        required_bytes = parse_mem_to_bytes(str(mem_needed)) or 0
        available = 0 if hw is None else (hw.memory.total_bytes or 0)
        if (
            ssh_caps is not None
            and ssh_caps.max_memory_bytes is not None
            and available > 0
        ):
            available = min(available, ssh_caps.max_memory_bytes)
        if available < required_bytes:
            return False

    if gpu_req:
        if hw is None:
            return False
        if not gpu_meets_requirements(hw, gpu_req):
            return False

    return True


def gpu_available_for(worker: Worker, task: TaskEnvelope, relays_only: bool) -> bool:
    """Whether a dispatch of ``task`` fits this worker's GPUs that nothing else holds.

    Applied when choosing among idle workers, never in ``hw_satisfies``: a held card
    is transient, so its worker stays in ``satisfying_workers`` and the task waits
    rather than failing as unschedulable.
    """
    hw = worker.hardware
    return hw is None or gpus_fit_dispatch(hw, task.spec, relays_only)


def capability_satisfies(worker: Worker, task: TaskEnvelope) -> bool:
    capabilities = worker.capabilities
    spec = task.spec
    if spec.taskType not in capabilities.supported_task_types:
        return False
    if isinstance(spec, SSHSpecStrict | SSHSpecTemplate) and not spec.interactive:
        return capabilities.ssh_noninteractive
    return True


def dedicated_gpu_memory_total_bytes(hw: WorkerHardware | None) -> int:
    if hw is None:
        return 0
    total = 0
    for entry in hw.gpu.devices:
        total += entry.memory_total_bytes or 0
    return total


def _parse_worker_from_redis(
    worker_id: str, value: dict[str, Any] | None
) -> Worker | None:
    if not value:
        return None

    def _loads(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def _ensure_str_list(items: Any) -> list[str]:
        if not isinstance(items, list):
            return []
        result: list[str] = []
        for item in items:
            if isinstance(item, str):
                norm = item.strip()
                if norm:
                    result.append(norm)
        return result

    env = _loads(value.get("env_json"), {})
    hardware_json = value.get("hardware_json")
    hardware = (
        None
        if hardware_json is None
        else WorkerHardware.model_validate_json(hardware_json)
    )
    _merge_gpu_availability(hardware, _loads(value.get("gpu_availability_json"), {}))
    capabilities_json = value.get("capabilities_json")
    capabilities = (
        WorkerCapabilities()
        if capabilities_json is None
        else WorkerCapabilities.model_validate_json(capabilities_json)
    )
    ssh_limits_json = value.get("ssh_limits_json")
    ssh_limits = (
        None
        if ssh_limits_json is None
        else SSHLimits.model_validate_json(ssh_limits_json)
    )
    tags = _loads(value.get("tags_json"), [])
    cached_models = _ensure_str_list(_loads(value.get("cache_models_json"), []))
    cached_datasets = _ensure_str_list(_loads(value.get("cache_datasets_json"), []))
    cache_updated_ts = value.get("cache_updated_ts") or None

    pid_val = value.get("pid")
    try:
        pid = int(pid_val) if pid_val is not None else None
    except (TypeError, ValueError):
        pid = None

    try:
        incarnation = int(value.get("incarnation", 0) or 0)
    except (TypeError, ValueError):
        incarnation = 0

    cost_val = value.get("cost_per_hour")
    try:
        cost_per_hour = float(cost_val) if cost_val is not None else None
    except (TypeError, ValueError):
        cost_per_hour = None

    return Worker(
        id=value.get("id", worker_id),
        alias=value.get("alias"),
        namespace=value.get("namespace", ""),
        cluster=value.get("cluster", ""),
        node_id=value.get("node_id", ""),
        node_alias=value.get("node_alias", ""),
        version=value.get("version"),
        status=WorkerStatus(value.get("status", "UNKNOWN")),
        started_at=value.get("started_at"),
        pid=pid,
        incarnation=incarnation,
        env=env,
        hardware=hardware,
        capabilities=capabilities,
        ssh_limits=ssh_limits,
        tags=tags,
        last_seen=value.get("last_seen"),
        cached_models=cached_models,
        cached_datasets=cached_datasets,
        cache_updated_ts=cache_updated_ts,
        cost_per_hour=cost_per_hour,
    )
