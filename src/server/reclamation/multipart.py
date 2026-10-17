"""Registered multipart uploads, each by its exact UploadId and surviving owner.

A writer registers an upload before it sends any part, so every upload with parts a
writer of this protocol started is known here by its exact id. Each record names the
logical owner whose life keeps the upload live — the dispatch writing it, the upload
obligation a success leaves owed once it accepts the dispatch's staged artifacts, or
the outcome finalization it materializes — and the worker registration that started
it. A record being aborted dooms the UploadId: its writer starts another.

A scope's records live in one hash. Uploads owned by a dispatch are indexed by it, so
moving them to the obligation the dispatch's success leaves costs one call per
dispatch, proportional to its own uploads; once moved, a dispatch's later
registrations take the obligation directly, until the obligation settles and its index
is released. A record leaves once a listing of the store's open uploads that started
after its registration no longer names its UploadId: the upload was completed or
aborted, and an UploadId never opens again.
"""

import json
import logging
import time
from collections.abc import Set
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from shared.content.multipart import MULTIPART_ACK_FRAME_KIND, MultipartVerdict
from shared.schemas.command import MediatedOpMessage

from ..clients.redis import (
    RECLAIM_SCOPES_KEY,
    RedisClient,
    content_finalization_key,
    content_finalization_scope_key,
    reclaim_multipart_dispatch_key,
    reclaim_multipart_key,
    worker_key,
)
from .lua import KINDS

if TYPE_CHECKING:
    from ..registries.worker import WorkerRegistry

# KEYS: records, dispatch index, scopes, registering worker[, finalization scope,
# finalization]. ARGV: upload id, object key, owner, registration, initiated at (ms),
# scope.
_REGISTER = KINDS + """
local wrong = wrong_kind({'hash', 'hash', 'set', 'hash', 'string', 'string'})
if wrong then
  return redis.error_reply('WRONGTYPE ' .. wrong)
end
local current = redis.call('HGET', KEYS[1], ARGV[1])
if current then
  local record = cjson.decode(current)
  if record.state ~= 'registered' then
    return {'doomed'}
  end
  if record.registration == ARGV[4] and record.key == ARGV[2] then
    return {'ack', current}
  end
  return {'unowned'}
end
local owner = cjson.decode(ARGV[3])
local indexed = false
if owner.kind == 'finalization' then
  if redis.call('GET', KEYS[5]) ~= ARGV[6] or redis.call('EXISTS', KEYS[6]) == 1 then
    return {'unowned'}
  end
elseif owner.kind == 'dispatch' then
  local moved = redis.call('HGET', KEYS[2], '~moved')
  if moved then
    owner = cjson.decode(moved)
  else
    indexed = true
  end
end
local record = cjson.encode({
  key = ARGV[2],
  state = 'registered',
  owner = owner,
  registration = ARGV[4],
  initiated_at = tonumber(ARGV[5]),
  content_protocol = tonumber(redis.call('HGET', KEYS[4], 'content_protocol')) or 0,
})
if indexed then
  redis.call('HSET', KEYS[2], ARGV[1], '1')
end
redis.call('HSET', KEYS[1], ARGV[1], record)
redis.call('SADD', KEYS[3], ARGV[6])
return {'ack', record}
"""

# KEYS: records, dispatch index. ARGV: dispatch id, owner.
_TRANSFER = KINDS + """
local wrong = wrong_kind(span(1, 2, 'hash'))
if wrong then
  return redis.error_reply('WRONGTYPE ' .. wrong)
end
if redis.call('HEXISTS', KEYS[2], '~moved') == 1 then
  return 0
end
local owner = cjson.decode(ARGV[2])
local moving, ids = {}, {}
for _, id in ipairs(redis.call('HKEYS', KEYS[2])) do
  table.insert(ids, id)
  local current = redis.call('HGET', KEYS[1], id)
  if current then
    local record = cjson.decode(current)
    if record.state == 'registered' and record.owner.kind == 'dispatch'
        and record.owner.dispatch_id == ARGV[1] then
      record.owner = owner
      moving[id] = cjson.encode(record)
    end
  end
end
local moved = 0
for id, record in pairs(moving) do
  redis.call('HSET', KEYS[1], id, record)
  moved = moved + 1
end
for _, id in ipairs(ids) do
  redis.call('HDEL', KEYS[2], id)
end
redis.call('HSET', KEYS[2], '~moved', ARGV[2])
return moved
"""


# KEYS: records, then each candidate's dispatch index.
# ARGV: id1, record1, ..., idn, recordn.
_PRUNE = """
if redis.call('TYPE', KEYS[1]).ok ~= 'hash' then
  return 0
end
local pruned = 0
for i = 1, #ARGV / 2 do
  local id = ARGV[2 * i - 1]
  if redis.call('HGET', KEYS[1], id) == ARGV[2 * i] then
    redis.call('HDEL', KEYS[1], id)
    if redis.call('TYPE', KEYS[i + 1]).ok == 'hash' then
      redis.call('HDEL', KEYS[i + 1], id)
    end
    pruned = pruned + 1
  end
end
return pruned
"""

# Records one prune reads and removes at a time.
PRUNE_PAGE = 256


class OwnerKind(StrEnum):
    DISPATCH = "dispatch"
    UPLOAD = "upload"
    """The upload a dispatch's success left owed of its accepted artifacts."""
    FINALIZATION = "finalization"


@dataclass(frozen=True)
class MultipartOwner:
    """The logical owner an upload stays live under."""

    kind: OwnerKind
    dispatch_id: str = ""
    task_id: str = ""
    idempotency_key: str = ""

    @classmethod
    def dispatch(cls, dispatch_id: str, task_id: str) -> "MultipartOwner":
        return cls(OwnerKind.DISPATCH, dispatch_id=dispatch_id, task_id=task_id)

    @classmethod
    def upload(cls, dispatch_id: str) -> "MultipartOwner":
        return cls(OwnerKind.UPLOAD, dispatch_id=dispatch_id)

    @classmethod
    def finalization(cls, idempotency_key: str) -> "MultipartOwner":
        return cls(OwnerKind.FINALIZATION, idempotency_key=idempotency_key)

    def encode(self) -> str:
        fields: dict[str, str] = {"kind": self.kind.value}
        match self.kind:
            case OwnerKind.DISPATCH:
                fields |= {"dispatch_id": self.dispatch_id, "task_id": self.task_id}
            case OwnerKind.UPLOAD:
                fields["dispatch_id"] = self.dispatch_id
            case OwnerKind.FINALIZATION:
                fields["idm"] = self.idempotency_key
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Registration:
    """One registration's verdict, with the record it left when acknowledged."""

    verdict: MultipartVerdict
    record: dict[str, Any] | None = None


class MultipartStore:
    """Registers multipart uploads and moves their ownership in the control store."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def register(
        self,
        scope: str,
        upload_id: str,
        key: str,
        owner: MultipartOwner,
        registration: str,
        now: float | None = None,
    ) -> Registration:
        """Register an upload under ``owner``, or answer why it may not proceed.

        ``registration`` is the registering worker's id; the record carries the
        content protocol that worker reported when it registered, or 0 when it
        reported none. A second registration of the same upload by the same
        registration is answered with the record already there. A finalization owns an
        upload only while control assigned it this scope and its winner is not yet
        recorded.
        """
        keys = [
            reclaim_multipart_key(scope),
            reclaim_multipart_dispatch_key(scope, owner.dispatch_id),
            RECLAIM_SCOPES_KEY,
            worker_key(registration),
        ]
        if owner.kind is OwnerKind.FINALIZATION:
            keys += [
                content_finalization_scope_key(owner.idempotency_key),
                content_finalization_key(scope, owner.idempotency_key),
            ]
        initiated_at = int((time.time() if now is None else now) * 1000)
        reply = list(
            self._rds.sync.eval(
                _REGISTER,
                len(keys),
                *keys,
                upload_id,
                key,
                owner.encode(),
                registration,
                str(initiated_at),
                scope,
            )
        )
        verdict = MultipartVerdict(str(reply[0]))
        record = json.loads(reply[1]) if len(reply) > 1 else None
        return Registration(verdict, record)

    def transfer(self, scope: str, dispatch_id: str, owner: MultipartOwner) -> int:
        """Move every upload a dispatch owns to ``owner``; returns how many moved.

        Each moved record keeps the registration and time that started it, and the
        dispatch's later registrations take ``owner`` directly. A second transfer of
        the same dispatch moves nothing.
        """
        keys = [
            reclaim_multipart_key(scope),
            reclaim_multipart_dispatch_key(scope, dispatch_id),
        ]
        return int(
            self._rds.sync.eval(
                _TRANSFER, len(keys), *keys, dispatch_id, owner.encode()
            )
        )

    async def transfer_async(
        self, scope: str, dispatch_id: str, owner: MultipartOwner
    ) -> int:
        keys = [
            reclaim_multipart_key(scope),
            reclaim_multipart_dispatch_key(scope, dispatch_id),
        ]
        return int(
            await self._rds.asyncio.eval(
                _TRANSFER, len(keys), *keys, dispatch_id, owner.encode()
            )
        )

    def release(self, scope: str, dispatch_id: str) -> None:
        """Drop a dispatch's index once its life as an owner has ended: the dispatch
        ended owing no upload, or the upload its success left owed settled.

        The records it owned stay until the store no longer holds their uploads open.
        """
        self._rds.sync.delete(reclaim_multipart_dispatch_key(scope, dispatch_id))

    async def release_async(self, scope: str, dispatch_id: str) -> None:
        await self._rds.asyncio.delete(
            reclaim_multipart_dispatch_key(scope, dispatch_id)
        )

    def prune(self, scope: str, open_ids: Set[str], registered_before: float) -> int:
        """Remove each record registered before ``registered_before`` whose UploadId
        ``open_ids`` does not name; returns how many were removed.

        ``open_ids`` is every upload the store held open in a listing that started at
        ``registered_before`` or later, so an upload it does not name was completed or
        aborted. A record that changed since it was read is left for the next prune.
        """
        cutoff_ms = registered_before * 1000
        records = reclaim_multipart_key(scope)
        pruned = 0
        cursor = 0
        while True:
            with self._rds.sync.control_pipeline(transaction=False) as pipe:
                pipe.hscan(records, cursor, count=PRUNE_PAGE)
                next_cursor, page = pipe.execute()[0]
            cursor = int(next_cursor)
            closed: list[tuple[str, str, str]] = []
            for upload_id, raw in page.items():
                if upload_id in open_ids:
                    continue
                try:
                    record = json.loads(raw)
                    initiated_at = float(record["initiated_at"])
                    dispatch_id = str(record["owner"].get("dispatch_id") or "")
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
                if initiated_at < cutoff_ms:
                    closed.append((upload_id, raw, dispatch_id))
            if closed:
                keys = [records]
                keys += [
                    reclaim_multipart_dispatch_key(scope, dispatch_id)
                    for _, _, dispatch_id in closed
                ]
                args = [
                    value for upload_id, raw, _ in closed for value in (upload_id, raw)
                ]
                pruned += int(self._rds.sync.eval(_PRUNE, len(keys), *keys, *args))
            if cursor == 0:
                return pruned


class OwnerResolver(Protocol):
    def __call__(
        self, task_id: str, worker_id: str, dispatch_id: str | None
    ) -> tuple[str, MultipartOwner] | None: ...


class MultipartRegistrations:
    """Answers a worker's registration of an upload it is about to send parts of.

    The owner is control's own: an outcome finalization the worker names is checked
    against the scope control assigned it, and otherwise the runtime resolves the
    dispatch holding the task on that worker, or the upload obligation its success
    left owed. A worker that holds neither in the scope it names owns nothing.
    """

    def __init__(
        self,
        store: MultipartStore,
        owner_of: OwnerResolver,
        worker_registry: "WorkerRegistry",
        logger: logging.Logger,
    ) -> None:
        self._store = store
        self._owner_of = owner_of
        self._workers = worker_registry
        self._logger = logger

    def on_registration(self, worker_id: str, payload: dict[str, Any]) -> None:
        upload_id = str(payload.get("upload_id") or "")
        if not upload_id:
            return
        verdict = self._verdict(worker_id, upload_id, payload)
        if verdict is None:
            return
        if verdict is not MultipartVerdict.ACK:
            self._logger.warning(
                "Multipart upload %s from %s is %s", upload_id, worker_id, verdict
            )
        self._answer(worker_id, upload_id, verdict)

    def _verdict(
        self, worker_id: str, upload_id: str, payload: dict[str, Any]
    ) -> MultipartVerdict | None:
        """The verdict on one registration, or None when the store could not decide,
        which leaves the writer to register again or give the write up."""
        scope = str(payload.get("scope") or "")
        key = str(payload.get("key") or "")
        task_id = str(payload.get("task_id") or "")
        dispatch_id = payload.get("dispatch_id")
        idempotency_key = payload.get("idempotency_key")
        if not scope or not key or not task_id:
            return MultipartVerdict.UNOWNED
        if isinstance(idempotency_key, str) and idempotency_key:
            owner = MultipartOwner.finalization(idempotency_key)
        elif (
            resolved := self._owner_of(
                task_id,
                worker_id,
                dispatch_id if isinstance(dispatch_id, str) else None,
            )
        ) is not None and resolved[0] == scope:
            owner = resolved[1]
        else:
            return MultipartVerdict.UNOWNED
        try:
            return self._store.register(scope, upload_id, key, owner, worker_id).verdict
        except Exception:
            self._logger.exception(
                "Could not register multipart upload %s from %s", upload_id, worker_id
            )
            return None

    def _answer(
        self, worker_id: str, upload_id: str, verdict: MultipartVerdict
    ) -> None:
        if (worker := self._workers.get_worker(worker_id)) is None:
            return
        self._workers.publish_mediated_op(
            worker,
            MediatedOpMessage(
                worker_id=worker_id,
                frame_kind=MULTIPART_ACK_FRAME_KIND,
                payload={"upload_id": upload_id, "verdict": verdict.value},
            ),
        )


__all__ = [
    "MultipartOwner",
    "MultipartRegistrations",
    "MultipartStore",
    "OwnerKind",
    "OwnerResolver",
    "Registration",
]
