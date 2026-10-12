"""Redis persistence for the resident-capacity control facts, in the layout and under
the contract ``resident.durability`` defines."""

from collections.abc import Mapping, Sequence
from typing import Any

from redis.exceptions import ResponseError

from ..clients.redis import (
    RedisClient,
    resident_cs_key,
    resident_family_history_key,
    resident_replica_history_key,
    resident_retired_key,
    serve_cs_key,
)
from ..resident.durability import (
    ResidentStateError,
    decode_live,
    decode_retired,
    decode_stored,
)
from ..resident.state import (
    INERT_REPLICA_STATES,
    ClaimState,
    ResidentSnapshot,
    RetiredInvocation,
    ServiceFamily,
    StoredReplica,
)
from ..serve import ServeSnapshot

_CHECK_TYPES = """
for _, key in ipairs(KEYS) do
  local kind = redis.call('TYPE', key)['ok']
  if kind ~= 'hash' and kind ~= 'none' then
    return redis.error_reply('WRONGTYPE ' .. key)
  end
end
"""

_LOAD_CHECK = _CHECK_TYPES + "return 1"

# KEYS: live, retired. ARGV: invocation id, record, its epoch high, the terminal claim
# state, then the live fields to delete, the request's first.
_RETIRE = _CHECK_TYPES + """
local request = redis.call('HGET', KEYS[1], ARGV[5])
if request then
  local consumed = cjson.decode(request)['consumed']
  if consumed == nil or consumed == cjson.null then
    return redis.error_reply('NOTCONSUMED ' .. ARGV[1])
  end
end
for i = 6, #ARGV do
  local claim = redis.call('HGET', KEYS[1], ARGV[i])
  if claim and cjson.decode(claim)['state'] ~= ARGV[4] then
    return redis.error_reply('OPENCLAIM ' .. ARGV[i])
  end
end
local stored = redis.call('HGET', KEYS[2], ARGV[1])
if stored then
  if tonumber(cjson.decode(stored)['epoch_high']) > tonumber(ARGV[3]) then
    return redis.error_reply('EPOCH ' .. ARGV[1])
  end
elseif not request then
  return redis.error_reply('NOEVIDENCE ' .. ARGV[1])
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[1], unpack(ARGV, 5))
return 1
"""

# KEYS: live, history. ARGV: history field, value, the inert replica states the first
# live field must be in (comma-joined, empty for none), then the live fields to delete.
_ARCHIVE = _CHECK_TYPES + """
if ARGV[3] ~= '' then
  local replica = redis.call('HGET', KEYS[1], ARGV[4])
  if replica then
    local state = cjson.decode(replica)['state']
    if not string.find(',' .. ARGV[3] .. ',', ',' .. state .. ',', 1, true) then
      return redis.error_reply('NOTINERT ' .. ARGV[4])
    end
  end
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[1], unpack(ARGV, 4))
return 1
"""

_INERT_ARG = ",".join(sorted(INERT_REPLICA_STATES))
_LOAD_KEYS = (
    resident_cs_key(),
    resident_retired_key(),
    resident_replica_history_key(),
    resident_family_history_key(),
)


def _refused(exc: ResponseError) -> ResidentStateError:
    return ResidentStateError(f"resident state refused the write: {exc}")


def _not_current(exc: ResponseError) -> ResidentStateError:
    return ResidentStateError(
        f"resident state is not in the current format ({exc}); delete the named key "
        "and restart"
    )


def _split(changes: Mapping[str, str | None]) -> tuple[dict[str, str], list[str]]:
    """Split changes into the fields to set and the fields to delete."""
    mapping = {field: value for field, value in changes.items() if value is not None}
    return mapping, [field for field, value in changes.items() if value is None]


def _queue_write(pipe: Any, changes: Mapping[str, str | None]) -> None:
    mapping, deleted = _split(changes)
    if mapping:
        pipe.hset(resident_cs_key(), mapping=mapping)
    if deleted:
        pipe.hdel(resident_cs_key(), *deleted)


def _retire_args(record: RetiredInvocation, fields: Sequence[str]) -> tuple[Any, ...]:
    return (
        _RETIRE,
        2,
        resident_cs_key(),
        resident_retired_key(),
        record.invocation_id,
        record.model_dump_json(),
        str(record.epoch_high),
        ClaimState.TERMINAL.value,
        *fields,
    )


def _replica_args(stored: StoredReplica, fields: Sequence[str]) -> tuple[Any, ...]:
    return (
        _ARCHIVE,
        2,
        resident_cs_key(),
        resident_replica_history_key(),
        stored.replica.replica_id,
        stored.model_dump_json(),
        _INERT_ARG,
        *fields,
    )


def _family_args(family: ServiceFamily, field: str) -> tuple[Any, ...]:
    return (
        _ARCHIVE,
        2,
        resident_cs_key(),
        resident_family_history_key(),
        family.family,
        family.model_dump_json(),
        "",
        field,
    )


def _retired(
    invocation_ids: Sequence[str], values: Sequence[str | None]
) -> list[RetiredInvocation | None]:
    return [
        decode_retired(invocation_id, value) if value is not None else None
        for invocation_id, value in zip(invocation_ids, values, strict=True)
    ]


def _family(values: Sequence[str | None]) -> ServiceFamily | None:
    [value] = values
    return decode_stored(ServiceFamily, value) if value is not None else None


def _serve(blob: str | None) -> ServeSnapshot | None:
    return ServeSnapshot.model_validate_json(blob) if blob else None


class ResidentRegistry:
    """Redis-backed durability for resident-capacity control facts."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def write(self, changes: Mapping[str, str | None]) -> None:
        with self._rds.sync.control_pipeline(transaction=True) as pipe:
            _queue_write(pipe, changes)
            pipe.execute()

    async def write_async(self, changes: Mapping[str, str | None]) -> None:
        async with self._rds.asyncio.control_pipeline(transaction=True) as pipe:
            _queue_write(pipe, changes)
            await pipe.execute()

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        try:
            self._rds.sync.eval(*_retire_args(record, fields))
        except ResponseError as exc:
            raise _refused(exc) from exc

    async def retire_invocation_async(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        try:
            await self._rds.asyncio.eval(*_retire_args(record, fields))
        except ResponseError as exc:
            raise _refused(exc) from exc

    def retired_invocations(
        self, invocation_ids: Sequence[str]
    ) -> list[RetiredInvocation | None]:
        ids = list(invocation_ids)
        return _retired(ids, self._rds.sync.hash_mget(resident_retired_key(), ids))

    async def retired_invocations_async(
        self, invocation_ids: Sequence[str]
    ) -> list[RetiredInvocation | None]:
        ids = list(invocation_ids)
        values = await self._rds.asyncio.hash_mget(resident_retired_key(), ids)
        return _retired(ids, values)

    def archive_replica(self, stored: StoredReplica, fields: Sequence[str]) -> None:
        try:
            self._rds.sync.eval(*_replica_args(stored, fields))
        except ResponseError as exc:
            raise _refused(exc) from exc

    async def archive_replica_async(
        self, stored: StoredReplica, fields: Sequence[str]
    ) -> None:
        try:
            await self._rds.asyncio.eval(*_replica_args(stored, fields))
        except ResponseError as exc:
            raise _refused(exc) from exc

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        try:
            self._rds.sync.eval(*_family_args(family, field))
        except ResponseError as exc:
            raise _refused(exc) from exc

    async def archive_family_async(self, family: ServiceFamily, field: str) -> None:
        try:
            await self._rds.asyncio.eval(*_family_args(family, field))
        except ResponseError as exc:
            raise _refused(exc) from exc

    def stored_family(self, family: str) -> ServiceFamily | None:
        return _family(
            self._rds.sync.hash_mget(resident_family_history_key(), [family])
        )

    async def stored_family_async(self, family: str) -> ServiceFamily | None:
        return _family(
            await self._rds.asyncio.hash_mget(resident_family_history_key(), [family])
        )

    def load_snapshot(self) -> ResidentSnapshot | None:
        """Load the live facts, ``None`` when nothing is stored.

        Refuses a resident key of the wrong type, naming the key to delete.
        """
        try:
            self._rds.sync.eval(_LOAD_CHECK, len(_LOAD_KEYS), *_LOAD_KEYS)
        except ResponseError as exc:
            raise _not_current(exc) from exc
        fields = self._rds.sync.hash_getall(resident_cs_key())
        return decode_live(fields) if fields else None

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        """Load the live facts, ``None`` when nothing is stored.

        Refuses a resident key of the wrong type, naming the key to delete.
        """
        try:
            await self._rds.asyncio.eval(_LOAD_CHECK, len(_LOAD_KEYS), *_LOAD_KEYS)
        except ResponseError as exc:
            raise _not_current(exc) from exc
        fields = await self._rds.asyncio.hash_getall(resident_cs_key())
        return decode_live(fields) if fields else None

    def stored_replicas(self) -> list[StoredReplica]:
        stored = self._rds.sync.hash_getall(resident_replica_history_key())
        return [decode_stored(StoredReplica, value) for value in stored.values()]

    async def stored_replicas_async(self) -> list[StoredReplica]:
        stored = await self._rds.asyncio.hash_getall(resident_replica_history_key())
        return [decode_stored(StoredReplica, value) for value in stored.values()]

    def stored_families(self) -> list[ServiceFamily]:
        stored = self._rds.sync.hash_getall(resident_family_history_key())
        return [decode_stored(ServiceFamily, value) for value in stored.values()]

    async def stored_families_async(self) -> list[ServiceFamily]:
        stored = await self._rds.asyncio.hash_getall(resident_family_history_key())
        return [decode_stored(ServiceFamily, value) for value in stored.values()]

    def save_serve_snapshot(self, snapshot: ServeSnapshot) -> None:
        self._rds.sync.set_value(serve_cs_key(), snapshot.model_dump_json())

    async def save_serve_snapshot_async(self, snapshot: ServeSnapshot) -> None:
        await self._rds.asyncio.set_value(serve_cs_key(), snapshot.model_dump_json())

    def load_serve_snapshot(self) -> ServeSnapshot | None:
        return _serve(self._rds.sync.get(serve_cs_key()))

    async def load_serve_snapshot_async(self) -> ServeSnapshot | None:
        return _serve(await self._rds.asyncio.get(serve_cs_key()))
