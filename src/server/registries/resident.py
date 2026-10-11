"""Durable persistence for the authoritative resident-capacity control facts.

The live facts are one hash, ``resident:cs``, with a field per entity; each step writes
only the fields it changed. A retired invocation's record, a retired replica's history
and a retired family's definition live in their own hashes, which a restart never loads.
Retirement and archiving run as one script each that refuses, before writing anything,
a key of the wrong type or a live state that does not support the move, then publishes
the record before it deletes the live fields.
"""

from collections.abc import Mapping, Sequence

from redis.exceptions import ResponseError

from ..clients.redis import (
    RedisClient,
    resident_cs_key,
    resident_family_history_key,
    resident_replica_history_key,
    resident_retired_key,
    serve_cs_key,
)
from ..resident.durability import ResidentStateError, decode_live
from ..resident.state import (
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

# KEYS: live, retired. ARGV: invocation id, record, its epoch high, then the live
# fields to delete, the request's first.
_RETIRE = _CHECK_TYPES + """
local request = redis.call('HGET', KEYS[1], ARGV[4])
if request then
  local consumed = cjson.decode(request)['consumed']
  if consumed == nil or consumed == cjson.null then
    return redis.error_reply('NOTCONSUMED ' .. ARGV[1])
  end
end
for i = 5, #ARGV do
  local claim = redis.call('HGET', KEYS[1], ARGV[i])
  if claim and cjson.decode(claim)['state'] ~= 'terminal' then
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
redis.call('HDEL', KEYS[1], unpack(ARGV, 4))
return 1
"""

# KEYS: live, history. ARGV: history field, value, '1' to require the first live field
# to be an inert replica, then the live fields to delete.
_ARCHIVE = _CHECK_TYPES + """
if ARGV[3] == '1' then
  local replica = redis.call('HGET', KEYS[1], ARGV[4])
  if replica then
    local state = cjson.decode(replica)['state']
    if state ~= 'stopped' and state ~= 'preempted' and state ~= 'failed' then
      return redis.error_reply('NOTINERT ' .. ARGV[4])
    end
  end
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[1], unpack(ARGV, 4))
return 1
"""


def _refused(exc: ResponseError) -> ResidentStateError:
    return ResidentStateError(f"resident state refused the write: {exc}")


class ResidentRegistry:
    """Redis-backed durability for resident-capacity control facts."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def write(self, changes: Mapping[str, str | None]) -> None:
        deleted = [field for field, value in changes.items() if value is None]
        mapping = {
            field: value for field, value in changes.items() if value is not None
        }
        with self._rds.sync.control_pipeline(transaction=True) as pipe:
            if mapping:
                pipe.hset(resident_cs_key(), mapping=mapping)
            if deleted:
                pipe.hdel(resident_cs_key(), *deleted)
            pipe.execute()

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        try:
            self._rds.sync.eval(
                _RETIRE,
                2,
                resident_cs_key(),
                resident_retired_key(),
                record.invocation_id,
                record.model_dump_json(),
                str(record.epoch_high),
                *fields,
            )
        except ResponseError as exc:
            raise _refused(exc) from exc

    def retired_invocations(self, invocation_ids: Sequence[str]) -> list[str | None]:
        return list(
            self._rds.sync.hash_mget(resident_retired_key(), list(invocation_ids))
        )

    def archive_replica(self, stored: StoredReplica, fields: Sequence[str]) -> None:
        self._archive(
            resident_replica_history_key(),
            stored.replica.replica_id,
            stored.model_dump_json(),
            fields,
            inert=True,
        )

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        self._archive(
            resident_family_history_key(),
            family.family,
            family.model_dump_json(),
            [field],
            inert=False,
        )

    def _archive(
        self,
        history_key: str,
        history_field: str,
        value: str,
        fields: Sequence[str],
        *,
        inert: bool,
    ) -> None:
        try:
            self._rds.sync.eval(
                _ARCHIVE,
                2,
                resident_cs_key(),
                history_key,
                history_field,
                value,
                "1" if inert else "0",
                *fields,
            )
        except ResponseError as exc:
            raise _refused(exc) from exc

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        """The live facts, or ``None`` when nothing is stored.

        Refuses to start on a resident key of the wrong type, such as the single-value
        state an earlier build wrote, naming the key to delete.
        """
        keys = (
            resident_cs_key(),
            resident_retired_key(),
            resident_replica_history_key(),
            resident_family_history_key(),
        )
        try:
            await self._rds.asyncio.eval(_LOAD_CHECK, len(keys), *keys)
        except ResponseError as exc:
            raise ResidentStateError(
                f"resident state is not in the current format ({exc}); delete the "
                "named key and restart"
            ) from exc
        fields = await self._rds.asyncio.hash_getall(resident_cs_key())
        return decode_live(fields) if fields else None

    async def stored_replicas_async(self) -> list[StoredReplica]:
        stored = await self._rds.asyncio.hash_getall(resident_replica_history_key())
        return [StoredReplica.model_validate_json(value) for value in stored.values()]

    async def stored_families_async(self) -> list[ServiceFamily]:
        stored = await self._rds.asyncio.hash_getall(resident_family_history_key())
        return [ServiceFamily.model_validate_json(value) for value in stored.values()]

    def save_serve_snapshot(self, snapshot: ServeSnapshot) -> None:
        self._rds.sync.set_value(serve_cs_key(), snapshot.model_dump_json())

    def load_serve_snapshot(self) -> ServeSnapshot | None:
        blob = self._rds.sync.get(serve_cs_key())
        return ServeSnapshot.model_validate_json(blob) if blob else None
