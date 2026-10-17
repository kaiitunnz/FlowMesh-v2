"""Which content an outcome finalization already produced.

A producer that may be re-driven must be able to tell whether its outcome was already
materialized, or a second drive re-runs a sampled model and settles a different result
under the same identity. That question is control-plane state: the index here maps a
finalization's ``idm-*`` to the reference it produced, and holds nothing else — never
the bytes, which live in the shared content store, and never a general name for content,
which an idempotency key is not.

Recording a finalization is a binding, so it protects the content it binds, and only
the winner's: a re-drive that loses is handed the first binding and protects nothing.
"""

import itertools
import json
from collections.abc import Callable
from enum import StrEnum

from shared.content import ContentReference
from shared.outcome import OutcomeManifest

from ..clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    RECLAIM_SCOPES_KEY,
    RedisClient,
    content_finalization_key,
    content_finalization_scope_key,
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
)

# KEYS: finalization, epoch, protected, pins, deleting, history, condemned, scopes.
# ARGV: epoch, manifest, digest, nonce, locator, verified revision, ttl, scope.
_RECORD = """
for i = 3, 8 do
  local kind = redis.call('TYPE', KEYS[i]).ok
  if kind ~= 'none' and kind ~= 'set' and kind ~= 'hash' then
    return redis.error_reply('WRONGTYPE ' .. KEYS[i])
  end
end
if ARGV[1] ~= '' and redis.call('GET', KEYS[2]) ~= ARGV[1] then
  return {'fenced'}
end
local existing = redis.call('GET', KEYS[1])
if existing then
  return {'winner', existing}
end
local digest = ARGV[3]
if redis.call('HEXISTS', KEYS[5], digest) == 1 then
  return {'retiring'}
end
if redis.call('SISMEMBER', KEYS[3], digest) == 0 then
  local revision = redis.call('HGET', KEYS[6], digest)
  if revision and revision ~= ARGV[6] then
    return {'verify', revision}
  end
end
local current = redis.call('HGET', KEYS[4], digest)
local holders = current and cjson.decode(current) or {}
table.insert(holders, {n = ARGV[4], l = ARGV[5]})
redis.call('HSET', KEYS[4], digest, cjson.encode(holders))
redis.call('HDEL', KEYS[7], digest)
redis.call('SET', KEYS[1], ARGV[2])
if tonumber(ARGV[7]) > 0 then
  redis.call('EXPIRE', KEYS[1], tonumber(ARGV[7]))
end
redis.call('SADD', KEYS[3], digest)
redis.call('SADD', KEYS[8], ARGV[8])
local kept = {}
for _, holder in ipairs(holders) do
  if holder.n ~= ARGV[4] then
    table.insert(kept, holder)
  end
end
if #kept == 0 then
  redis.call('HDEL', KEYS[4], digest)
else
  redis.call('HSET', KEYS[4], digest, cjson.encode(kept))
end
return {'committed', ARGV[2]}
"""

# How many times a recording re-checks a backing it verified before it gives up for now.
_VERIFY_ROUNDS = 3


class FinalizationDisposition(StrEnum):
    """Why a finalization was not recorded, and what its producer does next."""

    RETRY_RETIREMENT = "retry_retirement"
    """The content is being retired; the producer waits and records again."""
    REMATERIALIZE = "rematerialize"
    """The content is gone; the producer writes the same bytes again and records."""
    RETRY = "retry"
    """The backing could not be verified for now; the producer records again."""


class FinalizationRefused(Exception):
    """A finalization the index did not record, with what its producer does next."""

    def __init__(self, disposition: FinalizationDisposition) -> None:
        super().__init__(f"finalization not recorded: {disposition.value}")
        self.disposition = disposition


class StaleRootEpoch(RuntimeError):
    """This root no longer holds the root epoch and records nothing."""


class FinalizationIndex:
    """The reference each outcome finalization committed, by scope and key.

    ``epoch`` returns the root epoch this root records under, or None before it holds
    one; ``verify`` reads whether a reference's content is currently stored: True,
    False, or None when the store cannot say.
    """

    def __init__(
        self,
        redis: RedisClient,
        *,
        ttl_sec: float = 0.0,
        epoch: Callable[[], int | None] | None = None,
        verify: Callable[[ContentReference], bool | None] | None = None,
    ) -> None:
        self._rds = redis
        self._ttl = ttl_sec
        self.epoch = epoch
        self.verify = verify
        self._nonces = itertools.count(1)

    def assign_scope(self, idempotency_key: str, scope: str) -> None:
        """Record the scope control assigned the work this key settles.

        Control decides the scope when it authorizes the work — a dispatch's own, the
        one a permit carries, the one an admitted invocation materializes under — and
        records it here so the binding that key later reports is checked against what
        was assigned rather than against what the reporter says. Only control writes
        this; a producer never reaches it.
        """
        if not idempotency_key or not scope:
            return
        key = content_finalization_scope_key(idempotency_key)
        self._rds.sync.set_value(key, scope)
        if self._ttl:
            self._rds.sync.expire(key, int(self._ttl))

    def assigned_scope(self, idempotency_key: str) -> str | None:
        """The scope assigned to this key, or None if control assigned none."""
        return self._rds.sync.get(content_finalization_scope_key(idempotency_key))

    def find(self, scope: str, idempotency_key: str) -> OutcomeManifest | None:
        raw = self._rds.sync.get(content_finalization_key(scope, idempotency_key))
        return OutcomeManifest.model_validate_json(raw) if raw else None

    def record(
        self,
        scope: str,
        idempotency_key: str,
        content: ContentReference,
        *,
        provenance: str | None = None,
    ) -> OutcomeManifest:
        """Bind a finalization to its content, or return the binding already there.

        The first binding stands: a re-drive that produced the same content records
        nothing new, and one that produced different content still settles as the
        first, so an outcome never changes after it has been committed. One script
        decides the winner, checks the content is not being retired, and protects it,
        so two drives racing never both believe they won. Raises
        ``FinalizationRefused`` when the content cannot be bound now.
        """
        manifest = OutcomeManifest(
            content=content, provenance=provenance, idempotency_key=idempotency_key
        ).model_dump_json()
        epoch = self.epoch() if self.epoch is not None else None
        nonce = f"{epoch or 0}:fin:{next(self._nonces)}"
        locator = json.dumps({"kind": "finalization", "idm": idempotency_key})
        keys = [
            content_finalization_key(scope, idempotency_key),
            RECLAIM_ROOT_EPOCH_KEY,
            reclaim_protected_key(scope),
            reclaim_pins_key(scope),
            reclaim_deleting_key(scope),
            reclaim_history_key(scope),
            reclaim_condemned_key(scope),
            RECLAIM_SCOPES_KEY,
        ]
        verified = ""
        for _ in range(_VERIFY_ROUNDS):
            reply = list(
                self._rds.sync.eval(
                    _RECORD,
                    len(keys),
                    *keys,
                    "" if epoch is None else str(epoch),
                    manifest,
                    content.content_digest,
                    nonce,
                    locator,
                    verified,
                    str(int(self._ttl)),
                    scope,
                )
            )
            match reply[0]:
                case "committed" | "winner":
                    return OutcomeManifest.model_validate_json(reply[1])
                case "fenced":
                    raise StaleRootEpoch("a later root holds the root epoch")
                case "retiring":
                    raise FinalizationRefused(FinalizationDisposition.RETRY_RETIREMENT)
            present = self.verify(content) if self.verify is not None else None
            if present is False:
                raise FinalizationRefused(FinalizationDisposition.REMATERIALIZE)
            if present is None:
                raise FinalizationRefused(FinalizationDisposition.RETRY)
            verified = str(reply[1])
        raise FinalizationRefused(FinalizationDisposition.RETRY)
