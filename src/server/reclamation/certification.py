"""Whether the protection index of a scope is known to cover every binding naming it.

Protection is written at every commit seam by every root holding the hooks, so the
index misses only bindings committed before the first such root started. The hooks-on
marker holds the generation of the backfill that protects those: the epoch of a root
that started after the hooks went live, so every binding the backfill does not read was
committed under the hooks. Every later root keeps the hooks live, so the marker and the
backfill completed under it stand across restarts; a store that lost the marker, or a
backfill that found an undecodable key, starts a new generation that backfills again.
A scope is certified for the marker's generation once every key family's scan
completed in it, no key naming the scope failed to decode in it, no bound tree awaits
expansion, and no commit owes the scope a protection it could not confirm. Anything
less leaves the scope uncertified, and an uncertified scope is never collected.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..clients.redis import (
    RECLAIM_BACKFILL_FAULTS_KEY,
    RECLAIM_BACKFILL_KEY,
    RECLAIM_HOOKS_ON_KEY,
    RECLAIM_ROOT_EPOCH_KEY,
    RedisClient,
    reclaim_cert_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)

# The fault field naming every scope, for a key whose scope could not be read.
ALL_SCOPES = "*"

# KEYS: hooks-on, faults. ARGV: epoch. Keeps the marker's generation unless its
# backfill found an undecodable key, and then raises it to ``epoch``; never lowers it.
_MARK = """
local current = redis.call('GET', KEYS[1])
if current then
  if tonumber(current) >= tonumber(ARGV[1]) then
    return current
  end
  local faulted = false
  if redis.call('TYPE', KEYS[2]).ok == 'hash' then
    for _, generation in ipairs(redis.call('HVALS', KEYS[2])) do
      if generation == current then
        faulted = true
        break
      end
    end
  end
  if not faulted then
    return current
  end
end
redis.call('SET', KEYS[1], ARGV[1])
return ARGV[1]
"""

# KEYS: hooks-on, backfill. ARGV: generation, family.
_COMPLETE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[1])
return 1
"""

# KEYS: hooks-on, backfill, faults, unexpanded, uncertified, cert.
# ARGV: scope, family1..familyn.
_CERTIFY = """
local function kind(key, ...)
  local found = redis.call('TYPE', key).ok
  if found == 'none' then return true end
  for _, allowed in ipairs({...}) do
    if found == allowed then return true end
  end
  return false
end
if not kind(KEYS[2], 'hash') or not kind(KEYS[3], 'hash')
  or not kind(KEYS[4], 'set') then
  redis.call('DEL', KEYS[6])
  return {'wrongtype'}
end
local generation = redis.call('GET', KEYS[1])
if not generation then
  redis.call('DEL', KEYS[6])
  return {'no_marker'}
end
for i = 2, #ARGV do
  if redis.call('HGET', KEYS[2], ARGV[i]) ~= generation then
    redis.call('DEL', KEYS[6])
    return {'incomplete', ARGV[i]}
  end
end
if redis.call('HGET', KEYS[3], '*') == generation
  or redis.call('HGET', KEYS[3], ARGV[1]) == generation then
  redis.call('DEL', KEYS[6])
  return {'undecodable'}
end
local unexpanded = redis.call('SCARD', KEYS[4])
if unexpanded > 0 then
  redis.call('DEL', KEYS[6])
  return {'unexpanded', tostring(unexpanded)}
end
if redis.call('EXISTS', KEYS[5]) == 1 then
  redis.call('DEL', KEYS[6])
  return {'owed'}
end
redis.call('SET', KEYS[6], generation)
return {'certified', generation}
"""


class CertificationStatus(StrEnum):
    CERTIFIED = "certified"
    NO_MARKER = "no_marker"
    """No root with the hooks has marked when they went live."""
    INCOMPLETE = "incomplete"
    """A key family's scan has not completed in the current generation."""
    UNDECODABLE = "undecodable"
    """A key naming the scope, or one whose scope is unknown, did not decode."""
    UNEXPANDED = "unexpanded"
    """A bound tree's members are not yet confirmed."""
    OWED = "owed"
    """A commit owes the scope a protection it could not confirm."""
    WRONGTYPE = "wrongtype"
    """A coverage key holds a value of the wrong type."""


@dataclass(frozen=True)
class Certification:
    """Whether a scope is certified, and the reason when it is not."""

    status: CertificationStatus
    detail: str = ""

    @property
    def certified(self) -> bool:
        return self.status is CertificationStatus.CERTIFIED


class CoverageStore:
    """The hooks-on marker, backfill completion and scope certification."""

    def __init__(self, rds: RedisClient, families: Sequence[str]) -> None:
        self._rds = rds
        self._families = tuple(families)

    def mark_hooks_on(self, epoch: int) -> int:
        """Record that the protection hooks are live as of root ``epoch``; returns the
        backfill generation the marker then holds.

        A marker already standing keeps its generation, and with it the backfill
        completed under it, unless that backfill found an undecodable key: then
        ``epoch`` starts a new generation, so the store is read again. With no marker,
        ``epoch`` starts the first.
        """
        return int(
            self._rds.sync.eval(
                _MARK,
                2,
                RECLAIM_HOOKS_ON_KEY,
                RECLAIM_BACKFILL_FAULTS_KEY,
                str(epoch),
            )
        )

    def superseded(self, epoch: int) -> bool:
        """Whether a root later than ``epoch`` has started."""
        current = self._rds.sync.get(RECLAIM_ROOT_EPOCH_KEY)
        return current is not None and int(current) > epoch

    def generation(self) -> int | None:
        """The generation the marker holds, or None before any root marked it."""
        value = self._rds.sync.get(RECLAIM_HOOKS_ON_KEY)
        return int(value) if value else None

    def completed(self, generation: int) -> frozenset[str]:
        """The families whose scan completed in ``generation``."""
        stored = self._rds.sync.hash_getall(RECLAIM_BACKFILL_KEY)
        return frozenset(
            family for family, value in stored.items() if value == str(generation)
        )

    def complete(self, generation: int, family: str) -> bool:
        """Record ``family``'s scan as complete in ``generation``; False, recording
        nothing, when a later root has moved the marker past it."""
        return bool(
            int(
                self._rds.sync.eval(
                    _COMPLETE,
                    2,
                    RECLAIM_HOOKS_ON_KEY,
                    RECLAIM_BACKFILL_KEY,
                    str(generation),
                    family,
                )
            )
        )

    def record_fault(self, generation: int, scope: str | None) -> None:
        """Record that a key naming ``scope``, or an unknown scope, did not decode in
        ``generation``."""
        self._rds.sync.hash_set(
            RECLAIM_BACKFILL_FAULTS_KEY, {scope or ALL_SCOPES: str(generation)}
        )

    def certify(self, scope: str) -> Certification:
        """Re-check every condition and certify ``scope`` for the current generation
        when all hold, withdrawing its certification when any does not."""
        keys = [
            RECLAIM_HOOKS_ON_KEY,
            RECLAIM_BACKFILL_KEY,
            RECLAIM_BACKFILL_FAULTS_KEY,
            reclaim_unexpanded_key(scope),
            reclaim_uncertified_key(scope),
            reclaim_cert_key(scope),
        ]
        reply = list(
            self._rds.sync.eval(_CERTIFY, len(keys), *keys, scope, *self._families)
        )
        status = CertificationStatus(reply[0])
        return Certification(status, str(reply[1]) if len(reply) > 1 else "")


__all__ = [
    "ALL_SCOPES",
    "Certification",
    "CertificationStatus",
    "CoverageStore",
]
