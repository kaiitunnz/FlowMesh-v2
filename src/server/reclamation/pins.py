"""Assertion holds: an identity a binding is about to name, held off condemnation
until the commit naming it is confirmed.

Several assertions may hold one identity at once, each under its own nonce, and each
releases only its own hold. An acquisition validates every identity of a chunk before
it changes anything, so a refused chunk leaves no partial hold behind; an identity
being retired refuses before the protected fast path, since protection does not stop a
delete already issued.
"""

import itertools
import json
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import batched
from typing import Any

from ..clients.redis import (
    RedisClient,
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
)

# How many identities one acquisition script decides at a time.
CHUNK = 1000

# KEYS: protected, pins, deleting, history, condemned, unexpanded.
# ARGV: nonce, locator, mode, n, d1..dn, e1..en, m, manifest1..manifestm.
_ACQUIRE = """
for _, key in ipairs(KEYS) do
  local kind = redis.call('TYPE', key).ok
  if kind ~= 'none' and kind ~= 'set' and kind ~= 'hash' then
    return redis.error_reply('WRONGTYPE ' .. key)
  end
end
local nonce, locator, mode, n = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
local out, refused = {}, false
for i = 1, n do
  local d, expected = ARGV[4 + i], ARGV[4 + n + i]
  local verdict = 'A'
  if mode == 'bind' and redis.call('HEXISTS', KEYS[3], d) == 1 then
    verdict, refused = 'R', true
  elseif redis.call('SISMEMBER', KEYS[1], d) == 1 then
    verdict = 'P'
  elseif mode == 'bind' then
    local rev = redis.call('HGET', KEYS[4], d)
    if rev and rev ~= expected then
      verdict, refused = 'N:' .. rev, true
    end
  end
  out[i] = verdict
end
if refused then
  return out
end
for i = 1, n do
  if out[i] ~= 'P' then
    local d = ARGV[4 + i]
    redis.call('HDEL', KEYS[5], d)
    local current = redis.call('HGET', KEYS[2], d)
    local holders = current and cjson.decode(current) or {}
    table.insert(holders, {n = nonce, l = locator})
    redis.call('HSET', KEYS[2], d, cjson.encode(holders))
  end
end
local m = tonumber(ARGV[5 + 2 * n])
for j = 1, m do
  redis.call('SADD', KEYS[6], ARGV[5 + 2 * n + j])
end
return out
"""

# KEYS: pins. ARGV: nonce, d1..dn.
_RELEASE = """
if redis.call('TYPE', KEYS[1]).ok == 'none' then
  return 0
end
local released = 0
for i = 2, #ARGV do
  local current = redis.call('HGET', KEYS[1], ARGV[i])
  if current then
    local kept = {}
    for _, holder in ipairs(cjson.decode(current)) do
      if holder.n == ARGV[1] then
        released = released + 1
      else
        table.insert(kept, holder)
      end
    end
    if #kept == 0 then
      redis.call('HDEL', KEYS[1], ARGV[i])
    else
      redis.call('HSET', KEYS[1], ARGV[i], cjson.encode(kept))
    end
  end
end
return released
"""


class AcquireMode(StrEnum):
    """How an acquisition treats an identity's retirement state."""

    BIND = "bind"
    """The binding asserts the bytes are stored now: an identity being retired, or one
    retired before whose current backing is unverified, refuses."""
    PENDING = "pending"
    """The binding accepts bytes still to be uploaded: retirement does not refuse it,
    and the upload's durability check settles the backing later."""


class Verdict(StrEnum):
    ACQUIRED = "acquired"
    RETIRING = "retiring"
    """An identity has a retirement in progress; the writer waits for it."""
    NEEDS_VERIFY = "needs_verify"
    """An identity was retired before; its current backing must be verified."""


@dataclass(frozen=True)
class Acquisition:
    """The outcome of holding a set of identities for one assertion."""

    verdict: Verdict
    nonce: str
    held: tuple[str, ...] = ()
    """The identities this acquisition holds, which its release names."""
    retiring: tuple[str, ...] = ()
    unverified: dict[str, str] = field(default_factory=dict)
    """Each identity whose backing needs verifying, with its retirement revision."""


class Nonces:
    """Acquisition nonces unique to one root epoch."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return f"{self.epoch}:{next(self._counter)}"


def _keys(scope: str) -> list[str]:
    return [
        reclaim_protected_key(scope),
        reclaim_pins_key(scope),
        reclaim_deleting_key(scope),
        reclaim_history_key(scope),
        reclaim_condemned_key(scope),
        reclaim_unexpanded_key(scope),
    ]


class PinStore:
    """Acquires and releases assertion holds in the control store."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def acquire(
        self,
        scope: str,
        digests: Iterable[str],
        nonce: str,
        locator: dict[str, Any],
        mode: AcquireMode = AcquireMode.BIND,
        verified: dict[str, str] | None = None,
        manifests: Sequence[str] = (),
    ) -> Acquisition:
        """Hold ``digests`` under ``nonce`` for the binding ``locator`` names.

        ``verified`` gives the retirement revision an identity's backing was verified
        at; ``manifests`` enter the scope's unexpanded set with the hold. A refused
        chunk changes nothing, and the chunks this call held before it are released.
        """
        ordered = list(dict.fromkeys(digests))
        verified = verified or {}
        encoded = json.dumps(locator, sort_keys=True, separators=(",", ":"))
        held: list[str] = []
        chunks = list(batched(ordered, CHUNK)) or [()]
        for index, chunk in enumerate(chunks):
            extra = manifests if index == len(chunks) - 1 else ()
            args = [
                nonce,
                encoded,
                mode.value,
                str(len(chunk)),
                *chunk,
                *(verified.get(d, "") for d in chunk),
                str(len(extra)),
                *extra,
            ]
            keys = _keys(scope)
            verdicts = list(self._rds.sync.eval(_ACQUIRE, len(keys), *keys, *args))
            retiring = [d for d, v in zip(chunk, verdicts, strict=True) if v == "R"]
            unverified = {
                d: str(v)[2:]
                for d, v in zip(chunk, verdicts, strict=True)
                if str(v).startswith("N:")
            }
            if retiring or unverified:
                self.release(scope, held, nonce)
                return Acquisition(
                    Verdict.RETIRING if retiring else Verdict.NEEDS_VERIFY,
                    nonce,
                    retiring=tuple(retiring),
                    unverified=unverified,
                )
            held.extend(d for d, v in zip(chunk, verdicts, strict=True) if v != "P")
        return Acquisition(Verdict.ACQUIRED, nonce, held=tuple(held))

    def release(self, scope: str, digests: Sequence[str], nonce: str) -> int:
        """Release exactly ``nonce``'s holds on ``digests``; returns how many."""
        released = 0
        for chunk in batched(digests, CHUNK):
            released += int(
                self._rds.sync.eval(_RELEASE, 1, reclaim_pins_key(scope), nonce, *chunk)
            )
        return released


__all__ = ["AcquireMode", "Acquisition", "Nonces", "PinStore", "Verdict", "CHUNK"]
