"""Assertion holds: an identity a binding is about to name, held off condemnation
until the commit naming it is confirmed.

Several assertions may hold one identity at once, each under its own nonce, and each
releases only its own hold. An acquisition validates every identity of a chunk before
it changes anything, so a refused chunk leaves no partial hold behind, and once one
chunk refuses, the chunks after it are only read, so one refusal names every identity
that refuses. Retirement is read before protection: protection does not stop a delete
already issued, and an identity protected only as a pending upload or a tree member
names no stored bytes, so an identity retired before is verified, or reported, even
while it is protected.
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
    RECLAIM_SCOPES_KEY,
    RedisClient,
    reclaim_condemned_key,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
)

# How many identities one script decides at a time, which bounds how long it holds
# Redis: about 1 ms at this size.
CHUNK = 256

# KEYS: protected, pins, deleting, history, condemned, unexpanded.
# ARGV: nonce, locator, mode, probe, n, d1..dn, e1..en, m, manifest1..manifestm.
_ACQUIRE = """
for _, key in ipairs(KEYS) do
  local kind = redis.call('TYPE', key).ok
  if kind ~= 'none' and kind ~= 'set' and kind ~= 'hash' then
    return redis.error_reply('WRONGTYPE ' .. key)
  end
end
local nonce, locator, mode = ARGV[1], ARGV[2], ARGV[3]
local probe, n = ARGV[4] == '1', tonumber(ARGV[5])
local out, refused = {}, false
for i = 1, n do
  local d, expected = ARGV[5 + i], ARGV[5 + n + i]
  local verdict = 'A'
  local rev = mode ~= 'pending' and redis.call('HGET', KEYS[4], d)
  if mode ~= 'pending' and redis.call('HEXISTS', KEYS[3], d) == 1 then
    verdict, refused = 'R', true
  elseif mode == 'bind' and rev and rev ~= expected then
    verdict, refused = 'N:' .. rev, true
  elseif mode == 'member' and rev then
    verdict = 'H'
  elseif redis.call('SISMEMBER', KEYS[1], d) == 1 then
    verdict = 'P'
  end
  out[i] = verdict
end
if refused or probe then
  return out
end
for i = 1, n do
  if out[i] ~= 'P' then
    local d = ARGV[5 + i]
    redis.call('HDEL', KEYS[5], d)
    local current = redis.call('HGET', KEYS[2], d)
    local holders = current and cjson.decode(current) or {}
    table.insert(holders, {n = nonce, l = locator})
    redis.call('HSET', KEYS[2], d, cjson.encode(holders))
  end
end
local m = tonumber(ARGV[6 + 2 * n])
for j = 1, m do
  redis.call('SADD', KEYS[6], ARGV[6 + 2 * n + j])
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
    MEMBER = "member"
    """A bound tree's member, whose bytes may still be uploading: an identity being
    retired refuses, and one retired before is held and reported, for the upload's
    durability check to verify."""


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
    retired: tuple[str, ...] = ()
    """Held identities retired before, which a member acquisition reports."""


class Retirement(StrEnum):
    """Where an identity stands in its retirement, as read now."""

    NONE = ""
    RETIRING = "retiring"
    RETIRED = "retired"
    """Retired before and not being retired now; its current backing is unverified."""


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


def _retirement(
    digests: Sequence[str], deleting: list[Any], history: list[Any]
) -> dict[str, Retirement]:
    return {
        digest: (
            Retirement.RETIRING
            if retiring is not None
            else Retirement.RETIRED if retired is not None else Retirement.NONE
        )
        for digest, retiring, retired in zip(digests, deleting, history, strict=True)
    }


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
        acquisition holds nothing: the chunks held before the first refusal are
        released, the chunks after it are read without holding, and it reports every
        identity of every chunk that refuses. An exception may leave earlier chunks
        held under ``nonce``, which releasing ``nonce`` on ``digests`` ends.
        """
        ordered = list(dict.fromkeys(digests))
        verified = verified or {}
        encoded = json.dumps(locator, sort_keys=True, separators=(",", ":"))
        keys = _keys(scope)
        held: list[str] = []
        retired: list[str] = []
        retiring: list[str] = []
        unverified: dict[str, str] = {}
        chunks = list(batched(ordered, CHUNK)) or [()]
        for index, chunk in enumerate(chunks):
            probe = bool(retiring or unverified)
            extra = manifests if index == len(chunks) - 1 and not probe else ()
            args = [
                nonce,
                encoded,
                mode.value,
                "1" if probe else "0",
                str(len(chunk)),
                *chunk,
                *(verified.get(d, "") for d in chunk),
                str(len(extra)),
                *extra,
            ]
            verdicts = [
                str(v) for v in self._rds.sync.eval(_ACQUIRE, len(keys), *keys, *args)
            ]
            for digest, verdict in zip(chunk, verdicts, strict=True):
                if verdict == "R":
                    retiring.append(digest)
                elif verdict.startswith("N:"):
                    unverified[digest] = verdict[2:]
            if not (probe or retiring or unverified):
                held.extend(d for d, v in zip(chunk, verdicts, strict=True) if v != "P")
                retired.extend(
                    d for d, v in zip(chunk, verdicts, strict=True) if v == "H"
                )
        if retiring or unverified:
            self.release(scope, held, nonce)
            return Acquisition(
                Verdict.RETIRING if retiring else Verdict.NEEDS_VERIFY,
                nonce,
                retiring=tuple(retiring),
                unverified=unverified,
            )
        return Acquisition(
            Verdict.ACQUIRED, nonce, held=tuple(held), retired=tuple(retired)
        )

    def retirement(self, scope: str, digests: Sequence[str]) -> dict[str, Retirement]:
        """Return where each of ``digests`` stands in its retirement now."""
        standing: dict[str, Retirement] = {}
        for chunk in batched(digests, CHUNK):
            with self._rds.sync.control_pipeline(transaction=False) as pipe:
                pipe.hmget(reclaim_deleting_key(scope), list(chunk))
                pipe.hmget(reclaim_history_key(scope), list(chunk))
                standing.update(_retirement(chunk, *pipe.execute()))
        return standing

    async def retirement_async(
        self, scope: str, digests: Sequence[str]
    ) -> dict[str, Retirement]:
        standing: dict[str, Retirement] = {}
        for chunk in batched(digests, CHUNK):
            async with self._rds.asyncio.control_pipeline(transaction=False) as pipe:
                pipe.hmget(reclaim_deleting_key(scope), list(chunk))
                pipe.hmget(reclaim_history_key(scope), list(chunk))
                standing.update(_retirement(chunk, *await pipe.execute()))
        return standing

    def release_older(self, epoch: int, page: int = CHUNK) -> int:
        """Release every hold an acquisition of an earlier root epoch took; returns
        how many.

        A root takes its epoch before anything else, which fences every commit an
        earlier root could still issue, so what such a hold covered is either bound,
        and protected with its binding, or never will be, and its publication is
        reported again under this root.
        """
        released = 0
        for scope in sorted(self._rds.sync.set_members(RECLAIM_SCOPES_KEY)):
            key = reclaim_pins_key(scope)
            cursor = 0
            while True:
                with self._rds.sync.control_pipeline(transaction=False) as pipe:
                    pipe.hscan(key, cursor, count=page)
                    cursor, fields = pipe.execute()[0]
                stale: dict[str, list[str]] = {}
                for digest, raw in dict(fields).items():
                    for holder in json.loads(raw):
                        if _epoch_of(str(holder.get("n", ""))) < epoch:
                            stale.setdefault(holder["n"], []).append(digest)
                for nonce, digests in stale.items():
                    released += self.release(scope, digests, nonce)
                if not int(cursor):
                    break
        return released

    def release(self, scope: str, digests: Sequence[str], nonce: str) -> int:
        """Release exactly ``nonce``'s holds on ``digests``; returns how many."""
        released = 0
        for chunk in batched(digests, CHUNK):
            released += int(
                self._rds.sync.eval(_RELEASE, 1, reclaim_pins_key(scope), nonce, *chunk)
            )
        return released


def _epoch_of(nonce: str) -> int:
    head, _, _ = nonce.partition(":")
    return int(head) if head.isdigit() else 0


__all__ = [
    "AcquireMode",
    "Acquisition",
    "Nonces",
    "PinStore",
    "Retirement",
    "Verdict",
    "CHUNK",
]
