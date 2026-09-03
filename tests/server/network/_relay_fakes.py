"""Shared fakes for the reverse-relay substrate tests.

A deterministic in-memory binary-Redis stub models exactly the surface the substrate
uses — stream append / cursor read / MINID trim, hash CRUD, and ``SET NX PX`` with a
test-driven clock — so the relay and its recovery path are proven without a live Redis.
"""

from typing import Any

from server.network.reverse_relay import RelayDirection, RelayFrame, RelayFrameKind


class FakeBinaryRedis:
    """Models the binary-safe stream/hash/string surface with a controllable clock."""

    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict[bytes, bytes]]]] = {}
        self._seq = 0
        self._hashes: dict[str, dict[bytes, bytes]] = {}
        self._strings: dict[str, tuple[str, float]] = {}  # value, expires_at
        self.now = 0.0
        self.blocks: list[int | None] = (
            []
        )  # the block arg of each xread, for assertions

    async def xadd(self, name: str, fields: dict[bytes, bytes]) -> bytes:
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self._streams.setdefault(name, []).append((entry_id, dict(fields)))
        return entry_id.encode()

    async def xread(
        self, streams: dict[str, str], count: int, block: int | None
    ) -> list[Any]:
        self.blocks.append(block)
        out: list[Any] = []
        for key, after in streams.items():
            after_n = _idnum(after)
            items = [
                (eid.encode(), fields)
                for eid, fields in self._streams.get(key, [])
                if _idnum(eid) > after_n
            ][:count]
            if items:
                out.append((key.encode(), items))
        return out

    async def xtrim(self, name: str, minid: str, approximate: bool) -> int:
        keep_from = _idnum(minid)
        before = self._streams.get(name, [])
        self._streams[name] = [e for e in before if _idnum(e[0]) >= keep_from]
        return len(before) - len(self._streams[name])

    async def hset(self, name: str, mapping: dict[str, str]) -> int:
        h = self._hashes.setdefault(name, {})
        for k, v in mapping.items():
            h[k.encode()] = v.encode()
        return len(mapping)

    async def hgetall(self, name: str) -> dict[bytes, bytes]:
        return dict(self._hashes.get(name, {}))

    async def pexpire(self, name: str, ms: int) -> int:
        # The substrate only sets a leak-backstop TTL; presence is what the tests check.
        return 1 if name in self._hashes or name in self._strings else 0

    async def set(self, name: str, value: str, nx: bool, px: int) -> bool | None:
        live = self._live(name)
        if nx and live is not None:
            return None
        self._strings[name] = (value, self.now + px / 1000.0)
        return True

    async def get(self, name: str) -> bytes | None:
        live = self._live(name)
        return live.encode() if live is not None else None

    async def delete(self, name: str) -> int:
        # Redis DEL removes a key of any type; model strings, hashes, and streams alike.
        hit = (
            self._strings.pop(name, None) is not None
            or self._hashes.pop(name, None) is not None
            or self._streams.pop(name, None) is not None
        )
        return 1 if hit else 0

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> int:
        # Models the two owner-fenced lease scripts: compare the live value to the
        # owner, then pexpire (refresh) or del (release) only on a match.
        key, owner = keys_and_args[0], keys_and_args[1]
        if self._live(key) != owner:
            return 0
        if "pexpire" in script:
            self._strings[key] = (owner, self.now + int(keys_and_args[2]) / 1000.0)
            return 1
        if "del" in script:
            self._strings.pop(key, None)
            return 1
        return 0

    def _live(self, name: str) -> str | None:
        entry = self._strings.get(name)
        if entry is None:
            return None
        value, expires_at = entry
        if self.now >= expires_at:
            self._strings.pop(name, None)
            return None
        return value


def _idnum(entry_id: str) -> int:
    return int(entry_id.split("-")[0]) if entry_id not in ("0", "") else 0


def relay_frame(
    kind: RelayFrameKind,
    *,
    session_id: str = "rly-1",
    direction: RelayDirection = RelayDirection.TARGET_TO_ORIGIN,
    seq: int = 0,
    ack: int = 0,
    payload: bytes = b"",
) -> RelayFrame:
    return RelayFrame(
        kind=kind,
        session_id=session_id,
        invocation_id="inv-1",
        idm="idm-1",
        direction=direction,
        seq=seq,
        ack=ack,
        payload=payload,
    )
