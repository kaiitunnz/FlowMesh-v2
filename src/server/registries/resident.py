"""Durable persistence for the authoritative resident-capacity control facts.

The authoritative ``CS`` stores are fabric-scoped, so they persist under one snapshot
key
rather than per-workflow. Only authoritative facts are saved; derived views and capacity
telemetry are recomputed after a restart. Persistence follows the durable-ledger
pattern:
the snapshot is written after each authoritative mutation and rehydrated on startup.
"""

from ..clients.redis import RedisClient, ingress_cs_key, resident_cs_key
from ..ingress.state import IngressSnapshot
from ..resident.state import ResidentSnapshot


class ResidentRegistry:
    """Redis-backed snapshot store for resident-capacity control facts."""

    def __init__(self, rds: RedisClient) -> None:
        self._rds = rds

    def save_snapshot(self, snapshot: ResidentSnapshot) -> None:
        self._rds.sync.set_value(resident_cs_key(), snapshot.model_dump_json())

    async def save_snapshot_async(self, snapshot: ResidentSnapshot) -> None:
        await self._rds.asyncio.set_value(resident_cs_key(), snapshot.model_dump_json())

    def load_snapshot(self) -> ResidentSnapshot | None:
        blob = self._rds.sync.get(resident_cs_key())
        return ResidentSnapshot.model_validate_json(blob) if blob else None

    async def load_snapshot_async(self) -> ResidentSnapshot | None:
        blob = await self._rds.asyncio.get(resident_cs_key())
        return ResidentSnapshot.model_validate_json(blob) if blob else None

    def save_ingress_snapshot(self, snapshot: IngressSnapshot) -> None:
        self._rds.sync.set_value(ingress_cs_key(), snapshot.model_dump_json())

    def load_ingress_snapshot(self) -> IngressSnapshot | None:
        blob = self._rds.sync.get(ingress_cs_key())
        return IngressSnapshot.model_validate_json(blob) if blob else None
