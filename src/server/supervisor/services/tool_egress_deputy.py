"""Supervisor-side external-tool egress sidecar registry.

On a node selected as an egress target this binds and drops the per-target external-tool
sidecar. The control plane pokes it over the node-command seam only to bind or unbind a
target; the operation itself never crosses that seam — it reaches the bound sidecar over
the network plane (a direct dial, or a reverse-rendezvous frame the node's ``xt:*``
attachment bridges to the sidecar over loopback). The sidecar's provider — and its
deployment-global credential — is built from this process's local configuration and
never travels on the wire.
"""

import logging
from typing import Any

from ...config import WebSearchConfig
from ...services.external_tool_sidecar import (
    ExternalToolSidecarListener,
    ExternalToolSidecarServer,
)
from ...services.search_providers import LazySearchProvider, SearchProvider
from ...services.tool_egress import ExternalToolSidecar

_DEFAULT_INTERFACES = ("search/v1",)


class ToolEgressDeputyService:
    """Binds and drops a node's fence-gated external-tool egress sidecars."""

    def __init__(
        self,
        *,
        web_search_config: WebSearchConfig,
        provider: SearchProvider | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._cfg = web_search_config
        # Build the provider lazily so a node bound as an egress target constructs it —
        # and reads its deployment-global credential — only when it actually egresses,
        # never eagerly at startup: a keyed provider whose key is only on the server
        # must not crash a worker supervisor, nor force the key onto every node.
        self._provider = provider or LazySearchProvider(web_search_config)
        self._logger = logger
        self._sidecars: dict[str, ExternalToolSidecarListener] = {}

    async def bind_sidecar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind (or rebind) the fence-gated sidecar for a control-issued target."""
        target_id = str(payload["target_id"])
        interfaces = payload.get("interfaces") or _DEFAULT_INTERFACES
        server = ExternalToolSidecarServer(
            sidecar=ExternalToolSidecar(self._provider, self._logger),
            target_id=target_id,
            target_generation=int(payload["target_generation"]),
            provider=self._cfg.provider,
            interfaces=frozenset(str(i) for i in interfaces),
            policy_class=str(payload.get("policy_class", "default")),
            logger=self._logger,
        )
        listener = ExternalToolSidecarListener(server, route=str(payload["route"]))
        await self._drop(target_id)
        host, port = await listener.start()
        self._sidecars[target_id] = listener
        if self._logger is not None:
            self._logger.info(
                "tool sidecar bound target=%s provider=%s route=%s:%s",
                target_id,
                self._cfg.provider,
                host,
                port,
            )
        return {"bound": True, "host": host, "port": port}

    async def unbind_sidecar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop a target's sidecar; absent is a no-op."""
        target_id = str(payload["target_id"])
        await self._drop(target_id)
        if self._logger is not None:
            self._logger.info("tool sidecar unbound target=%s", target_id)
        return {"unbound": True}

    async def _drop(self, target_id: str) -> None:
        listener = self._sidecars.pop(target_id, None)
        if listener is not None:
            await listener.stop()

    async def stop(self) -> None:
        """Drop every bound sidecar on shutdown."""
        for target_id in list(self._sidecars):
            await self._drop(target_id)


__all__ = ["ToolEgressDeputyService"]
