"""The published, tenant-authorized service-family alias catalog.

A client selects only an alias from this deployment-published catalog; it never names a
model image, worker, endpoint, or routing policy. Each alias maps to a resident service
family and carries the tenants authorized to use it and its request profile. The catalog
is loaded from deployment configuration.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..task.v2.representations.operators import ServiceDependency, ServiceInterface


class PublishedAlias(BaseModel):
    """One published alias binding a tenant-visible name to a resident family.

    ``allowed_tenants`` is the set of tenant ids authorized to select the alias; an
    empty set authorizes any authenticated tenant. ``max_output_tokens`` sizes the
    invocation's admission profile (its credit demand); it does not clamp the engine's
    own token limit, which the opaque client request still carries. The service
    reference, interface, and isolation resolve to the same ``ServiceDependency`` a
    workflow leaf would, so an authorized ingress request reuses a warm family.
    """

    model_config = ConfigDict(frozen=True)

    alias: str
    service_ref: str
    interface: ServiceInterface = ServiceInterface.CHAT
    isolation: str | None = None
    allowed_tenants: frozenset[str] = frozenset()
    max_output_tokens: int | None = None

    def authorizes(self, tenant: str | None) -> bool:
        """Whether ``tenant`` may select this alias."""
        if not self.allowed_tenants:
            return True
        return tenant is not None and tenant in self.allowed_tenants

    def dependency(self) -> ServiceDependency:
        """The resident service dependency this alias resolves to."""
        return ServiceDependency(
            service_ref=self.service_ref,
            interface=self.interface,
            isolation=self.isolation,
        )


class AliasCatalog:
    """The set of published aliases, keyed by alias name."""

    def __init__(self, aliases: list[PublishedAlias]) -> None:
        self._aliases = {alias.alias: alias for alias in aliases}

    @classmethod
    def from_json(cls, raw: str) -> "AliasCatalog":
        """Parse the deployment alias catalog from its JSON document.

        The document is ``{"aliases": [ ... ]}`` or a bare list of alias objects. An
        empty or blank document yields an empty catalog.
        """
        text = raw.strip()
        if not text:
            return cls([])
        document: Any = json.loads(text)
        entries = (
            document.get("aliases", []) if isinstance(document, dict) else document
        )
        return cls([PublishedAlias.model_validate(entry) for entry in entries])

    def get(self, alias: str) -> PublishedAlias | None:
        """The published alias, or None when it is not published."""
        return self._aliases.get(alias)

    def all(self) -> list[PublishedAlias]:
        """Every published alias."""
        return list(self._aliases.values())
