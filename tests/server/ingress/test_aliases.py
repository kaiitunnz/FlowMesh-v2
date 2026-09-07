"""The published-alias catalog resolves only allowed aliases under tenant policy.

A client selects an alias by name; the catalog maps it to a resident service family and
gates selection by the caller's tenant. An unpublished alias is absent and an
unauthorized tenant is refused, so neither reaches admission.
"""

from server.ingress import AliasCatalog, PublishedAlias
from server.task.v2.representations.operators import ServiceInterface

_CATALOG_JSON = """
{
  "aliases": [
    {"alias": "chat-small", "service_ref": "org/model-a", "allowed_tenants": ["acme"]},
    {"alias": "open", "service_ref": "org/model-b"},
    {"alias": "embed", "service_ref": "org/embed", "interface": "embedding",
     "isolation": "iso-1", "max_output_tokens": 64}
  ]
}
"""


def test_catalog_parses_aliases_and_bare_list_forms():
    catalog = AliasCatalog.from_json(_CATALOG_JSON)
    assert {a.alias for a in catalog.all()} == {"chat-small", "open", "embed"}
    bare = AliasCatalog.from_json('[{"alias": "x", "service_ref": "m"}]')
    assert bare.get("x") is not None
    assert AliasCatalog.from_json("").all() == []


def test_alias_resolves_to_the_same_family_key_a_leaf_would():
    alias = AliasCatalog.from_json(_CATALOG_JSON).get("embed")
    assert alias is not None
    dependency = alias.dependency()
    assert dependency.service_ref == "org/embed"
    assert dependency.interface is ServiceInterface.EMBEDDING
    assert dependency.isolation == "iso-1"
    # The family key is the reuse domain a workflow embedding leaf lands on.
    assert dependency.service_family == "org/embed|embedding|iso=iso-1"


def test_tenant_gate_authorizes_only_listed_tenants():
    restricted = PublishedAlias(
        alias="a", service_ref="m", allowed_tenants=frozenset({"acme"})
    )
    assert restricted.authorizes("acme") is True
    assert restricted.authorizes("other") is False
    assert restricted.authorizes(None) is False


def test_empty_allowed_tenants_authorizes_any_authenticated_tenant():
    unrestricted = PublishedAlias(alias="a", service_ref="m")
    assert unrestricted.authorizes("anyone") is True
    assert unrestricted.authorizes(None) is True
