"""How the fabric names a stored immutable object.

A reference names bytes by the scope they are isolated in and the digest they hash to,
and carries only what a reader needs to fetch, verify, and interpret them. It is not a
location: it holds no endpoint, holder, URL, or worker-local path, so the same reference
resolves wherever the object is held and stays valid when its holder changes. It is not
a name either — an invocation outcome, a declared result, and a catalogued artifact each
keep their own binding to it, and nothing here says which of them these bytes are.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

OCTET_STREAM = "application/octet-stream"
# The only encoding the fabric stores today: bytes are held exactly as written.
CONTENT_ENCODING_IDENTITY = "identity"


class DigestAlgorithm(StrEnum):
    """The hash an object's immutable identity is taken with."""

    SHA256 = "sha256"


class ContentReference(BaseModel):
    """An authorization-scoped, digest-named reference to one immutable object.

    ``authorization_scope`` is the isolation namespace the authorized write context
    assigned, never an assertion a holder makes for itself, and never a credential:
    it says which namespace the object lives in, not that anyone may read it.
    Identity is the scope, the algorithm, and the digest together, so equal bytes in two
    scopes are two objects and deduplication never crosses a scope.
    """

    model_config = ConfigDict(frozen=True)

    authorization_scope: str
    digest_algorithm: DigestAlgorithm = DigestAlgorithm.SHA256
    content_digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = OCTET_STREAM
    content_encoding: str = CONTENT_ENCODING_IDENTITY

    @property
    def identity(self) -> tuple[str, str, str]:
        """The triple two references are the same object by."""
        return (
            self.authorization_scope,
            self.digest_algorithm.value,
            self.content_digest,
        )


__all__ = [
    "CONTENT_ENCODING_IDENTITY",
    "OCTET_STREAM",
    "ContentReference",
    "DigestAlgorithm",
]
