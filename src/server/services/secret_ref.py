from collections.abc import Mapping

from pydantic import SecretStr


class SecretRefResolver:
    """Resolve an authorized ``secret_ref`` to its server-side secret value.

    Policy-scoped: only a ref the deployment registered resolves, so a workflow can
    name a secret without ever carrying it. The values stay server-side; a caller uses
    the resolved secret only on the server-to-upstream path.
    """

    def __init__(self, secrets: Mapping[str, SecretStr]) -> None:
        self._secrets = dict(secrets)

    def resolve(self, ref: str | None) -> SecretStr | None:
        return self._secrets.get(ref) if ref else None
