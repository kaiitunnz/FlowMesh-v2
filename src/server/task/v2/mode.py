from enum import StrEnum

V2_API_VERSION = "flowmesh/v2"


class ExecutionMode(StrEnum):
    """Execution track selected for a submission."""

    V1 = "v1"
    V2 = "v2"

    @classmethod
    def from_api_version(cls, api_version: str | None) -> "ExecutionMode":
        """Select the v2 track only for the explicit v2 ``apiVersion``.

        Every other value — ``flowmesh/v1``, omitted, or anything else — stays on
        the default v1 track, so existing submissions are unaffected.
        """
        if api_version is not None and api_version.strip() == V2_API_VERSION:
            return cls.V2
        return cls.V1
