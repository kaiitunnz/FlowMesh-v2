from pydantic import BaseModel, Field

from shared.schemas.network import NetworkEndpointAdvertisement


class NodeInfo(BaseModel):
    namespace: str = Field(description="Node namespace.")
    cluster: str = Field(description="Node cluster.")
    alias: str = Field(description="Human-readable node alias.")
    version: str | None = Field(default=None, description="Node version.")
    started_at: str = Field(description="Node start timestamp.")
    tags: list[str] = Field(description="Node tags.")
    last_seen: str = Field(description="Last heartbeat timestamp.")
    max_gpu_count: int = Field(description="Total GPU count available on this node.")
    network_endpoint: NetworkEndpointAdvertisement | None = Field(
        default=None, description="Network-plane endpoint advertisement."
    )


__all__ = ["NodeInfo"]
