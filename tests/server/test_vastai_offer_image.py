"""A VastAI worker gets the image its offer's hardware can run."""

from unittest.mock import MagicMock

import pytest

from server.hooks import PrincipalContext
from server.supervisor.adapters.base import WorkerTokenType
from server.supervisor.adapters.vastai import VastAIWorkerAdapter, VastAIWorkerConfig
from server.utils.helpers import ResourcePool


async def _launched_image(gpu_name: str | None) -> str:
    client = MagicMock()
    client.search_offers.return_value = [{"id": 7, "gpu_name": gpu_name}]
    client.create_instance.return_value = {"success": True, "new_contract": 70}
    adapter = VastAIWorkerAdapter(
        token=WorkerTokenType("vast_0.token"),
        name="vast_0",
        config=VastAIWorkerConfig(docker_registry="reg", version="v1"),
        vastai_client=client,
        instance_pool=ResourcePool(),
        owner=PrincipalContext(
            principal_id="u",
            org_id="o",
            external_id="u",
            principal_type="user",
            scopes=[],
        ),
    )
    assert await adapter.start()
    return str(client.create_instance.call_args.kwargs["image"])


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_name", ["N/A", "n/a", " ", "", None, "None"])
async def test_a_gpuless_offer_gets_the_cpu_image(gpu_name: str | None) -> None:
    assert await _launched_image(gpu_name) == "reg/flowmesh_worker:v1-cpu"


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_name", ["RTX 4090", "H100 SXM", "B200"])
async def test_a_gpu_offer_gets_the_gpu_image(gpu_name: str) -> None:
    assert await _launched_image(gpu_name) == "reg/flowmesh_worker:v1-gpu"
