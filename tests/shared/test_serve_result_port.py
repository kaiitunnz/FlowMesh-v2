"""A serve result reports no engine port, and a stored one with a port still reads."""

import pytest
from flowmesh.models.result import DevModelResult as SdkDevModelResult
from flowmesh.models.result import ServeResult as SdkServeResult
from pydantic import BaseModel

from shared.schemas.result import DevModelResult, ServeResult


@pytest.mark.parametrize(
    ("model", "task_type"),
    [
        (ServeResult, "serve"),
        (DevModelResult, "dev_model"),
        (SdkServeResult, "serve"),
        (SdkDevModelResult, "dev_model"),
    ],
)
def test_a_serve_result_port_is_optional(
    model: type[BaseModel], task_type: str
) -> None:
    fresh = model.model_validate({"task_type": task_type, "model": "m"})
    stored = model.model_validate({"task_type": task_type, "model": "m", "port": 8001})
    assert fresh.model_dump().get("port") is None
    assert stored.model_dump()["port"] == 8001
