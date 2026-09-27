"""The SDK lists and fetches a workflow's published outputs, and types their errors."""

import pytest
import respx
from flowmesh import (
    ContentUnavailableError,
    FlowMesh,
    NotFoundError,
    OutputPendingError,
    OutputUnreadableError,
    ValidationError,
)
from flowmesh.models import OutputOutcome

from server.schemas.outputs import OutputOutcome as SrvOutputOutcome
from server.schemas.outputs import WorkflowOutputEntry as SrvWorkflowOutputEntry
from server.schemas.outputs import WorkflowOutputPage as SrvWorkflowOutputPage
from server.schemas.outputs import WorkflowOutputValue as SrvWorkflowOutputValue
from shared.schemas.result import EchoResult

from .router_app import TEST_BASE_URL, route_url


@pytest.fixture
def client() -> FlowMesh:
    return FlowMesh(base_url=TEST_BASE_URL, api_key="flm-test-key")


_PAGE = SrvWorkflowOutputPage(
    entries=[
        SrvWorkflowOutputEntry(
            cursor="c-1",
            name="fanout",
            cardinality="keyed_collection",
            value_type="echo",
            scope="scp-1",
            key="0",
            outcome=SrvOutputOutcome.SUCCESS,
        )
    ],
    next_cursor="c-1",
    prev_cursor="c-1",
    open=True,
)


@respx.mock
def test_list_outputs_passes_its_filters_and_cursor(client: FlowMesh) -> None:
    route = respx.get(route_url("list_outputs", workflow_id="wfl-1")).respond(
        json=_PAGE.model_dump(mode="json")
    )

    page = client.workflows.list_outputs(
        "wfl-1", limit=5, after="c-0", output="fanout", scope="scp-1"
    )

    params = dict(route.calls[0].request.url.params)
    assert params == {
        "limit": "5",
        "after": "c-0",
        "output": "fanout",
        "scope": "scp-1",
    }
    assert page.entries[0].key == "0" and page.open
    assert page.entries[0].outcome is OutputOutcome.SUCCESS


@respx.mock
def test_get_output_selects_a_member_and_types_its_value(client: FlowMesh) -> None:
    value = SrvWorkflowOutputValue(
        name="fanout",
        cardinality="keyed_collection",
        value_type="echo",
        scope="scp-1",
        key="0",
        outcome=SrvOutputOutcome.SUCCESS,
        value=EchoResult(items=[]),
    )
    route = respx.get(
        route_url("get_output", workflow_id="wfl-1", output_name="fanout")
    ).respond(json=value.model_dump(mode="json"))

    fetched = client.workflows.get_output("wfl-1", "fanout", scope="scp-1", key="0")

    assert dict(route.calls[0].request.url.params) == {"scope": "scp-1", "key": "0"}
    assert fetched.value is not None
    assert type(fetched.value).__name__ == "EchoResult"


@respx.mock
@pytest.mark.parametrize(
    ("status", "code", "error"),
    [
        (409, "output_pending", OutputPendingError),
        (503, "content_unavailable", ContentUnavailableError),
        (500, "output_unreadable", OutputUnreadableError),
        (404, "output_not_found", NotFoundError),
        (400, "invalid_request", ValidationError),
    ],
)
def test_an_output_error_raises_its_typed_exception(
    client: FlowMesh, status: int, code: str, error: type[Exception]
) -> None:
    respx.get(
        route_url("get_output", workflow_id="wfl-1", output_name="summarize")
    ).respond(status, json={"detail": {"code": code, "message": "because"}})

    with pytest.raises(error) as caught:
        client.workflows.get_output("wfl-1", "summarize")
    assert "because" in str(caught.value)
