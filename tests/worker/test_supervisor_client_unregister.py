"""A worker's UNREGISTER says whether its shutdown was requested."""

from typing import Any, cast

import pytest

from tests.worker.test_supervisor_client_dispatch_id import _client


@pytest.mark.parametrize("graceful", [True, False])
def test_an_unregister_carries_whether_the_shutdown_was_requested(
    graceful: bool,
) -> None:
    client = _client()
    client.unregister(graceful)

    _generation, frame = cast(
        tuple[int, dict[str, Any]], client._event_queue.get_nowait()
    )
    assert frame["type"] == "UNREGISTER"
    assert frame["graceful"] is graceful
