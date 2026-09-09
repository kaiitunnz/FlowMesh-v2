"""The control-relay carriage realizes the one transport it is for, and refuses others.

Control selects the transport for an attempt and hands the drive a carriage plan naming
it. ``ControlRelayCarriage`` returns its base sink for a ``control_relay`` plan and
refuses any other, so a direct or node selection never rides the relay by accident until
a PR adds a carriage that actually implements it.
"""

import pytest

from shared.network.relay_frame import RelayFrame
from shared.resident.carriage import (
    CarriageUnavailable,
    ControlRelayCarriage,
    ResidentCarriagePlan,
)


class _Sink:
    async def send(self, frame: RelayFrame) -> None:  # pragma: no cover - not driven
        pass


def test_a_control_relay_plan_selects_the_base_sink() -> None:
    sink = _Sink()
    carriage = ControlRelayCarriage(sink)
    assert carriage.select(ResidentCarriagePlan(session_id="rly-1")) is sink


def test_a_non_relay_transport_is_refused_not_reinterpreted() -> None:
    carriage = ControlRelayCarriage(_Sink())
    with pytest.raises(CarriageUnavailable):
        carriage.select(
            ResidentCarriagePlan(session_id="rly-1", selected_transport="worker_direct")
        )
