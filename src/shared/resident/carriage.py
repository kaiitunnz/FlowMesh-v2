"""The per-attempt transport carriage a claim-gated origin drive sends its frames over.

Resident-capacity control resolves an ordered route for every admitted invocation and
selects one transport candidate for the attempt. The carriage is how an origin drive
obtains the frame sink for that selection without knowing which transport it is: it
takes a ``ResidentCarriagePlan`` naming the selected transport and returns the sink
that carries the attempt's frames. Selecting the transport stays control's decision;
the carriage only realizes it, and never reinterprets one transport as another.

The sole realization here is ``ControlRelayCarriage``, the universal reverse-rendezvous
relay every deployment can reach. A trusted direct or node-relay carriage is a later
addition behind the same factory; declaring the seam now lets one slot in without
changing any drive that already takes its sink from a carriage.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .transport import ResidentFrameSink

# The base transport candidate every healthy attachment resolves, and the only one a
# carriage realizes here.
CONTROL_RELAY = "control_relay"


class CarriageUnavailable(Exception):
    """The plan names a transport this deployment has no carriage for."""


class ResidentCarriagePlan(BaseModel):
    """The transport selection control carries alongside an ``AdmissionHandoff``.

    It names the attempt's session and the transport candidate control selected from
    the resolved route, plus the route's epoch and listener generation for diagnostics.
    It is not authority: it mints no claim and chooses no capacity, and rides beside the
    handoff rather than inside it, so a carriage realizes only what control decided.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    selected_transport: str = CONTROL_RELAY
    route_epoch: int = 0
    listener_generation: int = 0


class ClaimGatedServiceCarriage(Protocol):
    """Realizes the transport a plan selected as an origin drive's frame sink."""

    def select(self, plan: ResidentCarriagePlan) -> ResidentFrameSink: ...


class ControlRelayCarriage:
    """The universal reverse-rendezvous relay carriage, over one base frame sink.

    Every attempt this carriage serves rides ``control_relay`` over the same base sink —
    the origin's authenticated attachment for a worker, the root's internal rendezvous
    attachment for the proxy. It refuses a plan naming any other transport rather than
    relaying it silently, so a direct or node selection never rides the relay by chance.
    """

    def __init__(self, base_sink: ResidentFrameSink) -> None:
        self._base_sink = base_sink

    def select(self, plan: ResidentCarriagePlan) -> ResidentFrameSink:
        if plan.selected_transport != CONTROL_RELAY:
            raise CarriageUnavailable(plan.selected_transport)
        return self._base_sink
