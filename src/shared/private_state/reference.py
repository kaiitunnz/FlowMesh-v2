"""The opaque identity of one activation's mutable private state."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PrivateStateIsolation(StrEnum):
    """Who may hold a valid attachment to a state lineage."""

    # The owning activation alone. Reuse by another activation requires a
    # policy-authorized fork into a new reference.
    ACTIVATION_EXCLUSIVE = "activation_exclusive"


class BundleProfile(StrEnum):
    """The component set a lineage's sealed generations carry."""

    AGENT_HARNESS = "agent_harness"


class ActivationPrivateStateReference(BaseModel):
    """A durable, opaque identity for the private state one activation owns.

    The reference names a state lineage, not its backing: it carries no bytes, path,
    storage URL, secret, or bearer access, and a physical copy of that state neither
    defines nor certifies it. Materializing or writing the state requires a
    :class:`PrivateStateBinding` selecting a generation plus an attachment authorizing
    a holder.
    """

    model_config = ConfigDict(frozen=True)

    reference_id: str
    instance_id: str
    activation_id: str
    tenant: str | None = None
    isolation: PrivateStateIsolation = PrivateStateIsolation.ACTIVATION_EXCLUSIVE
    profile: BundleProfile = BundleProfile.AGENT_HARNESS
