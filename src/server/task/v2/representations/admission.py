from pydantic import BaseModel, ConfigDict

from .operators import ServiceDependency
from .plan import ResidencyIntent, ResidencyWarmth, ServiceFamilyRequirement


class ResidentAdmissionBinding(BaseModel):
    """What one dispatchable resident consumer binds, read from its plan node.

    Joins the normalized logical ``dependency`` — the family identity, interface,
    and model compatibility a claim admits against — with the screened physical
    annotations of the same versioned node. Resident control reads the dependency
    for identity and the intent for its residency preference, so plan and
    admission cannot disagree about one node. It is a read-only projection: it
    names no replica, worker, or claim, and carries no admission authority.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: str
    dependency: ServiceDependency
    requirement: ServiceFamilyRequirement | None = None
    intent: ResidencyIntent | None = None

    @property
    def warmth(self) -> ResidencyWarmth | None:
        """The screened residency warmth preference, when the plan carries one."""
        return self.intent.warmth if self.intent is not None else None

    def compatible(self) -> bool:
        """Whether the physical requirement matches the logical dependency.

        A requirement that names another family, engine-batch key, or isolation
        domain describes a different node; its residency preference is not this
        dependency's to read.
        """
        if (requirement := self.requirement) is None:
            return False
        return (
            requirement.family == self.dependency.service_family
            and requirement.engine_batch_key == self.dependency.engine_batch_key
            and requirement.isolation == self.dependency.isolation
        )
