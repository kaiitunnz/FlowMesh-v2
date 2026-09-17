"""Attribute keys and span names for the FlowMesh telemetry substrate.

Every ``flowmesh.*`` string a span or metric can carry is defined here once, so no
other module hard-codes one. Span attributes live in exactly two namespaces:
``flowmesh.logical.*`` (workflow-level semantics) and ``flowmesh.physical.*``
(scheduling and carriage diagnostics) — a consumer building the logical view reads the
former only.
"""

from enum import StrEnum

from opentelemetry.semconv.attributes.service_attributes import (
    SERVICE_NAME,
    SERVICE_VERSION,
)

from shared.schemas.network import Transport

__all__ = [
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "ServiceName",
    "ProcessRole",
    "RESOURCE_NODE_ID",
    "RESOURCE_WORKER_ID",
    "RESOURCE_ROLE",
    "LOGICAL_ATTRIBUTE_PREFIX",
    "PHYSICAL_ATTRIBUTE_PREFIX",
    "LOGICAL_WORKFLOW_ID",
    "LOGICAL_OPERATOR_ID",
    "LOGICAL_ACTIVATION_ID",
    "LOGICAL_SCOPE_ID",
    "LOGICAL_PARENT_ACTIVATION_ID",
    "LOGICAL_LOOP_TIME",
    "LOGICAL_CHILD_INDEX",
    "LOGICAL_OPERATOR_KIND",
    "LOGICAL_RESULT_SLOT",
    "LOGICAL_OUTCOME",
    "PHYSICAL_WORK_ITEM_ID",
    "PHYSICAL_ATTEMPT_ID",
    "PHYSICAL_ATTEMPT_NO",
    "PHYSICAL_TASK_ID",
    "PHYSICAL_INVOCATION_ID",
    "PHYSICAL_WORKER_ID",
    "PHYSICAL_NODE_ID",
    "PHYSICAL_ALTERNATIVE_ID",
    "PHYSICAL_CLAIM_ID",
    "PHYSICAL_PERMIT_ID",
    "PHYSICAL_HANDOFF_ID",
    "PHYSICAL_REPLICA_ID",
    "GPU_ATTRIBUTE_PREFIX",
    "GPU_INDEX",
    "GPU_UUID",
    "PHYSICAL_SERVICE_FAMILY",
    "PHYSICAL_RELAY_SESSION_ID",
    "PHYSICAL_TRANSPORT",
    "PHYSICAL_MODEL",
    "PHYSICAL_STAGE",
    "PHYSICAL_WINDOW",
    "PHYSICAL_RETRY_OF_ATTEMPT",
    "SPAN_WORKFLOW",
    "SPAN_OPERATOR",
    "SPAN_EPISODE",
    "SPAN_ATTEMPT",
    "SPAN_BOUNDARY",
    "SPAN_TASK",
    "SPAN_EGRESS",
    "SPAN_ENGINE_REQUEST",
    "SPAN_SANDBOX_COMMAND",
    "CONTROL_SPAN_PREFIX",
    "TRANSPORT_SPAN_PREFIX",
    "ControlPlaneStage",
    "ControlPlaneWindow",
    "control_span_name",
    "transport_span_name",
]


class ServiceName(StrEnum):
    """The ``service.name`` resource value for each FlowMesh process."""

    SERVER = "flowmesh-server"
    SUPERVISOR = "flowmesh-supervisor"
    WORKER = "flowmesh-worker"


class ProcessRole(StrEnum):
    """The ``flowmesh.role`` resource value for each FlowMesh process."""

    ROOT = "root"
    SUPERVISOR = "supervisor"
    WORKER = "worker"


# Resource attributes (every span and metric from a process).
RESOURCE_NODE_ID = "flowmesh.node_id"
RESOURCE_WORKER_ID = "flowmesh.worker_id"
RESOURCE_ROLE = "flowmesh.role"

LOGICAL_ATTRIBUTE_PREFIX = "flowmesh.logical."
PHYSICAL_ATTRIBUTE_PREFIX = "flowmesh.physical."

# flowmesh.logical.* — workflow-level semantics.
LOGICAL_WORKFLOW_ID = f"{LOGICAL_ATTRIBUTE_PREFIX}workflow_id"
LOGICAL_OPERATOR_ID = f"{LOGICAL_ATTRIBUTE_PREFIX}operator_id"
LOGICAL_ACTIVATION_ID = f"{LOGICAL_ATTRIBUTE_PREFIX}activation_id"
LOGICAL_SCOPE_ID = f"{LOGICAL_ATTRIBUTE_PREFIX}scope_id"
LOGICAL_PARENT_ACTIVATION_ID = f"{LOGICAL_ATTRIBUTE_PREFIX}parent_activation_id"
LOGICAL_LOOP_TIME = f"{LOGICAL_ATTRIBUTE_PREFIX}loop_time"
LOGICAL_CHILD_INDEX = f"{LOGICAL_ATTRIBUTE_PREFIX}child_index"
LOGICAL_OPERATOR_KIND = f"{LOGICAL_ATTRIBUTE_PREFIX}operator_kind"
LOGICAL_RESULT_SLOT = f"{LOGICAL_ATTRIBUTE_PREFIX}result_slot"
LOGICAL_OUTCOME = f"{LOGICAL_ATTRIBUTE_PREFIX}outcome"

# flowmesh.physical.* — scheduling and carriage diagnostics.
PHYSICAL_WORK_ITEM_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}work_item_id"
PHYSICAL_ATTEMPT_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}attempt_id"
PHYSICAL_ATTEMPT_NO = f"{PHYSICAL_ATTRIBUTE_PREFIX}attempt_no"
PHYSICAL_TASK_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}task_id"
PHYSICAL_INVOCATION_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}invocation_id"
PHYSICAL_WORKER_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}worker_id"
PHYSICAL_NODE_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}node_id"
PHYSICAL_ALTERNATIVE_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}alternative_id"
PHYSICAL_CLAIM_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}claim_id"
PHYSICAL_PERMIT_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}permit_id"
PHYSICAL_HANDOFF_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}handoff_id"
PHYSICAL_REPLICA_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}replica_id"
PHYSICAL_SERVICE_FAMILY = f"{PHYSICAL_ATTRIBUTE_PREFIX}service_family"

GPU_ATTRIBUTE_PREFIX = "flowmesh.gpu."
GPU_INDEX = f"{GPU_ATTRIBUTE_PREFIX}index"
GPU_UUID = f"{GPU_ATTRIBUTE_PREFIX}uuid"
PHYSICAL_RELAY_SESSION_ID = f"{PHYSICAL_ATTRIBUTE_PREFIX}relay_session_id"
PHYSICAL_TRANSPORT = f"{PHYSICAL_ATTRIBUTE_PREFIX}transport"
PHYSICAL_MODEL = f"{PHYSICAL_ATTRIBUTE_PREFIX}model"
PHYSICAL_STAGE = f"{PHYSICAL_ATTRIBUTE_PREFIX}stage"
PHYSICAL_WINDOW = f"{PHYSICAL_ATTRIBUTE_PREFIX}window"
PHYSICAL_RETRY_OF_ATTEMPT = f"{PHYSICAL_ATTRIBUTE_PREFIX}retry_of_attempt"

# Span names — the catalog's fixed (non-parameterized) names.
SPAN_WORKFLOW = "flowmesh.workflow"
SPAN_OPERATOR = "flowmesh.operator"
SPAN_EPISODE = "flowmesh.episode"
SPAN_ATTEMPT = "flowmesh.attempt"
SPAN_BOUNDARY = "flowmesh.boundary"
SPAN_TASK = "flowmesh.task"
SPAN_EGRESS = "flowmesh.egress"
SPAN_ENGINE_REQUEST = "flowmesh.engine_request"
SPAN_SANDBOX_COMMAND = "flowmesh.sandbox.command"

CONTROL_SPAN_PREFIX = "flowmesh.control."
TRANSPORT_SPAN_PREFIX = "flowmesh.transport."


class ControlPlaneStage(StrEnum):
    """The control-plane boundaries instrumented as ``flowmesh.control.<stage>``.

    There is deliberately no ``nested`` member: a span tree already expresses
    enclosure, so a consumer wanting self-time subtracts child spans instead of
    reading a hand-maintained flag.
    """

    COMPILE_LOWER = "compile_lower"
    COMPILE_ASSEMBLE = "compile_assemble"
    COMPILE_EPISODES = "compile_episodes"
    COMPILE_FINALIZE = "compile_finalize"
    COMPILE_VALIDATE = "compile_validate"
    ENGINE_BUILD = "engine_build"
    DS_INITIAL_ADVANCE = "ds_initial_advance"
    DS_DRIVE = "ds_drive"
    DISPATCH = "dispatch"
    ADMISSION = "admission"
    PERMIT = "permit"
    RELAY = "relay"
    LEDGER_SNAPSHOT = "ledger_snapshot"


class ControlPlaneWindow(StrEnum):
    """Where a control-plane stage fires relative to a task's lifetime.

    Set explicitly at each instrumentation point (``flowmesh.physical.window``), never
    derived from span-tree shape: the correct anchor is where a task is submitted, not
    where the code that measures it happens to run.
    """

    SUBMIT = "submit"
    QUEUE = "queue"
    POST_START = "post_start"


def control_span_name(stage: ControlPlaneStage) -> str:
    return f"{CONTROL_SPAN_PREFIX}{stage}"


def transport_span_name(transport: Transport) -> str:
    return f"{TRANSPORT_SPAN_PREFIX}{transport}"
