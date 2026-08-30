"""Prefixed identifier factories for FlowMesh objects.

Each FlowMesh object kind gets a short, dashed prefix so that IDs are
self-describing in logs, API responses, and CLI output.
"""

import secrets
import uuid

PREFIX_WORKFLOW = "wfl"
PREFIX_TASK = "tsk"
PREFIX_WORKER = "wkr"
PREFIX_NODE = "nde"
PREFIX_SSH_CONNECTION = "scn"
PREFIX_SSH_SESSION = "ssn"
PREFIX_SUPERVISOR_COMMAND = "cmd"
PREFIX_ACTIVATION = "act"
PREFIX_SCOPE = "scp"
PREFIX_WORK_ITEM = "wki"
PREFIX_ATTEMPT = "att"
PREFIX_INVOCATION = "inv"
PREFIX_AUTHORITY_GRANT = "agr"
PREFIX_IDEMPOTENCY_KEY = "idm"
PREFIX_MODEL_SECRET = "msk"  # nosec B105 - object-id prefix, not a credential


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _uuid_hex() -> str:
    return uuid.uuid4().hex


def new_workflow_id() -> str:
    return f"{PREFIX_WORKFLOW}-{_uuid_str()}"


def new_task_id() -> str:
    return f"{PREFIX_TASK}-{_uuid_str()}"


def new_worker_id(seq: int) -> str:
    return f"{PREFIX_WORKER}-{seq}"


def new_node_id(seq: int) -> str:
    return f"{PREFIX_NODE}-{seq}"


def new_ssh_connection_id() -> str:
    return f"{PREFIX_SSH_CONNECTION}-{secrets.token_hex(16)}"


def new_ssh_session_id() -> str:
    return f"{PREFIX_SSH_SESSION}-{_uuid_hex()}"


def new_supervisor_command_id() -> str:
    return f"{PREFIX_SUPERVISOR_COMMAND}-{_uuid_hex()}"


def new_activation_id() -> str:
    return f"{PREFIX_ACTIVATION}-{_uuid_hex()}"


def new_scope_id() -> str:
    return f"{PREFIX_SCOPE}-{_uuid_hex()}"


def new_work_item_id() -> str:
    return f"{PREFIX_WORK_ITEM}-{_uuid_hex()}"


def new_attempt_id() -> str:
    return f"{PREFIX_ATTEMPT}-{_uuid_hex()}"


def new_invocation_id() -> str:
    return f"{PREFIX_INVOCATION}-{_uuid_hex()}"


def new_authority_grant_id() -> str:
    return f"{PREFIX_AUTHORITY_GRANT}-{_uuid_hex()}"


def new_idempotency_key() -> str:
    return f"{PREFIX_IDEMPOTENCY_KEY}-{_uuid_hex()}"


def new_model_secret_ref() -> str:
    return f"{PREFIX_MODEL_SECRET}-{secrets.token_hex(16)}"


__all__ = [
    "PREFIX_ACTIVATION",
    "PREFIX_ATTEMPT",
    "PREFIX_AUTHORITY_GRANT",
    "PREFIX_IDEMPOTENCY_KEY",
    "PREFIX_INVOCATION",
    "PREFIX_MODEL_SECRET",
    "PREFIX_NODE",
    "PREFIX_SCOPE",
    "PREFIX_SSH_CONNECTION",
    "PREFIX_SSH_SESSION",
    "PREFIX_SUPERVISOR_COMMAND",
    "PREFIX_TASK",
    "PREFIX_WORK_ITEM",
    "PREFIX_WORKER",
    "PREFIX_WORKFLOW",
    "new_activation_id",
    "new_attempt_id",
    "new_authority_grant_id",
    "new_idempotency_key",
    "new_invocation_id",
    "new_model_secret_ref",
    "new_node_id",
    "new_scope_id",
    "new_ssh_connection_id",
    "new_ssh_session_id",
    "new_supervisor_command_id",
    "new_task_id",
    "new_work_item_id",
    "new_worker_id",
    "new_workflow_id",
]
