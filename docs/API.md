# API reference (common endpoints)

Server runs at `http://localhost:8000` by default. The router source of
truth is `src/server/routers/v1/*.py`.

Workflow and task statuses (used in payloads and filters):
`PENDING`, `DISPATCHED`, `FAILED`, `CANCELLED`, `DONE`. Worker statuses:
`STARTING`, `IDLE`, `BUSY`, `STOPPING`, `STOPPED`.

## Authentication

Every endpoint under `/api/v1/*` (REST and WebSocket) authenticates via
the `Authorization: Bearer <token>` header. The token is routed through
the registered `IdentityProvider` chain (see `docs/PLUGINS.md`); with
no providers registered, every caller resolves to a default admin
principal. After authentication, every resource-scoped endpoint runs
the registered `PermissionChecker` chain; with no checkers registered,
all calls succeed (open by default). The classification of resource type
and action per endpoint lives in `src/server/routers/v1/`. Workers
self-authenticate the same way, sending `FLOWMESH_API_KEY` as the bearer.

## Workflows

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/workflows` | Submit a workflow. Body is YAML (`text/plain`) or JSON; set `Workflow-Format: n8n` for n8n graphs. |
| POST | `/api/v1/workflows/validate` | Parse without executing; for `flowmesh/v2` returns the compiled template/plan inspection. |
| GET | `/api/v1/workflows` | List workflows as cursor pages. Filters: `workflow_id`, `status`, `task_ids`, `dispatched_tasks`, `completed_tasks`, `failed_tasks`, `cancelled_tasks`. |
| GET | `/api/v1/workflows/{id}` | Workflow details + per-task summary. |
| GET | `/api/v1/workflows/{id}/logs` | Query logs (`limit`, `before`/`after` cursors). |
| GET | `/api/v1/workflows/{id}/logs/stream` | SSE log stream. |
| POST | `/api/v1/workflows/{id}/cancel` | Cancel a workflow and all in-flight tasks. |
| GET | `/api/v1/workflows/{id}/outputs` | List published outputs. |
| GET | `/api/v1/workflows/{id}/outputs/{name}` | Get one published output's value. |

### Published outputs

An output is named by the node it is published on, and a fetch selects a spawn's member
by `scope` and `key`. A spawn whose input failed publishes one failed member with no
scope or key, fetched by name alone. Errors carry `detail.code`:

| Status | `code` | Meaning |
|--------|--------|---------|
| 400 | `invalid_request`, `invalid_cursor` | A collection fetched without `scope` and `key`, or a malformed cursor. |
| 404 | `output_not_found` | No published output by that name, or no such member in a settled workflow. |
| 409 | `output_pending` | The member has not settled. |
| 503 | `content_unavailable` | The content store cannot be reached; retry. |
| 500 | `output_unreadable` | The member's bound content is missing or corrupt. |

## Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/tasks` | List tasks as cursor pages. Filters: identity and ownership `task_id`, `workflow_id`, `owner_id`, `org_id`, `supplier_id`, `local_name`, `graph_node_name`, `parent_task_id`, `merged_parent_id`; state `status`, `category`, `task_type`, `resident`, `completed`, `failed`, `attempts`, `max_attempts`; placement `assigned_worker`, `selected_worker`, `shard_index`, `shard_total`; and the lists `depends_on`, `pending_dependencies`, `dependents`, `merged_children`. |
| GET | `/api/v1/tasks/{id}` | Task details. |
| GET | `/api/v1/tasks/{id}/logs` | Query task logs. |
| GET | `/api/v1/tasks/{id}/logs/stream` | SSE task log stream. |
| POST | `/api/v1/tasks/{id}/stop` | Stop a running SSH or `serve` task; the task ends `DONE`. |

## Results

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/results/{task_id}` | Get task result JSON, read from the shared content store. |
| GET | `/api/v1/results/{task_id}/bundle` | Download tar.gz bundle (`?include=results,artifacts,logs,all`). |
| POST | `/api/v1/results/{task_id}/files` | Upload artifact (multipart). A path segment starting with `.fm-tmp-` is reserved and refused. |
| GET | `/api/v1/results/{task_id}/files/{filename}` | Download artifact. |
| GET | `/api/v1/results/{task_id}/logs` | Download archived `logs.jsonl`. |

## Content

The outcome-finalization index: the binding from a fabric idempotency key to the content it materialized, so a re-driven producer resolves its first materialization instead of producing again. Content bytes never cross the server — a worker reads and writes them directly in the shared content store. The binding lands in the scope control assigned the key's work; a `scope` the caller names must match it.

| Method | Path | Description |
|--------|------|-------------|
| PUT | `/api/v1/content/finalizations?idem={idm}&scope={scope}` | Bind the `ContentReference` in the body to an idempotency key; returns the `OutcomeManifest`. |
| GET | `/api/v1/content/finalizations?idem={idm}&scope={scope}` | Resolve the `OutcomeManifest` already bound to an idempotency key. |

## Traces

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/traces/workflows/{workflow_id}/{trace_type}` | Fetch workflow trace JSONL rows. `trace_type` is `spans`, `assets`, or `lineage`. |
| GET | `/api/v1/traces/workflows/analyze/{workflow_id}` | Run the trace analyzer and return a profile summary. |
| POST | `/api/v1/traces/tasks/{task_id}/{trace_type}` | Upload a per-task trace JSONL file. `trace_type` is `spans`, `assets`, or `lineage`. |

## Workers and nodes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/workers` | List workers. Filters: identity and status `id`, `alias`, `namespace`, `cluster`, `node_id`, `node_alias`, `version`, `status`, `stale`; the lists `tags`, `cached_models`, `cached_datasets`, `capabilities.supported_task_types`; `capabilities.ssh_noninteractive`; and hardware `hardware.cpu.model`, `hardware.gpu.driver_version`, `hardware.gpu.cuda_version`, `hardware.network.ip`. |
| GET | `/api/v1/workers/{id}` | Worker details + hardware. |
| GET | `/api/v1/nodes` | List nodes (supervisors). Filters: `id`, `alias`, `namespace`, `cluster`, `version`, `tags`. |
| POST | `/api/v1/nodes/register` | Register a node; `409 Conflict` while another live node holds the same alias, with the held lease's `lease_remaining_ms`. |
| GET | `/api/v1/nodes/{id}/workers` | List workers under a node. Filters: identity and status `id`, `alias`, `namespace`, `cluster`, `node_id`, `node_alias`, `provider`, `version`, `status`; and hardware `hardware.cpu.model`, `hardware.cpu.arch`, `hardware.cpu.name`, `hardware.gpu.driver_version`, `hardware.gpu.cuda_version`, `hardware.gpu.gpu_arch`, `hardware.network.ip`, `hardware.network.public_ipaddr`, `hardware.network.geolocation`, `hardware.host.os_version`. |
| GET | `/api/v1/nodes/workers` | List workers across every node, with the same filters. |
| POST | `/api/v1/nodes/{id}/workers/register` | Register worker under node. |
| POST | `/api/v1/nodes/{id}/workers/{alias}/{start,stop}` | Start/stop a worker. |

`/api/v1/stack/workers/...` wraps node-registered workers with local-only
container lifecycle and is what `flowmesh stack worker {up,down,...}`
calls.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/stack/workers` | List this node's workers. Filters: those of `/api/v1/nodes/{id}/workers` other than `node_id` and `version`. |
| GET | `/api/v1/stack/workers/providers` | Worker providers available on this node (e.g. `docker`, `external`, `vastai`). |
| POST | `/api/v1/stack/workers` | Create a worker on this node; `409 Conflict` when the requested `provider` is unavailable here. |

## SSH

| Method | Path | Description |
|--------|------|-------------|
| WS | `/api/v1/ssh/tasks/{task_id}/proxy` | WebSocket SSH proxy for proxy- and forward-mode SSH tasks. |
| GET | `/api/v1/ssh/connections` | List active SSH proxy/forward connections the server is relaying. Filters: `connection_id`, `session_id`, `access_mode`, `task_id`, `workflow_id`, `worker_id`, `node_id`, `username`, `source_ip`, `source_port`. |

Server policy toggles: `ENABLE_SERVER_SSH_PROXY`,
`ENABLE_SERVER_PORT_FORWARD`, `ENABLE_SERVER_SSH_CONNECTION_REGISTRY`.

## Serve

| Method | Path | Description |
|--------|------|-------------|
| GET, POST, PUT, DELETE, OPTIONS, HEAD | `/api/v1/serve/tasks/{task_id}/{upstream_path:path}` | Reach any endpoint a public `serve` task's engine serves, by its task ID (e.g. `.../v1/chat/completions`, `.../v1/models`). |

The request relays to the task's standing replica unchanged and the engine's own response
comes back unchanged, so an OpenAI-compatible client works against the task-ID route.
Requires a FlowMesh credential with `TASK` read access; the client `Authorization` is not
forwarded upstream.

## Resident

SYSTEM/ADMIN-gated read-only views into resident-capacity control. Empty when resident
capacity is disabled.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/resident/families` | List registered service families. |
| GET | `/api/v1/resident/replicas` | List replica incarnations (state, health, `serve_task_id`, endpoint host/port). Filters: `replica_id`, `family`, `state`, `healthy`, `serve_task_id`, `worker_id`. |
| GET | `/api/v1/resident/claims` | List credit-bearing admission claims and per-replica held credit. |

Endpoint responses carry host and port only — never an `api_key`. Read a replica's serving
logs via its `serve_task_id` through `GET /api/v1/tasks/{id}/logs`.

## Network

SYSTEM/ADMIN-gated route-discovery diagnostics and a reachability probe. With
`NETWORK_PLANE_ENABLED=false` these paths return 404. The echo carries no
resident traffic and dials only with `NETWORK_PLANE_PEER_ENABLED=true`. See
[`NETWORK_PLANE.md`](NETWORK_PLANE.md).

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/network/echo` | Resolve a route to a target listener and probe each forward-dial transport in order until one answers, updating reachability. |
| GET | `/api/v1/network/endpoints` | List advertised node network-plane endpoints. |
| GET | `/api/v1/network/reachability` | List derived directional reachability entries. |

## System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Top-level health check. |
| GET | `/api/v1/system/version` | Server version. |
| GET | `/api/v1/system/metrics` | System metrics snapshot. |

## Cursor pagination

List endpoints (`/api/v1/workflows`, `/api/v1/tasks`, log queries,
published outputs) accept `limit` and `before` / `after` cursors.
Cursors are opaque; do not parse them client-side.

Workflows and tasks are ordered by submission and return
`{entries, next_cursor, prev_cursor}`. Without a cursor, a request returns
the newest page; `before=<prev_cursor>` returns the next older page and
`after=<next_cursor>` the next newer one, each in submission order. `limit`
defaults to 100, at most 1000. A request setting both cursors is a `400` with
`detail.code` `invalid_request`, and a malformed cursor is a `400` with
`invalid_cursor`.

## List filters

A list route with a `Filters:` entry matches each filter exactly, as a string.
A boolean field matches `true`, `1`, `yes` or `on` and `false`, `0`, `no` or
`off`, in any case, and a comma-separated `tags` string matches any of its tags.
A repeated filter matches any of its values, and different filters all apply. A
list field matches when it holds a value, a dotted filter reads a nested field,
and a field that is unset, or whose parent is unset, matches `null`. Any other
query key is a `400` with `detail.code` `invalid_request`.
