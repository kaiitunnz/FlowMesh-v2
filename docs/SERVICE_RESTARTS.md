# Service restarts

FlowMesh supports fine-grained, in-place restarts: any Compose service — most
importantly the root server — can be recreated without tearing the cluster down
and without losing in-flight work. The same machinery covers every kind of
restart — a crash, a config change, a single-service redeploy, or a rolling
image bump — because the root's scheduling state is durable and rebuilt on
startup and task lifecycle events are replayable. A restarted root resumes its
in-flight workflows instead of dropping them.

## The restart primitive

`flowmesh stack restart [SERVICE ...]` (see [`CLI.md`](CLI.md)) recreates one or
more Compose services in place, leaving the rest of the stack running:

```bash
flowmesh stack restart server                # recreate just the server in place
flowmesh stack restart redis_control server  # recreate several services in one call
flowmesh stack restart                       # whole-stack drain + down + up
```

A service runs only on a node whose Compose profiles include it, so a node refuses to
restart one it does not run: the Redis services on a worker node, or `otel_collector`
without the `telemetry` profile. For each invocation it:

1. Drains the node's managed workers **once** if any named service manages
   workers (the `server` / supervisor), so their in-flight tasks are released
   and requeued onto other nodes.
2. Recreates only the named services (`--no-deps --force-recreate`), optionally
   pulling a new image (`--pull`, on by default). Redis and any unnamed service
   keep running.
3. Blocks (`--wait`) until the recreated services pass their healthchecks.

With no argument it restarts the whole stack (drain + `down` + `up`). Because
the worker-managing process is the `server` service on both root and worker
nodes, the same command works everywhere.

## What survives a restart

**Worker nodes.** Draining a node tears down its workers. Each worker gives up
the tasks it runs and unregisters, so the server requeues those tasks onto other
eligible nodes without spending an attempt. A v2 task that cannot safely re-run,
such as an `ssh`, `serve`, `api` or training task, fails instead, as it does
when its worker is lost. A draining worker finishes the calls it holds for
suspended steps, of inference and embedding leaves and agents alike, within its
stop window, and a call that cannot finish in time fails its task. A leaf whose
call finished runs its next step on another worker. An agent runs its steps on
the worker holding its sealed private state, so an agent whose private state
only a drained or lost worker held fails at its next step; one given up before
its first seal reruns elsewhere. A worker that leaves without unregistering,
such as one that crashed, is unregistered by its supervisor, and its tasks
requeue at the cost of an attempt when they can safely re-run and fail
otherwise. A recreated node's supervisor re-creates its configured workers,
which re-register themselves on startup. No cordon step is required.

**Undrained supervisor stops.** A supervisor that stops without a drain — a
crash, an OOM kill, or a `docker kill` of the `server` container — leaves its
Docker and Vast.ai workers running. When it starts again it takes them back from
their records: each keeps its container or instance and registers again under a
new id, and the tasks it held requeue at the cost of an attempt when they can
safely re-run and fail otherwise. A worker that does not register again within
five minutes is removed. A survivor keeps the settings and image it was launched
with until a draining `flowmesh stack restart`. Each entry of the configured
workers file sets `worker_config.worker_alias`, and a file missing one creates
none of its workers. A starting supervisor creates only the entries no recorded
worker holds, so removing an entry leaves its survivor running.

- The records live in the supervisor state store (`REDIS_SUPERVISOR_STATE_URL`,
  the control Redis by default). Losing or rolling it back leaves the workers it
  named running untracked; remove them by hand.
- A crash between renting a Vast.ai instance and recording it leaves the
  instance unrecorded: destroy it in Vast.ai, then run
  `flowmesh stack worker down <alias>`.
- Drain a node's Vast.ai workers before changing `VAST_API_KEY`: a supervisor
  forgets, without destroying, the instances its new key cannot see.
- A worker the supervisor launches again after a crash uses the deployment's
  `HF_TOKEN` and `NEBULA_API_TOKEN`, not values its create request set.
- A supervisor never adopts or removes a container or instance no record names,
  so a configured worker whose container name one holds fails to start until it
  is removed by hand.

**Node alias.** A node holds a lease on its `NODE_ALIAS` while it is live and
releases it when it shuts down cleanly, so a restarted node re-registers under
the same alias at once. A node that exits without unregistering (a crash or a
kill) leaves its lease behind; the replacement takes it over once the lease has
gone unrefreshed for half the node heartbeat TTL (60s by default), and startup
waits until then. Registering removes the old node's record, so the node list
shows only the replacement. A node whose alias another live node keeps refreshing
fails at startup.

**Root node.** The root holds the dispatcher's scheduling state in memory, so a
naive restart would lose every in-flight workflow. These mechanisms make a root
restart safe:

- **Durable scheduler state.** Each task's mutable state (status, attempts,
  assigned worker, failed workers, merge linkage), its dependency edges, and its
  epoch index are persisted to Redis on every transition, along with per-workflow
  epoch ordering and frontier. On startup the server rebuilds the full task DAG,
  ready queue, and epoch frontiers from these records (`TaskRuntime.rehydrate`).
  A transition's task records, workflow status-set membership, and schedule
  snapshot are written as a single atomic Redis transaction
  (`WorkflowRegistry.commit_transition`), so a crash mid-persist commits the whole
  transition or none of it. Event-driven transitions are additionally healed by replay;
  the API-driven workflow cancel relies on this atomicity alone.
- **Replayable task events.** Task lifecycle events flow through a durable Redis
  stream consumed from a persisted cursor. The ordering is what makes replay
  safe: a transition is written to durable scheduler state *before* its event is
  emitted, and the consumer advances the cursor only *after* it has handled an
  entry. Delivery is therefore at-least-once — a crash between handling an entry
  and persisting the cursor simply replays that entry on the next startup.
  Handlers are idempotent (a terminal task ignores late dispatch / start /
  update events, and a repeated completion is dropped), so replay cannot
  double-apply. Completions that occur while the root is down are replayed on
  startup rather than dropped. In-flight tasks are left assigned to their worker
  — surviving workers' completions arrive via the stream, and workers that
  genuinely departed are reclaimed by the watchdog. A task still being cancelled
  has its worker interrupted again.
- **Heartbeat grace for rehydrated work.** Worker heartbeats are dropped while
  the root is down, so a surviving worker briefly looks stale once the root is
  back. The watchdog gives any worker that owns rehydrated in-flight tasks an
  extended grace (`WORKER_REHYDRATION_GRACE_SEC`, default 120s) before it may
  reclaim those tasks, so a worker that is merely catching up is not mistaken
  for a dead one and its tasks are not needlessly requeued.
- **Vaulted credentials.** A workflow's inline credentials live in its
  credential vault until the workflow settles, so a task dispatched after a
  restart receives them. Startup keeps every live workflow's vault and drops the
  vault of a workflow that settled or never finished registering. A task record
  stored with its credentials inline has them vaulted, and its source redacted, by
  the first start that loads it.
- **Resident replicas.** A warm resident replica whose serve task holds its dispatch
  is re-attached and reused; any other is invalidated and re-materialized on demand.
  A gated serve request in flight at the restart fails, as its client connection ends
  with the root, and frees its replica slot. See
  [`RESIDENT_CAPACITY.md`](RESIDENT_CAPACITY.md).
- **SSH forward ports.** A running SSH task's `forward` session is served again
  on the port it was published on. A session whose port another process took
  while the root was down is unreachable until its task ends.

Rehydration runs inside the ASGI lifespan **before it yields**, so the server
does not accept traffic (and its healthcheck does not pass) until scheduling
state is fully restored. Readiness is therefore implicit — no separate probe is
needed, and `stack restart`'s `--wait` blocks until the node is genuinely ready.

The result is a brief control-plane pause on the root (the server container
recreate plus rehydration) during which workers keep running their tasks; no
workflow is lost.

## Use case: rolling image updates

Because each node survives an in-place server restart, a cluster can be moved to
a new image one node at a time without a full teardown. The rollout itself is
driven externally — by an operator or a cluster-management tool — using the same
primitive with an explicit tag:

```bash
# On each node host, in turn — update the root node last:
flowmesh stack restart server --image-tag <new-version>
```

Recreate one node at a time, leaving the others serving, and update the **root
node last** so the control plane is the final hop. Each worker node's in-flight
tasks requeue while it is down and its workers re-register once it is back; the
root's durable state carries its in-flight workflows across its own restart.

## Constraints

- **Recreate only the `server` service on the root.** Leave `redis_control` and
  `redis_telemetry` running so durable state and the event stream survive.
  Updating the Redis image is a heavier, control-plane-wide outage and is out of
  scope for a brief in-place restart.
- **Co-located root workers are recreated.** Workers running on the root host
  die with the root's supervisor; their in-flight tasks requeue and re-run,
  except a v2 task that cannot safely re-run, which fails. To avoid this, prefer
  not to run workers on the root node.
- **The no-worker grace restarts on a root restart.** The window before a task
  that no worker can satisfy is failed (`TASK_NO_WORKER_GRACE_SEC`) is tracked
  with ephemeral scheduler state that is intentionally not persisted, so it
  starts fresh after a restart. This is deliberate: the restart is itself a
  disruption, and a fresh window avoids grace-failing a waiting task the instant
  the control plane comes back.

## State lifetime

Cluster state (workflows, durable scheduler records, the task-event stream)
lives in the two Redis instances, which snapshot to disk (`redis-server --save`)
on named Docker volumes — `<slug>_redis_control_data` and
`<slug>_redis_telemetry_data`. The state therefore follows the *volumes*, not
the container or the server process:

- `stack restart` and `stack down` recreate or stop containers **without**
  `-v`, so the volumes and the state persist; Redis saves on the SIGTERM
  from a graceful stop and reloads the snapshot on the next start. This is what
  lets a restart (or a plain `stack down` / `stack up`) resume in-flight work.
- `flowmesh stack clean` is the only command that wipes the state: it runs
  `down -v`, removing the volumes. (`stack image prune` / `stack image rm` only
  delete images; they do not touch the volumes.)

Persistence is snapshot-based (RDB), not write-synchronous, so a *graceful*
restart preserves everything, but an abrupt loss of a Redis container (kill,
OOM, host crash) loses the writes made since its last snapshot, which the stack
takes about every 60s for control and 300s for telemetry. Recovery requires the
control Redis to keep every write it acknowledged, so a deployment that must
survive an abrupt loss of it runs the control Redis with append-only persistence
that syncs every write (`appendonly yes`, `appendfsync always`).
