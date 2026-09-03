# Resident-capacity control

Resident-capacity control serves a workflow's `resident` model bindings from reusable
physical capacity. An agent whose managed model resolves to a resident binding does not call
an external endpoint: the invocation is admitted to a compatible model-serving replica the
fabric materializes, sizes, and reclaims. Inference (vLLM) is the first resident family; the
GPU-free `dev_model` executor is a serving stand-in for the same admission path.

The subsystem is two actors over durable control-state (`CS`) stores. Actors do; stores
hold. It adds no worker-local service plane: a worker generically hosts a leased replica and
reports its endpoint and health, but owns no family directory, replica selection, scaling,
or routing.

## Actors

- **Admission controller** — the fast loop. It fair-orders claims from the DemandLedger,
  selects a compatible replica from the derived CapacityPools and the Replica directory,
  atomically records the fenced `ServiceClaim.RESERVED` credit, records an engine enqueue
  acknowledgement as `ACCEPTED`, and consumes a fenced terminal outcome from the
  orchestration ledger (`DS`) by `invocation_id` to release the credit. It writes only claim
  facts.
- **Lifecycle & scale manager** — the slower loop. It owns allocation leases and
  replica-directory lifecycle, performs policy-bounded, demand-driven scale-from-zero for
  approved plan-derived families, drains before an idle teardown, and reconciles preemption.
  Materializing and stopping a replica cross the flat worker plane by dispatching and
  cancelling a serve (or `dev_model`) task; an operator reads a resident replica's logs
  through the normal task-log path.

## Stores

Authoritative `CS` stores are durable and rehydrate on restart. Only `ServiceClaim` facts
are authoritative for admission credits; the two derived views recompute from them on every
read and cannot diverge.

| Store | Kind | Holds |
| --- | --- | --- |
| Service-family registry | Authoritative | Policy-approved, plan-derived family definitions. Registration materializes no capacity. |
| Replica directory | Authoritative | Live replica incarnations, endpoints, health, the incarnation fence, and a non-secret `ReplicaListenerAdvertisement` fenced by incarnation and listener generation. |
| DemandLedger | Read state | Unadmitted-claim references the controller fair-orders. Never promotes a claim itself. |
| Allocation leases | Authoritative | Lease records and lifecycle ownership per replica. |
| Invocation requests | Authoritative | The durable request record keyed by `invocation_id`, carrying the admission profile. |
| `ServiceClaim` facts | Authoritative | The per-admission credit and causal FSM. The sole authority for credits. |
| CapacityPools | Derived | Feasible replicas for a claim, net of every outstanding credit-bearing claim. |
| Admission-credit ledger | Derived | Per-replica held credit, recomputed from credit-bearing claims. |
| Capacity reports | Telemetry | Conservative normalized safe-capacity evidence, fenced by incarnation and report epoch. Never a credit authority. |

`DS` and `CS` link only through the stable `invocation_id` and fenced terminal outcomes;
neither side infers or overwrites the other's facts. The durable `DS` invocation holds the
identity and terminal state; the admission profile and the mid-claim credit state live only
in `CS`, so both persist to survive a restart.

## Claim admission FSM

A claim carries a fresh `claim_id` and admission epoch under its invocation's stable id.
`PENDING` and `TERMINAL` hold no credit; every other state is a credit-bearing nonterminal
fact.

```
PENDING → RESERVED → ACCEPTED → STREAMING → TERMINAL
PENDING / RESERVED → TERMINAL               (cancel, known enqueue failure, expiry)
RESERVED / ACCEPTED / STREAMING → UNCERTAIN → TERMINAL
TERMINAL --(permitted reissue)--> successor PENDING (same invocation_id, fresh epoch)
```

- **Fencing.** `RESERVED` stamps the selected replica's incarnation onto the claim, so a
  superseded incarnation cannot be mistaken for this credit. Concurrent reservations cannot
  overcommit a replica's reported safe slots.
- **Credit release.** A `TERMINAL` transition releases the derived credit. For an accepted
  or streaming claim the only normal release is a fenced terminal outcome recorded in `DS`
  and consumed by `invocation_id`; a stream close or a telemetry report alone never releases
  it. A pre-acceptance cancellation, known enqueue failure, or expiry records a terminal
  transition directly.
- **Loss and reissue.** A transient or ambiguous route loss moves a credit-bearing claim to
  `UNCERTAIN` and re-drives the boundary under the held credit — the runtime re-issues the
  same durable invocation, which resumes on the fenced replica — releasing nothing until a
  definite outcome. Only a completion, a fence rejection, or a clean engine refusal releases;
  a lost stream is held, never read as a completion. The hold is bounded by replica health,
  not a timer: a path that keeps failing preempts the replica after a bounded number of
  attempts, so the next resume finds no live incarnation and the fenced terminal releases.
  Once a claim is terminal, a permitted reissue raises a successor with a fresh admission
  epoch, never reopening a terminal claim or reusing its credit.

## Handoff and execution

A `RESERVED` claim authorizes one single-use, claim-bound admission handoff — the
pre-`ACCEPTED` bootstrap fence that binds the tenant subject, the fabric `idm-*` request
identity, the selected replica incarnation and listener generation, an expiry, and a
candidate route snapshot. It carries no raw engine endpoint or credential and is neither a
persisted control object nor a `RouteAuthorization`.

When the network plane is off, an in-server adapter consumes the handoff and relays the
request to the replica's OpenAI-compatible endpoint (read from the replica directory), the
claim-gated compatibility path. This path is single-shot: a post-acceptance ambiguous loss
settles the boundary, so its fenced terminal releases the credit rather than holding and
re-driving it as the native path does. That narrower ambiguous-loss window is a tracked
follow-up to bring onto the same hold-and-re-drive split.

When [`NETWORK_PLANE_ENABLED`](NETWORK_PLANE.md) is also on, the invocation is carried over
the native fabric path, whose live stream never crosses the server — though the deputy
returns the assembled completion to it to settle the invocation. The Lifecycle & scale
manager binds a per-replica **resident-facing sidecar** on the replica node and advertises
its non-secret listener; the sidecar is the enforced claim gate, validating every fence
against its own incarnation and listener generation before reaching the co-located engine.
Delivery is two-phase, server-driven over the node-command seam: a bootstrap poke delivers
the handoff over the origin node's deputy and obtains the engine enqueue acknowledgement, at
which point the Admission controller records `ACCEPTED` and issues the immutable
`RouteAuthorization`; a stream poke then carries the authorized response, with cancellation
and backpressure. The universal path is the reverse-rendezvous `control_relay`: the origin
deputy and the target sidecar each attach outward to the root, which bridges the framed
request and response between their per-node streams, so neither the origin nor the replica
node needs an inbound connection; the exact resident wire messages ride as opaque relay
payloads, so the sidecar and its claim gate serve them unchanged. The response is carried
under the relay's per-direction byte window, so a large completion is chunked and
flow-controlled rather than framed whole. A verified `worker_direct` or `node_relay` offload
may carry a reachable pair instead; an outbound-only fleet sets `RESIDENT_RELAY_ONLY` to
mandate the relay so no invocation attempts a forward-dial offload that would always fail. A pre-delivery offload connect
failure takes the already-resolved base candidate with no re-admission; a fence rejection or
a clean engine refusal is a definite release; a lost acknowledgement, ambiguous bootstrap,
or stream loss is `UNCERTAIN`, holds the credit, and re-drives until a definite outcome or
the fenced DS terminal, resuming from the durable relay cursor rather than re-running the
engine. The sidecar classifies a clean engine status as definite so no held slot leaks its
credit. A cancellation reaps both ends — the deputy pokes a cancel that closes the sidecar
connection and aborts the co-located engine request — so a cancelled invocation stops
promptly rather than waiting out the stream deadline. Request and stream emit claim-tagged
load evidence, tagged latency-sensitive service traffic versus bulk transfer.

The legacy serve proxy cannot reach a resident allocation: a resident replica's serve task
is marked resident and the proxy refuses it by allocation identity, independent of its
access mode. A resident allocation is reachable only through its claim-gated sidecar.

## Replica lifecycle and policy

```
ABSENT → MATERIALIZING → WARM ↔ BUSY
WARM / BUSY → DRAINING → STOPPED
WARM / BUSY → PREEMPTED / FAILED → reconcile and recreate
```

The first eligible `PENDING` claim for an approved family with no capacity triggers a
bounded zero-to-one materialization. Before creating an allocation, policy checks the allowed
model catalog, the per-family replica quota, and the concurrent cold-start limit; a refusal
is a typed durable outcome that creates no allocation, handoff, or credit and never
substitutes an external provider. A drain rejects new claims while admitted work reaches a
safe outcome. Idle teardown is off by default; when a retain window is configured, a
background sweep drains a servable replica that has held no credit past the window and stops
it once drained, and a later eligible claim materializes the family from zero again.

## Selection strategy

Replica selection is swappable and bound per approved family, not per workflow or request,
selected per deployment through `RESIDENT_SELECTION_STRATEGY`. The default
`batch-aware-best-fit` fills the feasible replica with the least remaining safe headroom
before spilling; `least-load` and `round-robin` are also available. The choice changes only
which feasible replica a claim reserves; the engine still owns continuous batching, token
scheduling, and KV allocation.

## Configuration

Resident-capacity control is off by default and enabled per deployment. See the `RESIDENT_*`
rows in [`ENV.md`](ENV.md) for enablement, the serving substrate (`serve` or `dev_model`),
the policy caps, the conservative admission-slot count, the cold-start budget, the per-family
selection strategy, the idle-teardown retain window (`RESIDENT_IDLE_RETAIN_SEC`, `0`
disables), and the relay-only mode (`RESIDENT_RELAY_ONLY`) that mandates the reverse-relay
for an outbound-only fleet.

## Observability

Read-only operator/admin visibility into resident capacity is exposed under `/api/v1/resident`,
gated to a SYSTEM/ADMIN principal. The endpoints return empty results when resident capacity
is disabled.

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/v1/resident/families` | Registered service families (family, engine/batch key, model ref, isolation, selection strategy, warmth). |
| GET | `/api/v1/resident/replicas` | Replica incarnations — live and inert — with state, health, backing `serve_task_id`, worker, lease, and endpoint host and port. Filterable by `family`. |
| GET | `/api/v1/resident/claims` | Credit-bearing admission claims and per-replica held credit, recomputed on read from the authoritative claims. |

A replica's endpoint is projected to host and port only; no `api_key` or serving credential
is ever returned.

To read a replica's serving logs, list replicas, take a replica's `serve_task_id`, and read
that task through the normal task-log path, `GET /api/v1/tasks/{serve_task_id}/logs` — resident
serve tasks are owned by the resolved system principal, so an operator reads them like any
other task.

Set `RESIDENT_SERVE_TTL_SEC` so a replica's backing serve task self-expires, bounding an
orphaned serve task if a teardown is missed.
