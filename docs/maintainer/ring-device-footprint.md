# Ring mode: device-footprint assumptions (audit + fix plan)

Status: living document. Owner: tancau. Last updated 2026-09-29.

## Why this exists

`NINFER_KV_RING=1` lets a Device KV pool smaller than the logical context serve
long sessions: the ring keeps a resident window on the Device and demotes the
rest to Host. The rest of the engine, however, was written for the dense
configuration where **every mapped logical page is Device-resident**. Each place
that relies on that equality is a latent failure once the ring is active.

Observed failures (all from the same class):

| symptom | first seen | site |
|---|---|---|
| `backend KV materialization requested without an allocation` | dflash2 draft budget | admission bundle vs chunked map |
| `KV committed frontier is invalid` (frontier = committed + 1) | 94465 / 117761 | MTP bridge commits backend before mapping it |
| `materialized sequence does not match its active entitlement` | 8K repro | ownership accounting assumed all mapped pages resident |
| `Paged KV reservation invariant was violated` (batch 1766 pages) | capture publish on 96K pool | capture mapped the whole prompt at once |
| `logical KV page has no Device replica` | capture on 96K pool | batch mapping read a demoted predecessor's Device index |
| `Paged KV reservation invariant was violated` (restore) | reuse on 96K pool | checkpoint restore reserved Host pages without freeing Device room |

## The rule

Any operation that needs Device KV pages must first make room. The ring already
does this inside `ensure_sequence_kv_mapped` (demote, then map). Every other
materialization path must do the same, or the planner must refuse the plan.

## Fixes landed

- `d075e9b` cap the planned Device KV demand at the pool (ring mode); the
  address keeps its logical entitlement so the mapping can still cover the prompt.
- `cb0ce62` `physical_occupancy` counts only physically allocated pages; the
  logical reservation is a promise satisfied by demotion, not held memory.
- `a406806` drop the Device predecessor hint when the page was demoted (the hint
  is optional for the allocator).
- `02f2e81` demote inactive addresses before restoring a checkpoint to the Device.
- capture publish: map only up to the prefill cursor + one chunk in ring mode,
  instead of the whole prompt (`transactions/capture.cpp`).

## Remaining sites (batch fix targets)

Class C — assume Device residency (will throw when a mapped page is Host-only):

- `storage/kv_store.h` `commit_activation` loop: requires every mapped page
  Device-resident, but `prepare_activation` reserves only up to the activation
  frontier. Needs the restore to have covered the whole range, or a frontier-only
  requirement.
- `storage/kv_store.h` prefix-fork path (`prepare_prefix_fork`,
  `commit_prefix_fork`, staged tail release): requires an all-Device prefix.
- `storage/kv_store.h` snapshot path (`active_snapshot_shape`,
  `commit_active_snapshot`): requires all-Device; largely covered by the
  publish-decline gate in `transactions/capture.cpp`, but not if the gate is
  bypassed.
- `storage/kv_store.h` `physical_page()` public accessor: raw Device accessor.

Class B — need Device room but do not free it:

- `transactions/materialization.cpp` Host->Device checkpoint restore
  (fixed in `02f2e81`; the residual case is a capacity contradiction: the pages
  that must be demoted belong to the very checkpoint being restored).
- `decode.cpp` DFlash context append: direct `ensure_mapped_to_tokens` without a
  preceding demote.
- `program_impl.cpp` causal-scoring lane: direct `ensure_mapped_to_tokens`.
- `storage/kv_store.h` prefix-fork growth reservation.

## The open design question

Reusing a checkpoint whose Device pages were demoted needs `restore_pages` free
Device pages. In ring mode `physical_peak_fits` deliberately skips the KV
dimensions, so admission does not see that need and does not evict victims for
it. When the pool is already held by the previous session's resident window the
restore cannot be funded and the request fails.

Resolved for the direct case (`02f2e81`, `3470fed`, `bfbcc5b`): before restoring,
demote the inactive addresses, then this address's own tail beyond the reuse
frontier; re-count until the free room covers the restore, and keep 16 pages of
slack for the tail COW and the first mapping chunk. The production shape (96K
pool, 127K prompt, 33 tools) now completes three turns with `cache 65,536
(51.7%, long anchor)` on turns 2-3.

Remaining: admission still does not price the restore, so it never evicts
victims for it -- the materialization succeeds only because it demotes the tail
itself. A pathological shape (a short reuse whose restore cannot fit) would still
fail rather than fall back.

## Pool size caps reuse depth

An anchor is published only if its own range fits the pool (the scoped-publish
scan). With a 96K pool (1500 pages) the 98304 frontier (1536 pages) is declined,
so the deepest banked anchor is 65536: a 127K session reuses ~51% and still
spends ~9.5 minutes recomputing the tail. With a 144K pool the endpoint itself
fits, so the same session reuses the whole prefix (seconds).

Consequence: the ring buys correctness past the pool, but turn latency is set by
how much of the session the pool can hold. Size the pool to the working session,
not to the smallest value that avoids a crash.

