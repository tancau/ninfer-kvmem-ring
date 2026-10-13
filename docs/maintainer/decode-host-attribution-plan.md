# Instrumentation plan: attribute the host-side cost of decode rounds
#
# Status: plan, not started. 2026-10-13.
#
# Why this exists: three times today the dominant cost sat on the host and was not
# recorded anywhere (`prefill=0.00` against 200 s of first token; `vision=0.0` against
# 308 s; a queue wait of 300 s). Separately, during long-context decode the engine
# burns ~5 logical cores of user-mode CPU while the GPU idles — work that has never been
# attributed to anything. The flag sweep showed the GPU side is at the plateau, so if
# there is another step left, it is on the host side, and the blocker is not knowledge
# of what to fix but absence of any signal.
#
# Design principle for all three stages: DO NOT CHANGE BEHAVIOUR. Every stage adds
# observation only. Anybody looking at output quality, timing behaviour, or the API
# must be unable to tell the instrumentation is present.

## Stage 0 — no source changes at all (ETA: 1 day)

Goal: establish whether the `host_work` records the engine already emits can be
trusted for the host-side decode tax, the same way the timing fields cannot.

- Collect the `host_work` blocks from the request log across the long-context decode
  runs we already have, and check they agree with the external evidence:
  (a) GPU utilisation traces during the same windows, (b) the engine process CPU
  accounting, (c) per-round wall time derived from `ttft` + completion/latency.
- If the aggregate host_work numbers line up with the external measurements,
  Stage 1 becomes a matter of granularity, not of new machinery.
- If they disagree (as the `timings_seconds` fields did three times), write down the
  discrepancy precisely: which direction, by how much, and under which conditions.
  That report IS the Stage 1 specification.

Kill: if Stage 0 shows `host_work` already explains the per-round cost, there is no
missing signal to build — go straight to Stage 2 with that field as the instrument.

## Stage 1 — a per-phase decode timer, observation only (ETA: 2-3 days)

Goal: split one decode round into named host phases with wall times.

Candidate phases (names from today's measurements; refine once timers exist):

1. `retrieve` — query scoring against the whole history to rank retrieval candidates
   (the `-NoLrfu` experiment showed the stall shrinks but persists, so retrieval
   accounts for part of the cost but is not the whole story).
2. `restore` — host-to-device transfers of the pages the planner selected.
3. `demote` — device-to-host evictions to make room.
4. `draft` — ngram context matching per round (width up to 31) plus the MTP proposal
   forward pass. The draft's attention spans the full context; the per-round cost of
   that scan is the open question from today's 5-core observation.
5. `verify` — the full-attention verification pass (GPU; should be near-constant per
   round and is the reference the other phases are compared against).

Implementation constraints:

- Timer fields accumulate per round and are emitted with the existing throughput
  records; no new event types, no new files, no behaviour change.
- Must distinguish "this phase is work" from "this phase is waiting on the GPU":
  record submission-side and completion-side timestamps so that host slowness and
  device backpressure can be separated (today's signature was GPU idle + CPU busy,
  which is host slowness, but the two must not be conflated again).
- Must cover the queue path: the 300 s queue wait was invisible until the proxy log
  was read. Queue admission and lane occupancy belong in the same record.

Kill: if per-phase timers show the dominant phases are things we cannot change on this
hardware (e.g. PCIe transfer time of a fixed byte count at measured bandwidth), the
investigation ends with "the wall is physical" and no Stage 2.

## Stage 2 — targeted change, gated on Stage 1 naming a culprit (ETA: unknown)

Cannot be planned until Stage 1 names the phase. The candidates, in order of today's
evidence:

1. If `draft` dominates: cap the draft's KV read (e.g. sink-64 + window, per the
   Windowed-MTP design) or drop ngram drafting on this workload. Note the upstream
   binary documents `--mtp-attention-window` but its KVMem path refuses the flag, so
   this would mean an engine change or a different build, not a flag.
2. If `retrieve` dominates: the scoring runs against the whole history every round;
   candidate changes are caching the scores across rounds within a request, or
   restricting scoring to pages that changed.
3. If `restore`/`demote` dominate: batching, residency hints, or a different eviction
   order. This is the Phase 3 work that was parked earlier, and it stays parked until
   Stage 1 proves it is the cost.
4. If `verify` dominates: nothing left to win; the wall is the model's compute.

Each candidate ships behind its own gate: output must be bit-identical under greedy
(verification-path changes) or recall-exact (retrieval changes), and wall time must
move outside noise on the 173 K anchor shape.

## What this plan deliberately does NOT include

- No KV-format change (int8 vs nvfp4): orthogonal to attribution, and it would move
  both the measurement and the thing being measured.
- No drafting-model change, no new speculative backend: the measured acceptance decay
  says the marginal draft is mostly wasted, which is a reason to shrink drafting, not
  to add more of it.
- No re-tuning of pool/budget/chunk/draft: the flag sweep closed that space.
- No touching the client: the prefix-instability defect is ZCode's, and its report
  already exists. This plan is about the engine's cost when the client is well behaved.

## Success criteria for the whole plan

1. Stage 0 report: can `host_work` be trusted, yes or no, with the discrepancy stated.
2. Stage 1: a per-round breakdown where every second of wall time is accounted
   (today's best record leaves ~90 % of some requests unattributed).
3. Stage 2: either a change that moves long-context decode wall time measurably with
   quality intact, or a written "the wall is physical" with the numbers to prove it.
   Either outcome closes the investigation; "we tuned flags" does not.
