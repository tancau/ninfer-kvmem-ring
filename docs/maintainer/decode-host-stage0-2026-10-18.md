# Decode-host attribution — Stage 0 report (2026-10-18)

Status: Stage 0 **complete** (analysis only; no GPU, no source changes).
Stage 0b (live capture) and Stage 1 (per-phase timers) are specified below, not
started. Companion: `decode-host-attribution-plan.md`.

## What was measured

Tool: `stage0_probe.py` (temp dir) over `C:\Bonsai-App\logs\requests-8081.jsonl` —
46,880 records, 148 engine instances, 116 instances with completed requests.
Production candidates are instances whose `server_start` argv carries
`infer-kvmem-server.exe`, `--vision-residency resident` and `--kvmem-budget`:
**33 instances, 940 requests** (budgets 36864, later 57344). Raw dump:
`stage0-out.json`.

Per request: `engine_timing.decode.{rounds, host_exposed_seconds,
device_wait_exposed_seconds}`, `timings_seconds.decode`, `speculative.*`.
Per round, from the `throughput` records:
`host_work.{decode_host_microseconds_per_round,
decode_device_wait_microseconds_per_round}` (3,837 records carry values).

## Result 1 — the two decode timers are internally consistent

Across 934 requests with both fields, `(host_exposed + device_wait_exposed) /
decode` has median **1.036** (min 1.021, max 1.463; the tail is cold-start and
graph-capture windows). The per-window throughput values agree with the
per-request ratios — e.g. `serve-16872`: host 1,503–1,988 µs/round against
device wait 38,861–50,700 µs/round, window totals ≈ 0.18 s of host time per 5 s.

This is the opposite of the `timings_seconds` history that started this plan
(prefill=0.00 against 200 s, vision=0.0 against 308 s, a 300 s queue wait).
Those fields still cannot be trusted; **decode** now has a working split.

## Result 2 — the curve (940 requests, medians per prompt bucket)

| prompt tokens | n | tok/s | s/round | host µs/round | devwait µs/round | acc/round | rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–5k | 442 | 84.7 | 0.0386 | 1,558 | 38,106 | 2.26 | 96 |
| 5–20k | 44 | 87.2 | 0.0497 | 2,123 | 48,666 | 3.09 | 20 |
| 20–60k | 238 | 118.8 | 0.0638 | 2,738 | 62,931 | 6.62 | 19 |
| 60–130k | 126 | 52.7 | 0.0547 | 3,248 | 53,418 | 1.90 | 198 |
| 130k+ | 84 | 53.8 | 0.0534 | 3,414 | 51,364 | 2.02 | 99 |

How to read it:

- **Decode wall per round grows 38.6 → 53.4 ms** from short context to 130K+
  (+38%). That ~15 ms/round is the long-context tax this engine currently pays.
- tok/s is not comparable across buckets: the 20–60k bucket runs 118.8 tok/s at
  6.62 accepted tokens/round (ngram-rich traffic), 60–130k runs 52.7 at 1.90
  (synthesis-heavy traffic). s/round is the machine property; acc/round is the
  workload property.
- **Host-exposed work is 4–6% of the decode wall at every context size**:
  1.6 → 3.4 ms/round, a 2.2× growth, but still two orders of magnitude below
  the wall. Device wait is 94–96%.

## Result 3 — what the meters do not cover (the important part)

- The `host_work` details registered in the engine are exactly three host
  phases — `admission_policy`, `context_progress`, `stats_publication`. There is
  **no phase** for retrieval scoring, page restore/demote, or ring bookkeeping.
- The 2026-10-13 observation that long-context decode burns ~5 logical cores of
  user-mode CPU is neither confirmed nor denied by this data, and it is **not
  explained** by it. `decode_host_microseconds_per_round` measures one thread's
  exposed work (1.5–3.4 ms/round); 5 cores over a 50 ms round is ~250 ms of CPU
  per round. Whatever burns those cores lives in threads that do not report to
  this field.
- `device_wait` is the executor blocked on the device timeline. That includes
  real GPU compute **and** any starvation (host-fed transfers, page movement,
  throttling). From the log alone the two cannot be separated.

Answer to the Stage 0 question ("can `host_work` be trusted?"): **yes for the
executor thread, no as a process-level instrument.** The decode split is real and
consistent, but it demonstrably does not cover the threads behind the 5-core
observation, so it cannot close the attribution alone. This is the discrepancy
the plan asked to write down.

## Stage 0b — the live capture that closes the gap (needs an idle GPU, ~15 min)

Run an experiment instance with the production recipe (own port and logs;
production untouched), issue one long-context decode (173K anchor shape), and
during it sample:

1. `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 500`;
2. the engine process CPU per thread — process total plus per-thread user time
   (psutil or `Win32_PerfFormattedData_PerfProc_Thread`);
3. the engine's own `throughput` records for the same seconds.

Decision rule:

- GPU util high (>60%) → the wall is device-bound; host levers are bounded.
  Record it and stop the host line.
- GPU util low (<30%) with 2–3+ cores busy in *named engine threads* → the wall
  is host-fed; the thread names say whether it is drafting, tiering, or scoring,
  and Stage 1 timers target that thread.
- Either way, A/B `--kvmem-host-mib 12288` vs `4096` at the same shape: if
  per-round device wait moves, page movement is on the critical path.

Cost: 15 minutes inside the next exclusive GPU window.

## Stage 1 (unchanged, gated on 0b)

Per-phase timers `retrieve / restore / demote / draft / verify`, submit-side and
completion-side stamps, emitted with the existing throughput records,
observation only. Stage 2 candidates stay as in the plan. One update: the "cap
the draft's KV read" candidate is now known to be **refused by KVMem by
construction** (`p1-mtp-window-2026-10-18.md`), so that particular lever needs an
upstream change, not a flag.
