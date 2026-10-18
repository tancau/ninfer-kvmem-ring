# Decode-host attribution — Stage 0 report (2026-10-18)

Status: Stage 0 **complete**; Stage 0b **executed** (2026-10-19, exclusive
window) — the host line is **closed** on the evidence below; Stage 1 is parked.
Companion: `decode-host-attribution-plan.md`.

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
the plan asked to write down. → Stage 0b (below) then ran the live capture: the
5-core behaviour does not reproduce on this build, and the line closes there.

## Stage 0b — live capture: executed (2026-10-19, exclusive window)

Instance: production artifact + production recipe minus `--vision`, port 18210,
own logs (`stage0b-*`). Two cold requests; during both, a 0.5 s sampler recorded
nvidia-smi util/memory and the engine process CPU plus per-thread deltas
(`gpu_cpu_sample.py`), giving three independent views.

| capture | prompt tok | prefill | decode | s/round | host µs/round | devwait µs/round | GPU util | engine cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 42,909 | 737.5 tok/s | 1,193 tok @ 47.3 tok/s | 44.2 ms | 1,499 | 44,155 | 96.0–99.0% | 1.19–1.25 |
| C2 | 119,392 | 336.6 tok/s | 629 tok @ 42.7 tok/s | 50.6 ms | 1,753 | 50,493 | 98.8–98.9% | 1.20–1.21 |

(Util figures are per-phase means of a 0.5 s series; single 0-samples appear
inside both prefill phases and none in decode.) Prefill is equally device-bound.
The process has exactly one hot thread (tid 215992, ~1.2 cores) in every phase
sampled. C2 is over the device window (B = 57,344 < 119,392): the ring is
demoting/restoring throughout — the regime the 5-core observation came from —
and it still shows 1.2 cores.

**Verdict.** The 2026-10-13 "~5 cores of user CPU while the GPU idles" does
**not** reproduce on this build and recipe, at 43K or at 120K. Decode is
device-bound (99% util); process CPU (~1.2 cores) is consistent with the meters'
host share (1.5–1.8 ms/round ≈ 3–4%). The first decision branch fires: wall is
device-bound, host levers are bounded — **host line closed**, Stage 1 not
justified.

Caveats, stated for honesty: single-request captures, no vision, no concurrent
traffic. The old observation may have come from mixed prefill+decode traffic or
from the vision path; if it reappears, re-open with the same sampler.

## Stage 1 — parked (Stage 0b fired the "device-bound" branch)

Per-phase timers `retrieve / restore / demote / draft / verify`, submit-side and
completion-side stamps, emitted with the existing throughput records,
observation only. Stage 2 candidates stay as in the plan. Two updates: the "cap
the draft's KV read" candidate is **refused by KVMem by construction** and, per
T1, worth ≈0 anyway; and the draft-step cost itself is now measured (≈5.2 ms per
extra step, `p1-mtp-window-2026-10-18.md` §5) — a device-side number, not a host
phase.
