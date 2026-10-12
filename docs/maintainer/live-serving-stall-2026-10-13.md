# Live-serving stall: 90-second zero-token windows on RC3

Recorded 2026-10-13, from the RC3 engine serving a real coding-agent client on the
3060 12G. This was found by accident (the user asked why GPU utilisation keeps
dropping) and it is the first thing in this project that looks like a **live-usage
defect** rather than a benchmark-shape artefact.

## What the user sees

GPU utilisation falls from 100% to 20-30% "for a while", then comes back up.

## What is actually happening

Sampled at 200-500 ms, correlated with the engine's own per-request records and its
process CPU time. In a 500-second window of active use (LRFU on):

| | share of window | mean GPU | mean power |
|---|---|---|---|
| bins producing tokens | 47% | 89.3% | 154 W |
| **bins producing zero tokens** | **53%** | **30.6%** | **26 W** |

Longest zero-token runs: **95 s, 95 s, 60 s, 15 s**.

During those windows:

- the engine's scheduler reports `prefilling: 1, running: 1` while `computed_prefill`
  and `committed_decode` are both 0, so it claims to be working and produces nothing;
- the engine process burns **3-7 CPU cores** (9.7 s and 27.8 s of CPU consumed in
  consecutive ~2 s samples);
- `resident_pages` reads **0** at the start of a window and then climbs page by page
  (76, 176, 268, 352, ... ~800) once the GPU burst begins;
- the GPU sits at 18-31% and 19-26 W, i.e. idle.

Then it flips to 100% / 169-170 W for 60-95 s of real prefill, and the token counters
move at 2000-4600 tokens per 5 s.

## Per-request accounting (the same thing seen from the engine's own timers)

Live instance, prompt sizes and phases:

| id | prompt | ttft | accounted | uncounted | prefill | decode | host_submit | device_wait |
|---|---|---|---|---|---|---|---|---|
| 23 | 53,851 | 324.6 s | 44.9 s | **279.7 s** | 0.00 | 12.9 | 103.9 s | 136.6 s |
| 19 | 51,540 | 185.2 s | 9.2 s | **176.0 s** | 0.00 | 9.0 | 5.4 s | 83.5 s |
| 20 | 52,158 | 177.4 s | 4.1 s | **173.3 s** | 0.00 | 4.0 | 1.2 s | 84.4 s |
| 21 | 52,753 | 183.2 s | 12.7 s | **170.5 s** | 0.00 | 12.6 | 1.7 s | 95.2 s |
| 22 | 53,137 | 186.2 s | 37.6 s | **148.6 s** | 0.00 | 37.4 | 2.6 s | 120.2 s |

Two observations:

1. `prefill` is recorded as **0.00 s** while TTFT is ~180 s. The prefill demonstrably
   happened (the throughput records show it), so this is a **timer-attribution gap**,
   not an absence of work. Any diagnosis that trusts `timings_seconds.prefill` on a
   continuation will conclude the wrong thing.
2. The engine attributes most of the wall clock to `device_wait_exposed_seconds`
   (83-137 s) while the GPU is measurably idle at ~20 W. "Waiting on the device"
   without device work means waiting on something behind the device boundary.

## What it is not (each hypothesis tested and refuted)

| hypothesis | evidence |
|---|---|
| the desktop's own GPU use | the floor is ~25% / 19 W with the engine idle; but during the stall the engine holds 10.4 GiB and reports `running: 1`, and the process CPU is pegged, which the desktop cannot explain |
| prefill-chunk stall (chunk 256) | a 46 s trace during prefill averaged **98.8%** with a single 0.4 s dip; no 0.45 s sawtooth |
| the 5 s statistics write | no 5 s periodicity in the dip pattern |
| host RAM exhaustion / paging | 14.8 GB free, pagefile 32 GB with **0.1 GB used**, `Pages Input/sec` ≈ 0 |
| the client sending overlapping requests | scoped to the live instance: `max simultaneously open = 1`, `starts while another open = 0` |

Note on that last one: scoping matters. The request log at
`C:\Bonsai-App\logs\requests-8081.jsonl` is **shared by every engine that has run on
this port** (franken left ~101 MB of history in it before RC3 started, and it is
449 instances long). An unscoped read reported "69 requests simultaneously open" and
queue waits of up to 588 s — all of it franken's own old test traffic. Every analysis
must filter on `server_instance_id`.

## Prime suspect

`history_hits` stays **0** while `history_misses` climbs, i.e. the reuse cache never
hits, and the engine is configured with LRFU on. LRFU scores the current query against
the whole history to rank retrieval candidates; with no hits, that scoring is pure
overhead, and our own earlier A/B showed LRFU contributes ~0 to decode speed, prefill
speed and acceptance on this workload.

## Running experiment

`-NoLrfu` changes exactly one thing (its three CLI parameters are dropped; everything
else, including `--kvmem-score-agg`, is identical). Baseline and after are measured
with the same script (`deadzone_stats.py`) over equal-length GPU traces:

```
BEFORE (LRFU on) : zero-token bins 53%   longest runs 95/95/60/15 s   mean 30.6% / 26 W
AFTER  (-NoLrfu) : pending
```

Decision rule fixed before the run: dead time below ~15% means LRFU was the cause;
unchanged means the cost is host-side KV rebuild/transfer and the search continues
there.

## Artifacts

- `gpu-long.csv` (LRFU on, 1200 samples), `gpu-nolrfu.csv` (after)
- `deadzone_stats.py` — one number for both runs, with an explicit instance argument
- `deadzone_probe.py` — GPU + engine CPU + engine usage, per second
- `align_probe.py` — per-5 s alignment of GPU samples against engine reports
- `req_phases.py`, `stall_probe.py`, `live_probe.py`, `recover_probe.py`
