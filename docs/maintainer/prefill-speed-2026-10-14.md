# Prefill speed: what the levers are, and the one that failed (2026-10-14)

The goal here is user-experienced speed, not throughput vanity. Decomposed into the
three waits that actually exist when this machine is used:

```
cold read   first time on a long context   258K -> 772 s      <- dominates
revisit     back to another long prefix    71.2 s -> 4.3 s    (SSD tier, measured)
decode      characters appearing           ~103 tok/s ~ 40 hanzi/s
                                           faster than reading, not a bottleneck
```

So "faster" means prefill. Decode tuning has no experience value on this box and is
not pursued.

## The prefill curve is a step, not a slope

```
16K   795.8     48K   740.6     96K   377.6     173K  351.5     258K  332.4
```

Rate drops 7% from 96K to 258K (351.5 -> 332.4) but halves from 48K to 96K
(740.6 -> 377.6). Something non-linear happens between 52K and 99K tokens, after which
the rate is flat. Two readings were available:

- **transfer-bound**: above the resident set, KV must be fetched over the bus each
  step, at a roughly constant per-token cost -> flat plateau, and a bigger resident
  set should move the knee right.
- **compute-bound**: the plateau is the steady-state attention rate for long context.

## The pool hypothesis is REFUTED (measured 2026-10-14)

`--kvmem-budget` is the device-resident selected history, and production still carried
upstream's 36864. It had never been tuned on RC3. If the transfer reading were right,
raising it should have lifted prefill. It did the opposite:

```
96K shape, one fresh engine and one cold request per budget

budget 36864 (production)   prefill 377.6   decode 130.3   VRAM 10310 MiB   262.6 s
budget 57344                prefill 349.9   decode 119.8   VRAM 11011 MiB   283.2 s
budget 81920                engine exits 1 (does not fit)
budget 114688               engine exits 1 (does not fit)
```

- Raising the resident set by 20,480 tokens made prefill **7% slower**, not faster, and
  cost ~700 MiB of VRAM. This is the opposite sign of the transfer reading.
- The ceiling is between 57344 and 81920: at 57344 VRAM is already 11011 of 12288 MiB,
  so ~24K more tokens of KV (~0.96 GB) overflows.
- **This also refutes the transfer-bound reading in general.** If bus traffic set the
  rate, a larger resident set would have helped substantially; it did not. The flat
  plateau is the compute-bound attention rate.

So the 96K step is *not* "the resident set ran out". Cause still unknown. Locating it
(64K, 80K points) costs ~15 minutes and would turn the question from "where" into
"why"; it is not yet done and is not assumed.

## Levers that do exist, and their size

From the 2026-10-13 sweep, all on the 46K shape, all with `--fast-prefill-kernel`
already on (production):

```
arm                                    prefill      delta
production (3 runs, tight)             800.1
--device-profile auto                  826.4        +3.4%
--prefill-cublas + --prefill-chunk 1024 840.4       +5.0%
```

Two caveats carried forward:

1. The sweep's cublas arm also moved the chunk (256 -> 1024), so the two effects were
   confounded. `--prefill-cublas` at chunk 256 has never been measured.
2. `--prefill-cublas` has a documented small perplexity cost. Its gate was a PPL
   measurement that cannot be run (no evaluator accepts the artifact, RC3 ships no
   perplexity binary, and the API refuses `logprobs`). The substitute gate is the
   greedy-equivalence protocol in `post-ring-optimization-candidates.md`; a prefill
   kernel is exactly the kind of change it can catch, because a different accumulation
   order shifts hidden states.

`--device-profile calibrate` has never been run at all; only `auto` has.

## Arm matrix result (measured 2026-10-14, 50,182-token shape, one engine per arm)

```
arm                      prefill   vs control   decode     vram     verdict
control                  728.7     +0.0%        88.5       10395
device-auto              759.9     +4.3%        93.7       10215    <-- the only clean win
device-calibrate         732.4     +0.5%        88.2       10453    noise
cublas                   711.8     -2.3%        86.9       10634    NEGATIVE
cublas-chunk1024         754.3     +3.5%        87.2       10677
cublas-auto-chunk1024    764.6     +4.9%        93.1       10671    +0.6% over device-auto
chunk1024                745.0     +2.2%        86.8       10622
```

Three conclusions, all of them corrections:

1. **The earlier "+5% from `--prefill-cublas`" was the chunk, not cublas.** Run on the
   same shape with everything else equal, `--prefill-cublas` at chunk 256 is
   **negative** (-2.3%). The sweep's cublas arm had also moved the chunk 256 -> 1024,
   and the chunk was doing the work. This is the confound flagged above, now resolved.
2. **`--device-profile auto` is the only clean lever: +4.3%**, and it costs nothing
   numerically — upstream's own default for this flag is `auto`, and the recipe (and
   therefore our production) overrides it to `off`. Shipping `auto` is reverting to the
   upstream default, not introducing anything.
3. **`calibrate` is noise (+0.5%).** Chunk 1024 is worth +2.2% on its own but only
   +0.6% once `device-auto` is already on — it duplicates what the profile does.

So the whole stack is worth +4.9% while carrying cublas's documented perplexity cost
and a greedy-equivalence gate. **+0.6% over `device-auto` alone is not worth that
gate.** Decision: ship `--device-profile auto` alone; do not ship cublas.

## What this adds up to

```
lever                                  measured            status
prefix reuse (in-process)              772 s -> 2.7 s       286x   already shipping
SSD tier (session switching)           71.2 s -> 4.3 s      16.6x  proven, not shipped
--device-profile auto                  +4.3% prefill               this result
--prefill-cublas                       negative at chunk 256       DROPPED
--kvmem-budget above 36864             negative                    DROPPED
```

Honest framing: the large speed numbers in this project come from **not paying twice**
for the same tokens, not from making the first pass faster. Raw prefill is compute-bound
at ~350 tok/s in long context and single-digit-percent tuning is all that is left in
flags. The 96K step remains the only 2x-shaped unknown, and it is not the device pool.

## A script bug worth recording

`pool_sweep.py`'s first run hung for 40 minutes: the worker was given a
`subprocess.PIPE` for stderr and nothing drained it. The worker logs while loading, the
64 KB pipe buffer fills, and it blocks **before binding the port** — which presents as
"process alive, 9 GB resident, port dead", easily mistaken for OOM or a deadlock in
the engine. Fix: stderr to a file or `DEVNULL`, never an undrained pipe. Same family as
the earlier "never pipe a process that spawns grandchildren" note.
