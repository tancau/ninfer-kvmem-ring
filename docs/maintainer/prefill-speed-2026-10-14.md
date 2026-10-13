# Prefill speed: the fast-prefill arena is the step (2026-10-14)

User-experienced speed, not throughput vanity. The three waits that actually exist on
this machine:

```
cold read   first time on a long context   258K -> 772 s      <- dominates
revisit     back to another long prefix    71.2 s -> 4.3 s    (SSD tier, measured)
decode      characters appearing           ~103 tok/s ~ 40 hanzi/s
                                           faster than reading, not a bottleneck
```

"Faster" therefore means prefill. Decode tuning has no experience value here and was
not pursued.

## Result

**The prefill curve steps at exactly `--kvmem-budget + --kvmem-gen-reserve`, and real
traffic sits just above the stock value of that sum.** Moving the budget from 36864 to
57344 (sum 53248 -> 73728) roughly doubles prefill for the requests that actually
arrive: **measured 382.6 -> 734.4 tok/s at 60K tokens, +92%**.

Shipped in `start_stack.ps1` and `Start-RC3.bat`. Rollback is one parameter (36864).

## How the step was found

The curve, production config, one engine per point:

```
16K   795.8     48K   740.6     96K   377.6     173K  351.5     258K  332.4
```

Rate drops 7% from 96K to 258K (351.5 -> 332.4) but halves from 48K to 96K
(740.6 -> 377.6). Two independent sessions reproduce both sides. Fine-grained points
around the transition:

```
50,182 tokens   728.7     fast
52,120 tokens   740.6     fast
59,872 tokens   385.2     slow   <-- step is between 52120 and 59872
63,748 tokens   382.6     slow
```

`--kvmem-budget 36864` + `--kvmem-gen-reserve 16384` = **53,248**, inside that window.
Discriminating test — budget 57344 makes the sum 73,728:

```
shape              budget 36864   budget 57344   change
59,872 tokens      385.2     ->   735.4          +91%
63,748 tokens      382.6     ->   726.8          +90%
79,286 tokens      slow      ->   369.9          still slow, as predicted
```

All three predictions held. Raising `--kvmem-gen-reserve` instead of the budget gives
the same answer (60K -> 715.9, 96K -> 353.8), so the threshold is **B + R as a total**,
not B alone. The 79,286 point also rules out sink/recent being part of the sum (if they
were, 79,286 would have been fast).

Independent confirmation from the regression log: every production instance reports
`occupancy.device_backend_kv_pages = 832`, and 832 x 64 = 53,248 = B + R. The arena is
that sum.

## Why it matters for this box specifically

```
DSH    57,756 tokens/turn   (measured)
ZCode  58,000-62,000        (measured)
stock arena: 53,248
```

Every production turn was paying the slow rate. This is not a synthetic benchmark win:
it is 385 vs 730 tok/s on the requests that actually arrive.

## Cost, measured

```
60K band        382.6 -> 734.4 tok/s      +92%
258K cold       772 s -> 826 s            -6.6%   (prefill 332.4 -> 310.6)
VRAM            10310 -> 10941 MiB        +631 MiB at 258K
recall 3/3      PARROT-5/15/25 exact      PASS
```

Arena ceiling: budget 73728 starts (arena 90,112, VRAM 11550); budget 81920 does not.
The refusal is explicit and is not VRAM exhaustion:

```
ERROR startup failed | finalizing target
FATAL server failed during startup | requested Engine runtime reservation requires
      4157653761 bytes, but only 3864002560 bytes are available for runtime capacity
```

A hard reservation shortfall of 294 MiB. Worth knowing, because the symptom looks like
an OOM.

Chosen value 57344: arena 73,728, which covers the measured workload with ~11.7K tokens
of margin, at 11011 MiB VRAM (1277 MiB headroom). If traffic moves above the arena, the
knob is `budget + gen-reserve` and the trade is +92% in-band against -6.6% above it.

## The pool hypothesis, and how it was wrong once

An earlier reading blamed the device pool alone and concluded "a bigger pool does not
help". That was measured at 96K, where **both** budgets are already above their own
arena, so the step could not be seen:

```
96K shape        budget 36864   prefill 377.6   VRAM 10310
                 budget 57344   prefill 349.9   VRAM 11011
                 budget 81920   refuses to start
```

Both readings are right about their own region. The corrected statement: a bigger arena
moves the step right and lifts everything below it, at the cost of ~7% above it. Only
the fine sweep around 50-64K tells them apart, and it is the reason the first conclusion
had to be retracted.

It also settles the transfer question: if bus traffic set the rate, the flat plateau
above the step would keep decaying with length. It does not (351.5 -> 332.4 from 96K to
258K, -6%).

## Flag arms ruled out (50,182-token shape, one engine per arm)

```
arm                      prefill   vs control   verdict
control                  728.7     -
device-auto              759.9     +4.3%        the only clean flag win, not stacked here
device-calibrate         732.4     +0.5%        noise
cublas                   711.8     -2.3%        NEGATIVE
cublas-chunk1024         754.3     +3.5%
cublas-auto-chunk1024    764.6     +4.9%        +0.6% over device-auto
chunk1024                745.0     +2.2%
```

Two corrections this table forced:

1. **The earlier "+5% from `--prefill-cublas`" was the chunk, not cublas.** The
   2026-10-13 sweep's cublas arm had also moved the chunk 256 -> 1024, and the chunk was
   doing the work. Isolated at chunk 256, cublas is negative.
2. **`--device-profile auto` is real (+4.3%) and free.** It is upstream's own default,
   which the recipe (and our production) overrides to `off`; shipping it is a revert,
   not a change.

`--device-profile auto` was deliberately **not** stacked on the arena change, so the
arena effect stays attributable on its own. It remains available as a separate
+4.3% if wanted.

`--prefill-cublas` carries a documented perplexity cost and would need the
greedy-equivalence gate; at +0.6% over `device-auto` it is not worth that gate.

## A script bug worth recording

`pool_sweep.py`'s first run hung for 40 minutes: the worker was given a
`subprocess.PIPE` for stderr and nothing drained it. The worker logs while loading, the
64 KB pipe buffer fills, and it blocks **before binding the port** — which presents as
"process alive, 9 GB resident, port dead", easily mistaken for OOM or a deadlock in the
engine. Fix: stderr to a file or `DEVNULL`, never an undrained pipe. Same family as the
earlier "never pipe a process that spawns grandchildren" note.
