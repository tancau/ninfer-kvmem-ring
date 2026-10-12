# Flag sweep results (2026-10-13)

Run against the production binary with one flag changed per arm, each arm verified to
have actually reached the worker's argv. Workload: 46,234-token cold prompt then five
continuation turns, 900-token ceiling each. Judged on the engine's own per-request
`timings_seconds` (prefill seconds, decode seconds) and derived rates, never on wall
time, which is confounded by how long each answer happens to be.

## Results

```
arm                        prompt   prefill    pre_tps   dec_tps   comp
production (3 runs)        46234    57.8±0.1   800.1±1   52.7±2    2893-3958
--ngram-draft-tokens 0     46234    57.3       806.8     53.2      2758
  (no --adaptive-mtp)      46234    57.5       803.9     50.0      2654
--device-profile auto      46234    55.9       826.4     51.4      2567
--prefill-cublas (+1024)   46234    55.0       840.4     49.6      1172
--mtp-attention-window N   -        ENGINE REFUSES TO START (see below)
```

- **prefill**: the three production runs are tight (57.7 / 57.8 / 57.9 s), so the
  `device-profile auto` (-3.3 s) and `--prefill-cublas` (-4.8 s) differences are real.
  Both are small: +3.4 % and +5.0 % throughput.
- **decode**: every arm falls inside the production spread (50.1-54.3 tok/s). Nothing
  tested changes decode.
- `--ngram-draft-tokens 0` and dropping `--adaptive-mtp` change nothing measurable.
- `--prefill-cublas` carries a documented "small perplexity cost", so +5 % prefill is
  gated on a PPL measurement that has not been done; it is not a free win.

## P1 killed: `--mtp-attention-window` is unusable under KVMem

`--mtp-attention-window 4096` (the windowed-MTP design from arXiv 2607.21535) is
documented in the worker's `--help`, including the sink-64 + window wording and
"verification is unchanged". It does not start:

```
--mtp-attention-window 4096 / 64  x  with/without --adaptive-mtp  x  with/without
--ngram-draft-tokens 0   ->  4/4 REJECTED, exit 1, generic message:
    "ninfer-serve: KVMem requires ordinary/fixed/adaptive MTP1..15, ...,
     1..8 text or resident/CPU-vision lanes, INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 fast
     prefill, and exclusive ownership of history caching"
control: no window, ngram 0, no adaptive  ->  ACCEPTED
```

The message does not name the flag, so the isolation above is what establishes that the
window is the rejected item. This is the **third** case today where the worker's help
documents a flag the runtime refuses (after `--vision-residency overlay`, and a
parameter that silently failed to reach the argv).

## Measurement hazards hit while running this

Three things would have produced a wrong answer if left uncaught:

1. **Flags silently not reaching the worker.** `-NgramDrafts` and `-NoAdaptiveMtp` were
   declared as parameters but never wired into the argv, so the ngram arm ran the
   production configuration and reported a 165.8 s total that looked like a result.
   The per-arm argv check (`--ngram-draft-tokens 0 in argv: False`) is what caught it.
   From now on every sweep arm prints whether its flag is present in the worker's
   command line before it runs.
2. **A duplicated `--device-profile`**. The base argv bakes in `off` and the override
   appended `auto`, so the argv contained both. Last-wins made it work, but a first-wins
   parser would have made the arm a silent production run. Overrides should replace in
   place, not append.
3. **Wall time is not comparable across arms.** Answers differ in length because
   max_tokens is a ceiling, not a target; the fastest-looking arm had produced the
   fewest tokens. Every number in the table comes from the engine's per-request
   timings instead.

## Verdict against the pre-research predictions

Written before the run in `optimization-pre-research-2026-10-13.md`:

```
predicted                                        measured
1  mtp-attention-window  +5-30 % decode           dead: flag rejected (mechanism
                                                  unavailable on this build)
2  device-profile auto    0-10 %, likely ~0       +3.4 % prefill, decode unchanged
3  prefill-cublas         +20-50 % prefill         +5.0 % prefill (gated on PPL)
4  ngram-draft-tokens 0    attribution + small    no effect either way
5  draft/adaptive sweep   no material gain        confirmed: no effect
```

Three of five predictions were right in direction, and the biggest one (P1) was killed
by the binary rather than by measurement. The headline stands: the production
configuration is already at the plateau, and what remains is ~5 % prefill at a quality
cost, not a decode win.
