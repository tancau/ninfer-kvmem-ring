# Optimization pre-research round 1 and priority call

Recorded 2026-10-13. Covers the five candidates in
`post-ring-optimization-candidates.md`. Everything here was established by reading the
engine source, reading the RC3 worker's own `--help` (425 lines, saved at
`rc3-worker-help.txt`), or from measurements already in the log. No GPU time was spent.

The headline: **three of the five turned out to be configuration, not engineering.**
The engine we now run in production already ships the mechanisms, some of them behind a
flag that is off by default.

## P1. Windowed MTP draft attention — a flag, and already the paper's design

```
--mtp-attention-window N   the MTP draft head attends to its first 64 keys and the
                           newest N before its query, not the whole history;
                           verification is unchanged (default 0: whole history)
```

That is the arXiv 2607.21535 design verbatim: a sliding window plus an attention sink of
64 keys, applied to the draft only, with verification untouched (so it is lossless by
construction). Upstream has already built it.

Source-side confirmation that the default is indeed "whole history": the draft's
envelopes are `{position+1, position+1}` and grow per draft step
(`src/models/qwen3_5/program/speculative/mtp.cpp`), and runtime agrees
(`/props` shows `mtp_resident_pages == resident_pages`, i.e. the MTP KV mirrors the full
context).

- Test: A/B `--mtp-attention-window 0` vs 4096 on a long-context decode. Verification
  unchanged means greedy output must be identical, which is the safety check the paper
  itself used.
- Cost: one restart + one decode measurement, ~15 min.
- Kill: decode unchanged outside noise, or output diverges.

## P2. Risk-controlled recall contract — documentation, and we hold the data

The paper's frame (arXiv 2609.27981) defines deployment risk as the frequency of
*request-level material degradation* versus full KV, with a tolerance and a confidence
requirement. Our engine's degradation mode is already measured:

```
recall boundary   facts inside the pool (anchor + tail)   exact          early 7/7
                  facts in deep-overflow mid-anchor       lost           0/3
                  (deep overflow = anchor+tail > pool; the oldest pages leave first)
```

So the honest contract is already stateable, and it is stronger than "no guarantee":
*exact within the pool, windowed beyond it, with the boundary at `anchor + tail ≤ pool`.*
What is missing for the paper's finite-sample guarantee is paired per-request runs
(same request, full-KV versus pooled) at scale; we have single-cell instances, not a
sample.

- Deliverable: a contract section in the ring brief plus a note on what a formal
  guarantee would require.
- Cost: hours, no GPU.

## P3. DASC (compress GDN state checkpoints) — knobs exist; the compression is engine work

The state-checkpoint subsystem is a first-class cost in this engine, and it is exposed:

```
--gdn-state-fp16              halve each host state image            (we already use it)
--host-state-slots N          pinned host checkpoint states          (default 8)
--device-state-slots N        checkpoint states on the device        (default = concurrency)
--suspend-snapshot pageable|pinned   where a suspended state waits
--use-alt-prefix-caching      pool KV blocks and state snapshots share a budget (default 8192)
--release-diverged-checkpoints       evict the private checkpoint the next prompt diverges from
```

DASC's contribution over this would be compressing the snapshots themselves (retention
horizons per head/channel). That is an engine change, not a flag. But the flags let us
**measure the state share** first: if `--host-state-slots 0` or `--use-alt-prefix-caching`
changes reuse behaviour materially, the state side is worth attacking; if nothing moves,
DASC would be optimising a small term.

- Test: replay the "short conversation, several turns" workload under
  `--host-state-slots 0|2|8` and `--use-alt-prefix-caching` on/off; read cache hits and
  the un-attributed time.
- Cost: several restarts, ~1 h.
- Kill: no measurable change in hit rate or wall time.

## P4. Delayed eligibility — write-up only

Already shipped as the skip-restore fix; the paper (arXiv 2610.03007) supplies the
mechanism name ("delayed causal observation"). One paragraph in the decode-fix section.
Cost: minutes.

## P5. Options that were open — now a flag sweep on the production binary

```
--device-profile auto|off|calibrate     default auto; OUR CONFIG SETS off
                                        (inherited from the package launcher's default)
                                        -> we are running with route tuning disabled
--prefill-cublas                        "hand wide prefill GEMMs to cuBLAS: a large prefill
                                        speedup for a small perplexity cost; wants a larger
                                        --prefill-chunk"  -> directly targets the 200-300 s
                                        cold prefills, but crosses the PPL gate
--ngram-draft-tokens 0..63              lets ngram's own contribution be measured on RC3,
                                        no need to revisit franken
--adaptive-mtp                          on/off A/B
--draft-tokens 1..15                    draft width sweep
--lookup-ngram N                       (same family)
no equivalent of TensorRT-LLM's use_relaxed_acceptance_for_thinking
--post-thinking*                        exists, but it is an answer-sampling preset
                                        (temperature 0.2), not a speed knob
```

`--device-profile off` is worth calling out: the package launcher defaults it to `off`,
we copied that, and the worker's own default is `auto`. So a documented automatic
tuning path is sitting unused.

## Priority call

Ordered by (expected gain x confidence) / cost. All of 1-5 are configuration changes on
the binary already in production, so none of them risks the shippable state.

| # | item | why this rank | cost | gate |
|---|---|---|---|---|
| 1 | `--mtp-attention-window` | the paper's own mechanism, upstream-built, lossless by construction, targets the decode path we already measured as host-dominated | ~15 min | greedy output identical |
| 2 | `--device-profile auto/calibrate` | free route tuning we have switched off; default is `auto` | ~15 min | decode/prefill improve |
| 3 | `--prefill-cublas` | documented "large prefill speedup" and the cold prefill is our worst user-visible cost (200-300 s) | ~1-2 h (needs PPL) | PPL within tolerance of 5.639521 |
| 4 | `--ngram-draft-tokens 0` | quantifies ngram's contribution; if ~0 we stop paying its CPU (the LRFU lesson) | ~20 min | decode unaffected |
| 5 | `--draft-tokens` / `--adaptive-mtp` sweep | small, bounded; closes the "locked on an old binary" question | ~40 min | decode improves |
| 6 | P2 risk contract (documentation) | our differentiator, no GPU, strengthens the shipping claim | hours | none |
| 7 | P3 state-slot / alt-caching sweep | measures whether the state side is worth DASC's engineering at all | ~1 h | any measurable change |
| 8 | P4 note | minutes | none | none |
| 9 | ternary weight repacking (BITCOS) | artifact + kernel work for maybe 0.6 GiB; not on the path | weeks | parked |

## Predicted outcomes for the five flag tests

Written before running any of them, so the results cannot be rationalised afterwards.
Confidence is stated; the failure mode is stated; the decision rule is fixed.

### 1. `--mtp-attention-window 4096`

- **Mechanism**: caps the draft's KV read at sink-64 + N instead of the whole context.
  Verify is untouched, so output must be bit-identical.
- **Prediction: decode +5% to +30%, most likely +10-20%** at 135-173 K. The paper's
  +28-44% is measured at 1 M; at a tenth of that the draft's read is a smaller share.
- **Confidence: medium.** The mechanism is real and upstream-built, but the magnitude
  depends on the draft's share of per-round cost, which we have not measured on this box.
- **Failure mode (the one that matters)**: if decode is *host-bound* rather than
  GPU-read-bound, cutting the draft's GPU read buys nothing. We have evidence the host
  side is heavy: during one decode the engine burned ~5 cores of user CPU. Item 1 and
  item 5 are both GPU-side and both conditional on decode being GPU-bound.
- **Decision rule**: accept only if decode improves outside noise AND greedy output is
  identical. Reject on divergence regardless of speed.

### 2. `--device-profile auto` (we currently run `off`)

- **Mechanism**: per-GPU route selection, calibrated on a device with no stored profile.
- **Prediction: 0% to +10%, most likely near zero.** The launcher recipe disables it,
  which usually means it was neutral or unstable on their hardware.
- **Confidence: low-medium** that it does anything; high confidence that it is the
  cheapest thing on the list to try.
- **Failure mode**: a calibration pass that picks a worse route on a WDDM card (this
  machine has already shown WDDM-specific surprises), i.e. a small regression.
- **Decision rule**: keep `auto` only if measurably better; otherwise revert to `off`.
  Also worth watching: whether `calibrate` finds a profile for the 3060 at all (ours is
  not in their validated set).

### 3. `--prefill-cublas` (with a larger `--prefill-chunk`)

- **Mechanism**: wide prefill GEMMs go to cuBLAS instead of the hand-written tile.
- **Prediction: prefill +20% to +50%**, with a small, real perplexity change.
- **Confidence: medium-high on speed** (this is the ordinary cuBLAS-vs-hand-tile gap),
  **low on the quality size** until measured.
- **Why it matters most of the five**: the cold prefill is our worst user-visible cost —
  200-300 s on a prefix-miss turn at 135 K, and the client kills streams past 300 s. A
  +30% prefill is 60-90 s off the worst case.
- **Failure mode**: it crosses our quality gate. Our project's line has been "recall
  exact, PPL within tolerance"; cuBLAS changes accumulation order, so PPL will move and
  the old 5.639521 figure becomes non-comparable. Recall markers should survive (they are
  robust to tiny numeric drift), but that must be measured, not assumed.
- **Decision rule**: accept only if PPL stays within a tolerance we declare *before*
  measuring, and recall stays exact. Otherwise it becomes a documented trade-off, not a
  default.

### 4. `--ngram-draft-tokens 0`

- **Mechanism**: removes copy-from-context drafting; its cost is a context-sized search
  per round, which is host CPU.
- **Prediction**: accept rate and decode *fall* on repetitive content (our synthetic
  shape is deliberately repetitive, so the fall will be large there) and barely move on
  real conversation traffic. CPU use should fall.
- **Confidence: medium** on direction, low on magnitude.
- **The real prize**: attribution. We have never explained the ~5 cores of user-mode CPU
  during decode. If turning ngram off removes them, that is the answer, and it changes
  what to optimise next. This is the "the LRFU lesson" again: a feature that measures
  ~0 in value but costs host CPU every round.
- **Failure mode**: nothing measurable either way, in which case the CPU cost is
  elsewhere (LRFU is already off) and the search continues.

### 5. `--draft-tokens` sweep and `--adaptive-mtp` on/off

- **Mechanism**: draft width; adaptive chooses 3..N from measured survival.
- **Prediction: no material gain.** The per-position acceptance decay we have measured
  ([280,169,124] at draft 3; [162,105,56,2,2,...] deeper) says the marginal draft is
  mostly wasted. Adaptive exists precisely to trim that.
- **Confidence: medium-high** that this is a no-op, based on the decay shape.
- **Failure mode**: none. This is housekeeping to close the "locked on an old binary's
  value" question, not a search for gains.
- **Decision rule**: keep whatever wins by >2%; otherwise leave the defaults alone.

### Overall

```
best case (all five land)   decode +10-30%, prefill +20-50%, plus attribution of the
                            decode-time CPU
worst case                  items 1/2/5 are no-ops because decode is host-bound;
                            item 3 is blocked by the quality gate; item 4 explains
                            the CPU and buys nothing directly
expected                    one real gain (prefill-cublas, gated), one likely small
                            gain (draft window), one attribution (ngram), two no-ops
```

**No 3x anywhere.** The 3x in this project came from fixing a defect (the self-feeding
restore loop), and this round has found no defect — it has found configuration.

### One prerequisite before the sweep

Because items 1 and 5 are GPU-side and the dominant measured costs today were host-side
(and un-attributed), the first thing to establish is **whether RC3's decode at long
context is GPU-bound or host-bound**. One cheap measurement: GPU utilisation and the
engine's CPU time during a steady long-context decode. If the GPU sits at ~100%, the
draft work is on the critical path and items 1/5 are worth testing. If the GPU idles and
the CPU is busy, the sweep should start with item 4 (the CPU attribution) instead.

That single measurement reorders the list, so it goes first.


- P1 was "the biggest unexamined lever, possibly engine surgery". It is a flag. Its
  open question (envelope semantics) no longer matters for the experiment, though it
  would matter if the flag underdelivers and we need to build our own variant.
- The remaining performance work is **a flag sweep plus one documentation task**, not a
  development programme. That is worth saying plainly: the engine's remaining headroom,
  as far as this round can see, is in its own configuration.
- Everything measured today says the same thing three times: the large costs are
  host-side and un-attributed (stall, prefix miss, CPU vision). The flag sweep should
  therefore judge arms on **wall time and cache behaviour**, not on the engine's own
  `timings` fields, which have been wrong in all three cases.
