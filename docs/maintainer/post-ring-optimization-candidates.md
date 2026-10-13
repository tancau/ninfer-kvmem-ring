# Post-ring optimization candidates (test plan)

Recorded 2026-10-12. Everything here is a **candidate with a test attached**, not a
commitment. Each entry says what it is, why it might apply to us specifically,
what to measure, what it costs, and what would kill it.

Context that produced this list: the RC3 comparison (below) showed the gap is
architectural, not tuning, and two literature sweeps turned up mechanisms that map
onto the ring's own pathologies. Nothing here is started.

## Baseline to beat (measured 2026-10-12, 173K shape, server-reported timings)

| | ship (franken, pool 98304) | RC3 default recipe (`--lookup-ngram` n/a) |
|---|---|---|
| prefill | 109.6 tok/s | 285.3 tok/s |
| decode | 24.65 tok/s | 41.02 tok/s |
| draft accept | 44.1% (mtp, draft 3) | 35.1% (adaptive mtp4 + ngram31) |
| early-marker recall | 3/3 | 3/3 |
| prefix reuse (continuation) | 64K partial hit | 99.93% |

RC3 attribution arm (`-NoLrfu -ScoreAggregation max`, same shape): prefill 288.9,
decode 42.92, accept 36.6%. **LRFU contributes ~0** (the deltas are inside noise).
The 2.6x prefill / 1.66x decode gap is the retrieval architecture + recipe
(chunk 256, B/R = 36864/16384, dual INT8 KV, adaptive MTP4, ngram31), not history
weighting. RC3's own release notes agree: their LRFU is retrieval TopK ranking only
— no decode-time score accumulation, no adaptive budget, no quota mixing.

## P1. Windowed MTP draft attention (the biggest unexamined lever)

Source: *Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token
Context*, arXiv 2607.21535.

Claim: at long context the built-in MTP draft head runs **full attention over the
entire KV at every draft step**, so the draft read grows linearly with context and
comes to dominate the draft cost — precisely where speculation is most valuable.
Measured at 1M: the draft phase alone adds +92% to +138% on top of the bare verify.
The fix is a StreamingLLM-style sliding window + attention sink applied to the
**draft's attention only**; verification is untouched, so it stays lossless
(the full-attention target still decides every accepted token). At 1M: +28% to +44%
on per-decode-step cost, input-invariant, growing with context. It also notes the
draft KV is a measured 7.7-11% of total KV at 1M and can be reclaimed as a compact
ring buffer at zero quality cost.

Why it might be ours: the strong-anchor decode pathology we already fixed once
(7.5 -> 35 tok/s, the self-feeding demote/restore loop) is a draft/verify read-tax
symptom. Our MTP head has never been checked for whether it attends the full anchor.
If it does, this is a second decode lever of the same order as the first fix, and it
is training-free and verify-preserving. The "compact ring buffer" language is our
own territory.

Test 1 (zero GPU): read the MTP draft path and answer — does the draft attend the
full KV or a bounded window? Look for the draft attention call site and its KV
length. This is a code question; answer it before spending GPU time.

Test 2 (if the draft is full-context): window the draft's attention at W + sink,
measure decode on the strong-anchor shape against the 24.65 baseline, and confirm
byte-identical output under greedy (the paper's own losslessness check).

Kill if: the draft is already windowed (then there is nothing to window), or
greedy output diverges, or decode does not move outside noise.

Cost: Test 1 is free. Test 2 depends on where the draft attention lives; if it is
not reachable without touching verification, this becomes engine surgery and drops
to P3.

### Pre-research result (2026-10-13, source reading — no GPU spent)

**The claim holds in our engine too, and the mechanism for a fix already exists.**

1. The draft's attention calls pass the full context as the visible extent.
   `src/models/qwen3_5/program/speculative/mtp.cpp`:
   ```cpp
   const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
   const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
   ...
   for (int i = 1; i < mtp_proposal_extent; ++i) {
       const auto visible = static_cast<std::uint32_t>(position + i + 1);
       const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
       card.mtp_forward_ar_step(..., envelope, ...);
   }
   ```
   Every draft step's envelope grows with `position`, i.e. the draft attends the whole
   prefix, exactly as the paper describes.

2. Runtime corroborates it: `/props` reports `mtp_resident_pages` equal to
   `resident_pages` (624 / 624 in one sample), so the MTP KV mirrors the full context
   rather than a window.

3. The engine already contains a **page-level attention mask** mechanism:
   `src/ops/softmax_attention/dense/causal_cache/launch.h` carries a device bitmap over
   64-token KV pages (`NINFER_PROTO_SELECT`, modes `all` / `even` / `first_half` /
   `off_1_in_4`), documented in the source as throwaway scaffolding whose "real port
   replaces this with a per-request selection produced by the retrieval policy".
   Dropping pages from attention is therefore a solved problem in this codebase; the
   mask just is not per-call yet.

4. To be confirmed before writing any code: the exact semantics of
   `CausalAttentionExecutionEnvelope {min_visible_keys, max_visible_keys}`. The kernel
   filters *windows* by that range (`small_t.cu`: `if (window < min || window > max)
   return;` then `include(min); include(max);`), and the planner uses
   `max_visible_keys` against `prompt_visible_limit` to choose an attention route
   (`graph_profiles.cpp`). Both uses are consistent with "the envelope states the valid
   *length* of the visible sequence", which would **not** express a start offset — in
   which case windowing the draft must go through the page mask rather than the
   envelope. This is the one open question, and it is answerable by reading, not by
   measuring.

### Revised verdict

Still the best remaining lever, and now a **bounded** change rather than an unknown:
the draft reads full context (confirmed twice), the mask mechanism exists, and the open
question is a semantics question with a definite answer in the source. The original Test
1 (does the draft attend the full KV?) is answered; the next step is Test 1b — determine
whether a per-call mask can be threaded into the draft's attention ops, or whether the
envelope is the right lever after all.

Kill criteria unchanged: if the per-call mask cannot be threaded without touching the
verification path, this becomes engine surgery and drops to parked.

## P2. Risk-controlled recall contract (framing, no code)

Source: *Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets*,
arXiv 2609.27981.

Claim: average quality-memory trade-offs hide requests whose utility degrades
materially. Reformulate eviction as a deployment risk-control problem — a target
risk level (frequency of material degradation relative to full-KV on the same
request) plus a confidence requirement — and select a retention policy from
calibration data with a finite-sample guarantee, falling back to full KV when no
compressed policy is certified. Notable in their results: the same contract
certifies different aggressiveness for different benchmarks, and an empirically
acceptable policy can still fail finite-sample certification.

Why it might be ours: we already run acceptance lines (survival, decode >= 20,
tail >= 85, markers exact) but they are engineering intuition. This gives a formal
frame and, more usefully, an honest way to state the ring's contract: pool-internal
is exact, pool-external is windowed, and here is the measured degradation risk.
That is a stronger claim than any competitor currently makes, and it is
documentation, not a change to the engine.

Test: restate the existing acceptance matrix and the hole-region results as a risk
statement. Deliverable is a section in the ship report, not a code change.

Kill if: we cannot produce paired full-KV/compressed runs on the same request
(that is what the risk definition needs). Note we can: the acceptance matrix
already pairs them.

Cost: hours. No GPU.

## P3. DASC: compress GDN state checkpoints (ours to lose)

Source: *DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving*,
arXiv 2608.30386.

Claim: hybrid linear-attention models update recurrent state in place, so prefix
reuse requires **state checkpoints alongside full-attention KV**; storing those
checkpoints in full increases memory pressure, which causes more evictions and more
repeated prefill. DASC observes that GDN/KDA heads and channels retain prefix
information over very different timescales ("retention horizons"), derives those
horizons from model weights, selects long-horizon state units, and packs them into
a ragged checkpoint layout. Conservative configurations compress KDA checkpoints
2.63x while staying close to full caching; at fixed checkpoint memory budget, mean
TTFT falls 42.6% and input throughput rises 68.4%. It states the Qwen/GDN case
follows the same quality-efficiency trend, extending from channel-wise KDA to
head-wise GDN.

Why it might be ours: this is the ring's problem statement written by someone else.
We run Qwen3.5 with GDN, we store host-backed anchors, and we just measured that
RC3 reuses a continuation at 99.93% while we reuse 64K partially. The most likely
explanation for that gap is the state-checkpoint side of prefix reuse, which is
exactly what DASC attacks. Two possible readings, both worth testing: (a) DASC
explains the RC3 gap and we could close it, (b) DASC shows our rate of repeated
prefill can be cut 2.6x, which is worth real wall-clock on 262K first turns.

Test: instrument what a continuation actually has to restore in our engine (state
vs full-attention KV), and measure the repeated-prefill fraction on the 173K
continuation case. Then decide whether a retention-horizon split is implementable
in `context.cpp`'s anchor path.

Kill if: our anchors are already state-light (i.e. the gap has a different cause),
in which case the search moves to whatever the instrumentation shows.

Cost: instrumentation is moderate; an actual implementation is engine surgery and
stays parked until the instrumentation justifies it.

## P4. Delayed eligibility for recently written state (already shipped, needs writing up)

Source: *AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning*, arXiv
2610.03007.

Claim: two effects are missed by routing-only eviction — low-attention entries can
carry large value payloads whose removal changes future predictions, and newly
generated states can appear stale before later queries have had a chance to read
them. AvoKV-E separates *when* an entry may be judged from *how* it is ranked:
"delayed causal observation" protects recent entries until later queries have had a
causal opportunity to read them.

Why it might be ours: this is the theory behind the skip-restore fix we already
shipped. We stopped the decode path from re-fetching an anchor it was about to have
eaten again; the justification we had was four reproducible runs, not a mechanism.
This gives the mechanism. The value is in the write-up (and in being able to say
why, not just that it worked), plus the possibility that delayed eligibility should
also apply to the prefill path.

Test: none needed. Deliverable is a citation and two paragraphs in the decode-fix
section of the ring brief.

Cost: minutes.

## P5. Options still open from earlier

- `--lookup-ngram N` on franken: scaffolded (`ngram_supervise.ps1`, `ngram_ab.py`,
  `ngram_ab.ps1`) and never run to completion — the RC3 comparison superseded it as
  "just a flag", but the A/B machinery is ready and the flag is in the deployed
  binary. Note RC3 ships ngram at 31 by default, so this is no longer a differentiator
  if we migrate.
- Draft-token count by device: Unispec (ACL 2026) notes the draft length that is
  optimal on a datacenter GPU can degrade a consumer card (they name the RTX 3090).
  We locked draft 3 on an older binary. RC3 ships "adaptive MTP4". Re-measure 3 vs
  4 vs 5 on this card before assuming either.
- Relaxed acceptance during the thinking phase: TensorRT-LLM's MTP config exposes
  `use_relaxed_acceptance_for_thinking` (relax the accept test, biased by top-k and
  a delta, while thinking). Our thinking share is large (per-round: 350-730 of
  470-1000 completion tokens). Cheap to check whether NInfer's MTP path has an
  equivalent knob.
- Ternary weight repacking: BITCOS (arXiv 2609.16338) measures symbol distributions
  across 29 ternary checkpoints, Bonsai among them, and packs ternary weights at
  2 - z bits by zero density (up to 1.28x GEMV on CPU, 1.27x on Xe2). If Bonsai's
  zero density is near what the paper reports for ternary models generally
  (~0.4-0.5), our 5.54 GiB PTQ1 artifact could shrink, which converts directly into
  pool headroom. But it needs an artifact repack plus a matching kernel, and it is
  CPU/Intel-GPU oriented in the paper. Parked as a far-term VRAM idea.
- What is deliberately not on this list: learned eviction (LKV, DBTrimKV,
  SideQuest), hybrid upcycling (HyLo). All need training or an
  attention-semantics change, and none of them is where our measured gap is.
  (ActKV was on this list; it moved out on 2026-10-13 — see P6.)

## Literature watch 2026-10-13 (no GPU spent)

Second sweep, same verdict as the first: nothing here reopens the frontier, but two
items are worth filing and one is a possible future lever.

1. **vLLM PR #54637 — automatic prefix caching for GDN hybrids under MTP speculative
   decoding.** Per-block SSM-state checkpointing during prefill plus dual read/write
   anchors so decode and spec-decode reuse cached states. Two things matter for us:
   (a) it is independent confirmation that state checkpoints are the price of prefix
   reuse on GDN hybrids (the P3/DASC direction); (b) it names a bug class to watch
   for — *a draft rejection crossing a state-block boundary silently reads a stale
   checkpoint*. We cannot use the code (different engine), but if our decode ever
   shows acceptance anomalies correlated with block boundaries, this is the first
   place to look.

2. **SGLang issue #36935 — branch reuse silently collapses when the recurrent-state
   checkpoint is gone, even with KV 100% resident**, and only the last node's state
   gets MRU-refreshed. Same theme from a second engine: on hybrids, KV residency is
   necessary but not sufficient for reuse; the state checkpoint is the binding
   constraint. Filed as supporting evidence for the P3 ordering (measure the state
   share before optimising anything else).

3. **SoK "The KV Cache Is the New Memory Wall" (arXiv 2609.30854).** Survey with design
   rules per hardware/context/quality. Useful framing reference for the docs; confirms
   our operating regime (KV-bound at 262 K on 12 G) is the expected one, not an
   anomaly of our setup.

4. **Lever under verification: 4-bit KV (rk4v4-e8).** Researched 2026-10-13, no GPU spent.

   The `rk*` family in our worker's `--kv-dtype` list is confirmed 4-bit KV, not an
   exotic experimental flag (sources: KobusG/ninfer-engine `sm120a-quantized-kv-vision`
   branch docs, UDPSendToFailed/ninfer-4090, sergiuszm/ninfer-4090, natpate/ninfer-windows
   0.4.0 which merged all four as "live compressed-KV cache quantization"):

   ```
   rk8v4     rotated INT8 keys + packed 4-bit values
   rk4v4     rotated packed 4-bit keys and values (Hadamard, cosine 97.8% vs FP32)
   rk4v4-e8  E8-lattice 4-bit keys + packed 4-bit values (cosine 98.7%)
   rk2v4-e8  2-bit E8 keys + 4-bit values (cosine 96.2%; one fork declined to ship
             it "on evidence", others run it in production profiles)
   ```

   Measured on 4090/5090-class cards with Qwen3.8-27B (int8 baseline 35,219 B/token):
   rk4v4-e8 costs **17,979 B/token (51%)**, decode at parity (59.68 vs 60.03 tok/s,
   inside run variance), prefill **-20%** (pack/unpack on the write path), retrieval
   5/5 validated, and our own worker's help claims "verified history reuse" for all
   nine KV formats. Our help also states fast-prefill supports the rk family, so the
   +12.5% stacks instead of conflicting.

   Why it matters for this card: halving bytes/token doubles the effective retrieval
   budget at fixed VRAM — 36864 to roughly 73 K. That directly attacks the recall
   boundary (`anchor + tail <= pool`): the deep-overflow hole (mid-anchor 0/3) exists
   because the pool is smaller than the history, and a 2x pool is the first thing
   that has ever promised to close it rather than work around it.

   Open questions, in order:
   a. Do the rk kernels actually run on sm_86 (3060)? The published validations are
      sm_89/sm_120a. Our binary lists the formats, which is necessary but not
      sufficient — a startup + one short request answers it in minutes.
   b. Does 4-bit KV on *ternary* weights preserve exact recall? All published cosine
      numbers are on dense Qwen3.8; stacking 4-bit KV under 1.58-bit weights is
      untested anywhere visible. Our NIAH gate (3 markers, exact quote) is the right
      test, plus the PPL bar.
   c. Only if (a) and (b) pass: re-run the hole probes at 2x budget and see whether
      the mid-anchor region becomes quotable.

## rk4v4-e8 verdict on sm_86 (measured 2026-10-13, same card, same artifact)

(a) Starts and serves: yes. Ready in 6 s, VRAM 9418 MiB (vs ~10.3 GiB int8),
short request correct to the byte modulo one dropped period (sampling, not recall).

(b) Quality gate on the 173 K shape:

```
                       int8 (production)    rk4v4-e8
T1 prefill             285.3 tok/s          358.1 tok/s
T1 decode              41.02 tok/s          57.70 tok/s
draft accept           35.1% (426/1212)     57.0% (146/256)
early recall           3/3 exact            3/3 exact
reuse (continuation)   99.93%               99.93% (179935 / 180052)
```

(T1 trajectories differ by sampling, so decode/accept are directional, not exact.
Prefill is trajectory-independent. VRAM saving ~0.9 GiB is arithmetic.)

(c) The hole question answers itself: mid-anchor markers (PARROT-2145/2341/2538 at
~45/52/59 K tokens) are quotable **at the current 36 K budget, 3/3 exact in 9 s**
— the same markers that are 0/3 on franken's 96 K oldest-first pool. The hole was a
property of the eviction policy, not of the budget size: retrieval TopK fetches the
asked-for blocks regardless of position, while oldest-first had already evicted them.
No 2x-budget experiment is needed because there is no hole to close on this engine.

Caveat that still stands: the first hole probe returned 0/3 for a probe-harness
reason, not an engine reason — `max_tokens=600` was consumed entirely by thinking
(600/600 reasoning, empty answer), the same truncation seen before. Re-ran at 2000:
thinking 448 + answer 152 chars, `finish=stop`, 3/3. Probe harnesses on this engine
must budget output for thinking first.

Still open before any production switch: the PPL gate (dense-run perplexity vs the
5.639521 bar). Recall-exact is necessary but not sufficient; a 4-bit KV under
1.58-bit weights needs the distribution-level check. That test is queued behind GPU
availability, not behind any technical question.

## Greedy-equivalence results (measured 2026-10-13, replaces the PPL gate)

The PPL gate as defined turned out to be un-runnable: franken's offline evaluator
predates the rk formats (`--kv-dtype bf16|int8|fp8|rk8v4|nvfp4|k8v4`, no rk4v4-e8),
RC3 ships no perplexity binary, and the serving API refuses `logprobs=true`. The
substitute used here is stricter at the decision level: identical prompts, temperature
0, fixed seed, both engines, outputs compared token-for-token (the "greedy-exact"
standard from the Windowed-MTP paper). Control first: **int8 vs int8 = 7/8 identical**
(the one difference is a single word, different/distinct), so the engine is very nearly
deterministic and any wider divergence is attributable to the KV format.

```
format        identical/8   nature of divergences
int8 (ctrl)   7/8           one word: different/distinct (engine noise floor)
rk8v4         5/8           paraphrases; one factual fork ("different from modern
                            steel" vs "a type of steel") on open-ended prose
rk4v4-e8      3/8           paraphrases throughout open-ended prose; one content
                            selection change (which Eiffel facts to list)
```

Precision-shaped tasks (code, exact long-context quote, exact JSON format) are
**identical under all three formats**. Every divergence is in open-ended prose, and
none is garbling or recall failure — but the negation case shows the shift reaches
fact selection, not just wording.

Verdict: **neither 4-bit option passes strict greedy-equivalence; stay on int8 for
production.** The recall gate (3/3 exact, including the hole region) and the speed
numbers for rk4v4-e8 remain true, so 4-bit stays a documented option with measured
tradeoffs — the trigger that would reopen it is VRAM pressure (e.g. a larger retrieval
budget that does not fit in int8), not speed. rk8v4 is the closer of the two if that
day comes (5/8 vs 3/8, 8-bit keys intact).

   Update 2026-10-13 (literature, no GPU): quality evidence now exists for **our exact
   artifact** (`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`) under 4-bit KV, from
   wongkiller/ninfer-all `docs/performance/reference-2026-09.md` and KasoLu/ninfer-all:

   ```
   RTX PRO 6000 + Bonsai 27B + rk4v4    1,048,576 tokens  2/3 needles  55.3 tok/s
   RTX 5090     + Bonsai 27B + rk4v4    1,048,576 tokens  2/3 needles  56.2 tok/s
   RTX 5090     + Bonsai 27B + rk8v4      978,944 tokens  3/3 needles  46.9 tok/s
   (KasoLu recipe: --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4
    --gdn-state-fp16 --rope-yarn, "largest context a 24 GB card holds")
   ```

   Two consequences. First, the risk ordering changes: **rk8v4 goes first**, not
   rk4v4-e8 — it keeps 8-bit keys (cosine 99.4% vs FP32), scores 3/3 where rk4v4
   scores 2/3 on the same model, and is the smallest step away from our int8.
   Second, upstream master (`Neroued/ninfer --kv-dtype`) has **no** rk modes at all
   (bf16|int8|fp8|nvfp4|k8v4 only) — they are a 4090-fork feature that KVMem RC3
   carries. So this lever exists only because we migrated; on franken it never existed.

Not applicable (need training or a different engine): CompressKV, ResidualKV/DeltaKV,
FG²-GDN, hybrid upcycling, learned eviction. Background only: massive-activation
studies (quantisation relevance), the online-compaction empirical study (TE robustness
is consistent with our oldest-first experience, but offers no lever).

## P6. ActKV-style action-region eviction on top of ring KV (queued 2026-10-13)

Filed because a competing stack was published on the same day with the same base
(KVMem) and a different answer at the KV layer. This is the comparison that made
the idea concrete, so it is recorded here rather than in the literature watch.

**What they run:** Strata + KVMem + ActKV, claimed on 3060Ti 8GiB + 32 GiB host,
advertising 128K/256K/1M. "Speed kept, quality kept, capacity kept." No model named
in the title or description; no method, no numbers beyond the title.

**Strata, verified 2026-10-13** (`github.com/Niko1221/Strata`, MIT, 23.5k stars,
2.3k forks, 1892 commits, has a paper + community benchmark tables). It runs
Qwen3.8-Flash-Next — a huge MoE with 24,576 experts, of which about 10 fire per
token — by splitting the model across the whole PC: hot experts on the GPU, all
experts in RAM, the rest on the CPU, a lookup table on the SSD. Speculative
"guess then check" (1.6-1.8x, the same idea as our MTP), prefill in 8,192-token
chunks. Official floor is **12 GB VRAM + 32 GB RAM**; measured 53-94 tok/s write
and 1,620-2,650 tok/s read on an RTX 5070 12 GB (Q2_0..IQ3_S).

**The architecture point, which matters more than the numbers:** Strata and this
project attack *different bottlenecks*. Strata exists because the **weights** do not
fit — it is a weight-offload engine. This project exists because the **KV cache**
does not fit at 262K — it is a KV-offload engine. Same "share the work across the
whole PC" idea (Strata's own kitchen metaphor), applied to different halves of the
memory problem. The B站 stack composes all three layers: Strata (weights) + KVMem
(KV placement) + ActKV (KV size). That composition is why an 8 GiB card is claimed
to hold 1M context at all, and it is also below Strata's own 12 GB floor — the KV
layers are what buy the sub-spec operation.

**Two of Strata's own README statements independently confirm our measurements**
(useful, and worth citing because it is the largest project in this space):

- "Long prompts: reads the first message in full, about 1 minute per 30,000 tokens.
  Follow-up messages start in seconds." That is 500 tok/s cold and near-instant
  reuse — the same shape as our 772 s -> 2.7 s (286x) continuation result, and the
  same prefill ballpark as our 332-800 tok/s curve. Our numbers are normal for
  consumer hardware, not slow.
- Thinking off/low/medium/high with "Off is the fastest. High is best for hard
  questions." Same levels we measured as unreliable in the middle; same operating
  conclusion we reached (off for routine work).

**What they do better than us, honestly:** distribution and community. MIT license,
one-click installers, AI-assisted setup, an MCP server, multi-GPU docs, community
benchmark tables, a paper, 23.5k stars. Our project is a single-model, single-card
vertical by comparison. Our edge is depth on one configuration with visible
methodology — which is exactly the axis their README operates on, and exactly the
axis the B站 video skipped.

**ActKV** (arXiv 2609.31395, submitted 2026-09-25, ~3 weeks before we saw it):
KV cache *compression* for agentic inference. Three parts — (i) action-oriented
eviction ranking entries by contribution to action generation (action-region
attention scores + attention-aware LRFU), (ii) confidence-driven adaptive budget
(sliding-window confidence monitoring), (iii) page-aware compression management,
three primitives with custom kernels. Compression runs only at ORA boundaries
(end of each observation-reasoning-action round). Reported: 98.53% of FullKV accuracy
at 25.98% of peak KV memory, 3.97x token throughput, 3.58x task throughput; on
ALFWorld 81.34% and WebShop 101.41% of FullKV accuracy at 10% and 25% budgets.
Implementations for Transformers v5.3.0 and vLLM v0.19.0.

### The actual difference

```
ActKV : lossy in-place eviction. 75-90% of KV is discarded, irreversibly.
        Trades quality for VRAM. Real agent benchmarks (ALFWorld/WebShop).
ring  : lossless offload + retrieval. Nothing is discarded; all KV stays in host
        RAM at full precision, the device holds a fixed window plus IDF-retrieved
        pages. Trades host RAM for VRAM. Evidence: recall gates + real clients.
```

So the two answer different questions. "How much of the KV is worth keeping" is
ActKV's question; "where does the KV live" is ours. Composing them is the actual
lever, and it has a concrete payoff on our machine: we pin ~8 GiB of host KV at
262K, and 25% compression of that would free roughly 6 GiB of RAM.

### Why this is cheap to evaluate for us, not expensive

We already have the two ingredients the paper's hard part needs:

- Page-granular eviction and restore — that is the ring.
- ORA boundaries — our production load *is* the agent loop (DSH/ZCode tool
  turns), so the cadence ActKV compresses on is the cadence we already run.

And we hold a measurement that sharpens the paper into a testable prediction,
because ActKV's ranking is "action-region attention scores **and** attention-aware
LRFU", while we measured LRFU's contribution in our own engine as ~0 (see
`rc3-acceptance-2026-10-12.md`, LRFU attribution arm: prefill 285.3 vs 288.9,
decode 41.02 vs 42.92, accept 35.1% vs 36.6% — all noise).

### Pre-registered predictions (written before any measurement)

1. Porting only the LRFU half changes nothing measurable. Basis: our own
   attribution arm. If it moves anything, the attribution is wrong and that is
   worth knowing on its own.
2. The gain, if any, comes from the action-region scoring, and it will show up as
   *host RAM freed at equal recall*, not as speed. Basis: we drop nothing today,
   so there is no eviction cost to recover — the only thing eviction buys here is
   capacity.
3. At the budget we would actually ship (well above 25%), recall will be
   indistinguishable from today's 3/3 and the deliverable becomes a capacity
   number. Basis: the paper's own curve, where accuracy loss only appears at 10%.
4. The honest risk: their 1M-on-8G claim is arithmetically a 25%-budget claim, and
   their own 10% number is 81.34% of FullKV. If we ever want to compare, the
   comparison axis is quality at fixed budget on the same model and card, which
   nobody has published.

### Test plan

- Phase 1 (no GPU): re-read the paper's eviction criterion against our page/ring
  structures; produce the mapping or the reason it does not map. Deliverable is a
  write-up, like P4.
- Phase 2 (GPU, ours): implement action-region scoring only, evicting from the
  *host* pool at ORA boundaries. Gate is our own hole-region recall 3/3 at the
  258K shape plus the early markers; speed is measured but is not the gate.
- Phase 3 (only if Phase 2 holds): the same on-device, to compare against the
  paper's budget axis directly.

Cost: Phase 1 hours. Phase 2 is a real kernel change — days, not minutes. That is
the honest cost, and it is why this sits behind the write-ups rather than in front
of them.

## What this list is not

It is not a decision to migrate to RC3. That decision is behind the 262K gate
(RC3 default recipe on the 258K shape: survival, prefill/decode, early-marker
recall 3/3). The candidates above are independent of that outcome: P1 and P3 would
apply on either engine, P2 and P4 are write-ups, P5 is housekeeping, and P6 is
composed with whatever engine we end up on because it is a layer above both.
