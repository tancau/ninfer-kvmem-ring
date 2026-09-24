# Multi-GPU expert offload

Goal, in the user's terms: run int8 KV with several large concurrent sessions, which a single 24 GB
3090 cannot do. This is the design, the reasoning behind it, and the status.

## What it does

With `--devices 0,1`, the expert / MLP block of each layer is materialized on the second GPU.
Rank 0 keeps everything else -- embeddings, attention, GDN, norms, the head -- and therefore keeps
the KV cache, the GDN recurrent state, round state and the whole context-cache machinery. During a
forward pass the residual stream crosses to rank 1 for each layer's mlp tail and comes straight
back.

Every byte rank 0 sheds becomes KV, which is the entire point.

## Why the expert block, and not whole layers

The first plan was a layer split: layers `[0,k)` on one card, `[k,N)` on the other. Reading the
execution path showed that is the expensive way to get what we want.

KV lives with attention, so the question is not "how do we divide the model" but "how much can we
get **off** the card that serves attention". Three things are pinned to rank 0 regardless:

- **attention** needs its KV cache,
- **GDN** needs its recurrent state,
- **the head** writes round state and the persistent prefill-hidden buffer, which callers of
  `run_layers` reach directly (`rmsnorm` into `prefill_hidden_`, then `io_.logits` / `io_.token`).

Moving whole layers drags KV pools, GDN state pools, decoder state, state images and the host
offload / context cache along with them: 33 `decoder->` sites, 43 `state_images->` sites and ~198
KV-paging sites, plus the planner.

The expert block is pinned to nothing, and it is **~88% of the weights** -- 17.3 GiB of 19.6 on the
35B-A3B (465 MB per layer x 40), and a similar share of the 27B's dense MLP. Offloading only that
leaves every one of those subsystems untouched.

| | KV room on rank 0 | tokens at ~10 KiB | surface |
|---|---|---|---|
| single card today | ~3.4 GiB | ~262k | -- |
| offload 20 layers' experts | ~12.6 GiB | ~1.3M | small |
| **offload all experts (default)** | **~21.2 GiB** | **~2.2M** | small |
| full layer split | ~26 GiB | ~2.7M | 274+ sites |

83% of the benefit for a fraction of the risk.

## What it is not

**It does not make anything faster.** The two cards work in sequence, not in parallel: rank 0 idles
during each mlp tail and rank 1 idles the rest of the time. Total weight bytes read per token are
unchanged, so decode speed is roughly what one card gives, minus the crossing overhead. The win is
capacity.

**Peer access is not required.** Crossings stage through pinned host memory, which works on any
pair. Confirmed on a rented bridgeless 2x 3090, where `cudaDeviceCanAccessPeer` returns 0.

**NVLink would help prefill, and barely touch decode.** Decode crossings are ~4 KiB and
latency-bound, so a bridge saves microseconds against a decode step of tens of milliseconds --
which is why decode reaches 90% of single-GPU with MTP and no bridge at all. Prefill crossings are
~4 MB and bandwidth-bound: 336 MB per chunk is the whole of the remaining 16% gap, and that is
where a bridge would pay. It is the one place the earlier "NVLink is irrelevant" claim was too
broad -- true per token, not true per prefill chunk.

## How it works

- **`PipelineSplit` / `RankOwnership`** (`core/pipeline_split.h`) answer which rank holds a layer's
  expert block, and pin everything else to rank 0. A single-rank split is the identity mapping,
  which keeps every consumer a no-op on one GPU. `experts_on_last` is the default; an empty rank 0
  is the point of it, not a bug.
- **`DeviceArena` carries one backing per rank** and switches between them, so one workspace serves
  both cards and all 76 existing `work_` call sites are unchanged. `Scope` records the rank it was
  taken on -- a scope opened on one rank legitimately outlives a switch to another, and restoring a
  foreign bump pointer would hand out overlapping scratch. `reset()` clears every rank, since it
  marks a round boundary.
- **`Program`** allocates a workspace per rank, each while its own device is current.
- **`run_mlp_tail`** crosses the residual stream to the card holding the experts, runs the whole
  tail there, and copies the result back into the caller's rank-0 buffer.
  `post_attention_norm` is bound *with* the expert block so the offloaded card does rmsnorm and the
  expert matmuls back to back -- one crossing out and one back, rather than two of each.
- **Ordering** uses per-rank streams and fences, never a host sync: record on the producer, wait on
  the consumer, issue the copy on the consumer's stream.
- **Loading** is one binding pass over one artifact. Each device object carries the rank that owns
  it (`Binder::device_rank`), the plan records a device capacity per rank, and `materialize()`
  allocates one arena per rank and uploads that rank's objects under `ScopedDeviceRank` with its own
  staging pass — an H2D copy and the event retiring its slot both belong to the destination device.
  Every parameter is bound and validated exactly as on one GPU, so `bind_view`, `Model` and
  `Parameters` are unchanged.
- **Overlay Vision residency and a multi-device split are mutually exclusive**, and the combination
  is rejected at startup rather than half-served: the evict-ranked tail exists so a Vision window can
  borrow weight memory on the card that runs Vision, and an offloaded rank holds expert blocks with
  nothing lendable.

## The trap

The residual stream is workspace memory and each rank has its own workspace, so a crossing is
between two arenas. The offloaded rank must run against *its* buffer, and the result must land back
in the caller's. Getting that wrong yields a program that runs, reads another device's memory
through a stale pointer, and emits plausible rubbish rather than crashing.

That is why the hardware test compares **greedy text against the single-GPU reference** rather than
checking that it starts. Clean startup and sensible memory figures are both compatible with a build
that is silently reading the wrong device.

## Measured on two RTX 3090s

Bridgeless pair, PCIe, 4,000-token prompt, int8 KV. Output is byte-identical to the single-GPU
reference in every split row, including with MTP and on both models.

### Qwen3.6-35B-A3B

| config | prefill | decode | free for KV |
|---|---|---|---|
| single GPU | 6.34k | 176.8 | 3.71 GiB |
| single GPU + MTP3 | 6.31k | 267.1 | 2.87 GiB |
| split | 5.30k (84%) | 137.1 (78%) | **21.1 GiB** |
| **split + MTP3** | 5.25k (83%) | **240.5 (90%)** | **20.3 GiB** |

**Run it with MTP.** Speculation processes several tokens per forward pass, so the same crossings
serve four tokens instead of one and the decode penalty falls from 22% to 10%.

### Qwen3.8-27B

| config | prefill | decode | free for KV |
|---|---|---|---|
| single GPU | 1.14k | 37.0 | 7.38 GiB |
| split | 1.04k (91%) | **37.6 (102%)** | **16.5 GiB** |

Its decode is unaffected: the 27B is dense and compute-bound, so the crossings are cheap relative
to the work. Its weights also divide less dramatically (6.79 / 9.13 GiB against 2.15 / 17.4 on the
35B) because its attention and GDN projections at hidden 5120 are large next to its dense MLP.

### Capacity, which is the point

`--kv-capacity auto`, largest each mode resolves:

| | max total KV | note |
|---|---|---|
| single GPU | 237,248 tokens | 262,144 context **fails to start at all** |
| split, C=8 @ 262k | **1,929,728 tokens** | 8 sessions at ~241k each, 1.08 GiB spare |

**8.1x the KV**, and a single 3090 cannot run 262k context at any concurrency.

Concurrency is capped at 8 by `kMaximumConcurrency` (`include/ninfer/types.h`), which is a
compile-time constant sizing arrays in the admission policy and engine core -- not a memory limit.
Raising it is a separate, bounded change.

## Alternative considered: tensor-parallel MLP split

[devon-caron/ninfer-dual-3090-nvlink](https://github.com/devon-caron/ninfer-dual-3090-nvlink)
implements a working dual-3090 split on an older NInfer base, by physically sharding each MLP's
packed Q4/Q5 weight matrices across both cards (partial tensor parallelism) rather than offloading
whole expert blocks. Reported on their machine: prefill +41%, decode +17.5%, restricted to the
dense 27B target.

That design was not ported here, for three reasons. Their base and ours had diverged by 280
commits against their 2 by the time it was evaluated, and a trial merge produced ~90 conflict
hunks with no mechanical resolution -- the two trees had rewritten the same execution path
(`program_impl.h`) for unrelated reasons. Their decode gain, on inspection, comes from a design
that crosses the link twice per split MLP per layer (128 crossings/token for the 27B); ours crosses
once per layer's mlp tail, and expert offload rather than weight sharding gets 83% of the KV-room
win (see the table above) without needing NVLink or a rewrite of the execution path. Their design
also excludes the 35B-A3B target this fork tunes for by default.

## How performance got here

Three findings, in order, because two of them contradicted the obvious guess:

**The prefill loss was CUDA graphs, not the transfers.** The split disabled graph capture because
`cudaMemcpyPeerAsync` cannot be recorded into a graph. The control that settled it:
`NINFER_KEEP_EXPERTS=39`, two crossings per pass instead of eighty, still gave 1.90k prefill --
and a single GPU with `--no-cuda-graph` gave 1.88k. The same number. Capture was worth a factor of
3.4 on prefill and had been written off at 0.7% from a decode-only measurement.

**Staging through pinned host fixes both problems at once.** A D2H/H2D pair is an ordinary graph
node where the peer copy is not, and it is also about twice as fast on a bridgeless pair (0.33 ms
against 0.69 ms for 4 MB). Prefill went 1.38k -> 4.27k.

**A crossing is serial, so its halves never overlap.** Prefill scales linearly with crossing count
at ~0.765 ms each for a 4 MB stream, against the ~0.33 ms bandwidth implies -- exactly what a
strict D2H-then-H2D costs. Splitting the byte range into four pieces pipelines them: 4.27k -> 5.30k.

The remaining gap is real PCIe traffic, 336 MB per prefill chunk.

## Status


- [x] Layer/rank mapping, with tests (`tests/test_pipeline_split.cpp`).
- [x] Per-rank arena switching, with tests (`tests/test_arena_ranks.cpp`).
- [x] Rank-partitioned bindings and per-rank materialization, both targets.
- [x] Per-rank workspaces.
- [x] Cross-device mlp tail execution.
- [x] **27B**, all three of its layer-binding paths (groupwise, NVFP4, and the Qwen3.8 NVFP4/FP8
      mix).
- [x] `--devices N,M` on the CLI, so the split can be checked by output rather than by inspection.
- [x] Hardware equivalence run: greedy text from the split against the single-GPU reference,
      byte-identical on both targets (see Measured on two RTX 3090s, above).
- [x] KV capacity measured with `--kv-capacity auto` in both modes (see Capacity, above).
- [ ] CUDA graphs across the boundary. Capture is per-device, so a cross-device schedule cannot be
      one graph. Whether decode capture survives the offload, and what it costs if not, is
      unmeasured.
- [ ] Concurrency and context-cache behaviour under the split. These live entirely on rank 0 so
      they should be unaffected, but "should be" is not "measured".

## Earlier hardware runs

A dual 3090 box (bridgeless, `nvidia-smi topo -m` reporting `PHB`) confirmed the pieces this is
built on:

```
peer 0->1=0
enable_peer 0->1=peer access is not supported between these two devices
memcpy_peer_1MiB=OK
device_context=OK size=2 model_parallel=1 peer_access=0
```

Peer access is unavailable on a consumer pair and irrelevant: the copy works anyway. `DeviceContext`
constructs across both cards with distinct per-rank streams and fences.
