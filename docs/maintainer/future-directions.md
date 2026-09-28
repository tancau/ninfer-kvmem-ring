# Future directions (after the KV ring work)

Recorded 2026-09-29. Ordering: current KV work first; these are next.

## 1. Bigger models via weight stratification (MoE expert offload)

Idea (precedent: Strata, which itself credits ninfer): keep the hottest MoE
experts on the Device, all experts in Host RAM, and compute non-resident experts
on the CPU concurrently with the GPU. The GPU never waits on the CPU.

- Benefit: a model whose resident-expert set does not fit VRAM can still run,
  with the rest served from RAM. Raises the model-size ceiling.
- Cost: NInfer's kernels are CUDA and its artifacts are `.ninfer` with
  nvfp4/ternary quantization. There is no CPU-side expert kernel for that format
  (llama.cpp/ggml has one for GGUF, which is a different format). A port is weeks:
  CPU expert kernels for the in-house quantization, CPU/GPU concurrent scheduling,
  pinned-memory pipelines.
- Hard limit: it is bounded by Host RAM, not VRAM. On 32 GB (about 20-25 GB usable
  after the OS and apps) only models whose non-resident weights fit that can run.
  A 125B-class model needs 30-50 GB of experts: out of reach.

## 2. Raise the RAM ceiling first

32 GB -> 64 GB is a cheap step (one kit). It raises the ceiling for (1) and also
enlarges the Host KV arena the ring demotes into, which is one of the ring's own
capacity bottlenecks. Do this before writing CPU kernels.

## 3. Chunked prefill at 8192

Current `--prefill-chunk` default is 1024. At concurrency 1 a larger chunk may
raise ingestion throughput; measure before changing (at concurrency > 1 a larger
chunk also widens the inter-token stall).

## 4. The three axes are the same problem

KV stratification (the ring), weight stratification (expert offload) and Host RAM
are three dimensions of one ceiling: VRAM + RAM. The current work attacks KV;
these attack weights; RAM raises both.
