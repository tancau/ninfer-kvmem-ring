# Precision ladder: community Swift-1.5 Qwen3.8 conversions on the RC3 engine (2026-10-18)

Runbook. Nothing here touches production; every run is an **experiment instance**
on port 18210 with its own logs. Needs an exclusive GPU window (ComfyUI and
GPT-SoVITS closed, production stack stopped by the user).

## What and why

`H:\kvmem-ninfer` holds community conversions of Qwen3.8-27B (the same base
family as the production Bonsai artifact) in the `gguf_blocks_v1` dialect our
engine reads:

| artifact | bytes on disk | status |
|---|---:|---|
| `swift15_iq2xs_mtp.ninfer` | 9,420,962,560 | loaded and run before (60K decode, GSM8K-50) |
| `swift15_iq3xxs_mtp.ninfer` | 11,092,478,720 | **never loaded** — this runbook's target |

Question being answered: what does the IQ3_XXS rung of the community quant ladder
give on our engine — MMLU-Pro quality and context cost — against our ternary
baseline (76.4% on the same 280-item slice, zero truncation) and against the
community's own claims carried in the video research.

## Fit math (measured, from the previous iq2xs load)

- Card total 12,884,246,528 B; iq2xs weights 8,767,884,330 B → 3,046,113,280 B
  free → arena tested **45,056 + 16,384 = 61,440 tokens** (2.21 GB payload).
- iq3xxs is ~2.32 GB bigger → expect ≈0.7 GB free → arena ≈ **12–16K tokens**
  (vs 61,440). KVMem still allows the full 262,144-token logical context via host
  tiering, but with a small device window long-context decode will be slow; treat
  quality as the primary deliverable and speed as a side reading.

## Steps

1. **Start the experiment engine** (production recipe minus vision; own logs):

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Administrator\AppData\Local\Temp\opencode\ninfer_qwen38_load.ps1 `
     -Model H:\kvmem-ninfer\swift15_iq3xxs_mtp.ninfer -Port 18210 `
     -Budget 8192 -Reserve 4096 -Sink 2048 -Recent 2048
   ```

   Fallbacks if validation or OOM rejects the combination (try in order):
   `-Budget 6144 -Reserve 4096 -Sink 1536 -Recent 1536`; then
   `-Budget 4096 -Reserve 2048 -Sink 1024 -Recent 1024`. The loader prints the
   load summary (`available_after_weights`, real arena) — copy it into the result
   notes.

2. **Smoke test:** one short `/v1/chat/completions` request; `/props` answers;
   VRAM steady via `nvidia-smi`.

3. **MMLU-Pro, same slice as the baseline:** first preserve the baseline file,
   then run:

   ```
   copy C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results-baseline-bonsai.json
   py C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro_test.py 18210 280
   copy C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results-swift15-iq3xxs.json
   ```

   Protocol is already aligned: 280 items, seed 20261016, temperature 0,
   `reasoning_effort: none`, `max_tokens` 3072 with capped-item detection, engine
   `--default-max-tokens 16384` (the no-truncation setting from the baseline run).
   Item-level comparison against the baseline slice is meaningful because it is
   the *same* sampled items.

4. **Context cost, same session:**
   - short-context decode: 256-token generation after a ~1K prompt → tok/s;
   - arena actually granted: from the load summary (`kv_payload_bytes`,
   `available_after_startup_bytes`);
   - long-context sanity: one ~12K-token prompt (fits the small window best
   case) with a 256-token answer; if it completes, record prefill and decode;
   do not expect the 60K shape to be meaningful at this arena.
   - for reference, iq2xs measured the same way: 60K shape → prefill 437.5 tok/s
   (59,534 tokens in 136 s), decode 27.8 tok/s, drafts accepted 125/389 (32.1%).

5. **Optional second rung:** repeat step 3–4 for `swift15_iq2xs_mtp.ninfer`
   (loads with the 61,440 arena) so the ladder has two points measured on the
   same day, same protocol. GSM8K-50 for iq2xs already exists.

6. **Cleanup:** stop the engine process, confirm VRAM back to baseline, keep
   `qwen38-*` logs.

## Reporting table to fill

| artifact | load | arena | MMLU-Pro 280 | decode short | decode ~60K | prefill ~60K |
|---|---:|---:|---:|---:|---:|---:|
| Bonsai 2 27B ternary (prod, ref) | — | 73,728 | **76.4** | ~103 | 41.0 (173K shape) | 285 (173K) |
| swift15_iq2xs | 128 s | 61,440 | (optional) | — | 27.8 | 437.5 |
| swift15_iq3xxs | ? | ? (≈12–16K) | **target** | ? | — | — |

Caveats to carry into any write-up: same suite and item count as the published
reference, **not the same items**; the community quant ladder's own tablet
numbers are vendor-side claims from the video research (cited as such); this
ladder measures the community artifacts, not the production recipe.
