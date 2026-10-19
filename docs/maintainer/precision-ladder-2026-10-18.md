# Precision ladder: community Swift-1.5 Qwen3.8 conversions on the RC3 engine (2026-10-18)

Runbook. Nothing here touches production; every run is an **experiment instance**
on port 18210 with its own logs. Needs an exclusive GPU window (ComfyUI and
GPT-SoVITS closed, production stack stopped by the user).

**Status 2026-10-19: rung 1 done, window closed.** iq3xxs: **MMLU-Pro 226/280 =
80.7%** (4 items still capped at the 8,192 fix budget) vs the ternary baseline's
76.4% on the same slice — **+12 items net** (23 gained / 11 lost at item level).
Rung 2 (iq2xs) was offered and deferred by the user; the runbook stays valid.
Engine stopped, VRAM back to 615 MiB.

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

   Then re-run capped items exactly like the baseline did, with a watched cap:

   ```
   py C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro_fix.py 18210 8192
   copy C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results.json C:\Users\Administrator\AppData\Local\Temp\opencode\mmlupro-results-swift15-iq3xxs.json
   ```

   **Do not raise the fix budget above ~8192 on this arena**: prompt + answer
   must fit the 12,288-token device window (~8.5K is fine; 16,384 would overflow
   into ring demotion and change what the attention sees mid-answer).

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

## Execution log (2026-10-19)

- Load (rung 1: Budget 8192 / Reserve 4096 / Sink 2048 / Recent 2048):
  **succeeded first try**, no fallback needed. Cold read from `H:\` 2 m 38 s
  (62.7 MiB/s), device weights 9.71 GiB, KV runtime 12,288 tokens (8,192 + 4,096),
  steady VRAM 11,441/12,288 MiB (engine reports free 308 MiB). Model id from
  `/v1/models`: `swift-1.5-qwen3.8-27b-gsq-rco-iq3xxs`.
- Short-context decode (from the MMLU request rows, 100–1,100-token prompts):
  **52–64 tok/s**, s/round 64–69 ms, host ~1.5 ms/round, acceptance ~50%
  (e.g. first item: 2213/4349). Production reference at 0–5k: 84.7 tok/s.
- MMLU-Pro 280 started 2026-10-19 00:04; capped items to be re-run with
  `mmlupro_fix.py 18210 8192` (cap argument above).
- 01:23 the OpenCode location service **evicted the workspace mid-run** and tore
  down the still-attached background shell; the runner was killed between items
  (no traceback anywhere; engine logged its last request as `cancelled`). The
  per-item checkpoint survived: **205/280 (77.6%) intact**. `mmlupro_test.py`
  gained resume support (skip-done; same seed ⇒ identical sample order — the
  checkpoint's `i` indexes line up 1:1), and both scripts are launched as
  detached processes now so a repeat eviction cannot kill them
  (`mmlupro_fix.py` also checkpoints after every item).
- Resume 01:44 → main run finished 02:07: **221/280 = 78.9%**, 25 items capped.
- Fix pass 03:23→04:04 at budget 8,192 (25 items, 40 min, decode 40–57 tok/s):
  **AFTER FIX 226/280 = 80.7%, still capped 4** — all four in `engineering`, and
  all four are wrong on the baseline too, so the residual cap does not tilt the
  comparison (input-side upper bound if they were answerable: 82.1%).
- Item-level result vs the baseline (same 280 sampled items): both correct 203,
  only-iq3xxs 23, only-baseline 11, neither 43 → **net +12**. By category:
  other +4, law +3, philosophy +3; chemistry −1, business −2; the rest ±0/+1.
- Window closed: engine stopped, VRAM back to 615 MiB, progress page and its
  temporary firewall rule removed; production stack left stopped for the user.

## Reporting table to fill

| artifact | load | arena | MMLU-Pro 280 | decode short | decode ~60K | prefill ~60K |
|---|---:|---:|---:|---:|---:|---:|
| Bonsai 2 27B ternary (prod, ref) | — | 73,728 | **76.4** | ~103 | 41.0 (173K shape) | 285 (173K) |
| swift15_iq2xs | 128 s | 61,440 | (optional) | — | 27.8 | 437.5 |
| swift15_iq3xxs | 158 s | 12,288 | **80.7** (4 capped) | 52–64 | — | — |

Caveats to carry into any write-up: same suite and item count as the published
reference, **not the same items**; the community quant ladder's own tablet
numbers are vendor-side claims from the video research (cited as such); this
ladder measures the community artifacts, not the production recipe; the
baseline's fix rounds ran at 16,384 (production window) vs the 8,192 cap here —
the four residual capped iq3xxs items (all also wrong on the baseline) are the
only place that shows.
