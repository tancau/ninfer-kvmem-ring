# RC3 comparison and acceptance — 2026-10-12

Upstream `kvmem/kvmem-llama.cpp` released v0.18.0-ninfer-rc3 on 11 Oct 2026 (commit
`938c2f1`, source branch `feat/multi-backend-framework`). It ships an sm86 prebuilt
for RTX 30 series and uses the same NInfer v3 `.ninfer` artifact we already run, so
it is directly comparable on this machine. This file records the measurements; it
does not decide anything on its own.

## What was run

- Package: `ninfer-server-rtx30-sm86-ui-938c2f1.zip` (sha256 verified against
  `SHA256SUMS-ninfer-only.txt`), extracted to `D:\kvmem-rc3\pkg\...`.
- Launched with the package's own `start-ninfer.ps1` and **only two overrides**:
  `-Model` pointing at our existing artifact, and `-Context 262144`. Everything else
  is the upstream default recipe: B/R = 36864/16384, prefill chunk 256, dual INT8 KV,
  adaptive MTP4 + ngram31, thinking budget 8192, host KV budget 12288 MiB,
  sink/recent 5120 each, LRFU on (lambda 0.5, top-frac 0.9, GQA max).
- Port 18200, single client, same 87-message / 33-tool shapes used for our own
  acceptance matrix (shared builders, byte-identical prefixes).
- Startup: ready in ~10 s, model id `bonsai2-27b`, max_model_len 262144, VRAM 10.3 GiB
  of 12 GiB. It fits with the default recipe; no parameter had to be reduced.

## Acceptance matrix (same shapes, same artifact, same card)

| cell | ship (franken, pool 98304) | RC3 default recipe |
|---|---|---|
| OpenAI 173K | prefill 109.6 tok/s, decode 24.65, accept 44.1% | prefill 285.3, decode 41.02, accept 35.1% |
| OpenAI 173K recall (early 3) | 3/3 | 3/3 |
| OpenAI 262K | survived 46 min, decode ~21 | survived 15.3 min, decode 46.08 |
| OpenAI 262K recall (early 3) | 3/3 | 3/3 |
| Anthropic 50K | survived | 82 s, recall 3/3, blocks thinking+text, end_turn |
| Anthropic 262K | survived, cache_read 65536, three markers exact | cold 880 s (253,369 input tokens), recall 3/3 |
| Anthropic 262K continuation | (same shape, partial reuse) | **5.0 s**, cache_read 253,364 (99.998%), recall 3/3 |

Notes on the numbers:

- All server-reported. RC3 non-stream responses carry a `timings` block
  (`prompt_per_second`, `predicted_per_second`, `draft_n`, `draft_n_accepted`);
  client-side wall-clock decode rates are unreliable (streaming flushes make
  TTFT ≈ elapsed) and were discarded.
- The Anthropic route reports usage as `input_tokens` / `cache_read_input_tokens`.
  The continuation cell read 253,364 of 253,369 input tokens from cache.
- Recall gate uses markers in the message region (PARROT-5/15/27 for the Anthropic
  shape, PARROT-1050/1275/1499 for the 173K sequenced shape). Padding units are
  all-PARROT-999 in the 262K shape, so the padding region has no discriminating
  markers — same limitation as our own 262K gate.

## LRFU attribution arm

Same 173K shape, `-NoLrfu` (and the aggregation override was lost in argument
forwarding, so the arm is NoLrfu + GQA max, i.e. a clean single-variable diff
against the default arm's LRFU + GQA max):

| | default (LRFU on) | NoLrfu | delta |
|---|---|---|---|
| prefill | 285.3 tok/s | 288.9 | noise |
| decode | 41.02 tok/s | 42.92 | noise |
| draft accept | 35.1% | 36.6% | noise |

**LRFU contributes ~0 on this workload.** The 2.6x prefill and 1.7x decode gap is
the retrieval architecture and recipe (prefix-identity inheritance, TopK retrieval,
chunk 256, dual INT8, adaptive MTP4 + ngram31), not history weighting. Upstream's
own release notes are consistent with this: their LRFU is retrieval-TopK ranking
only, with no decode-time score accumulation, adaptive budget, or quota mixing.

## Prefix-reuse behaviour (the largest single difference)

On a continuation with an identical prefix:

| | ship | RC3 |
|---|---|---|
| Anthropic 262K continuation | (partial; 65536 cache_read previously measured) | 5.0 s total, cache_read 253,364 / 253,369 |
| OpenAI 173K continuation | 1409 s, cache 64K of 172908 | 5.5 s, cache_n 256,049 (at 262K shape 6 s) |

Same operation, ~256x difference in wall clock. Our engine restores the anchor and
recomputes the tail on every continuation; RC3 matches on prefix identity and
restores only the selected content. This is the mechanism behind the prefill gap,
and it is the same mechanism behind the slow rounds in our own 301-interrupt saga.

## Behaviour differences to verify before any production change

1. `reasoning_budget_tokens` in the request is **not honoured** on the NInfer path
   (sent 128, observed 3000/3000 reasoning tokens, no control). That field belongs
   to the llama.cpp route. Budget control is the engine-wide
   `--default-thinking-budget`.
2. Tool calls: well-formed under the strict-schema default (`arguments` is a JSON
   string, `id` present, `finish_reason=tool_calls`). An `anyOf` schema did not
   produce the "explicit error" the release notes warned about; it emitted a value
   satisfying one branch. Both behaviours are acceptable but differ from ours.
3. Host RAM: with the engine loaded and a 253K session in host cache, working set
   7.8 GiB, free RAM 16.2 GiB of 31.95 GiB. The 12 GiB host KV budget fits, but the
   accounting is different from ours (we pin ~8 GiB for host KV at 262K).

## Not yet done

- Thinking-fuse behaviour at the actual cap (probe in flight at time of writing).
- Multi-turn continuation stress (several successive commits on one session).
- Restart cycles and orphan/VRAM-release checks.
- Real client end-to-end (ZCode through the Anthropic route).
- Production launcher migration (current launcher points at franken on 8081).
