# The `franken/v0.11` line

`franken/v0.11` is the ashalliants `ninfer-3090` v0.11.0 release (317ddea7) plus the behaviours
this fork needs in production, re-implemented on the v0.11 tree rather than cherry-picked from
the earlier v0.10-based line. Each commit stands on its own and carries its rationale; this note
is the map.

## What the line carries over v0.11.0

| area | behaviour | where it lives in the v0.11 tree |
|---|---|---|
| context cache | a private capture that cannot be placed reclaims the oldest eligible private resident (publication order), so every conversation regains reuse from its second turn | `runtime/engine/context_cache/resource_manager.h`, `shared_capture_planner.h`, the capture gate in `models/qwen3_5/program/transactions/capture.cpp` |
| context cache | an infeasible shared capture takes the same reclamation route, so the shared reuse frontier keeps advancing under pressure | `resource_manager.h` |
| context cache | replacement scenarios are offered only when no catalog slot is vacant, so an exact repeat does not displace the leading-instruction prefix the next question needs | `resource_manager.h` |
| context cache | the demand window behind the committed mask holds 64 requests, beyond the replay distance of a workload that replays a few dozen prefixes twice | `context_portfolio_value.h`, `materialization_planner.h`, `resource_manager.h`, `shared_capture_planner.h` |
| context cache | a private-only capture with no committed demand and no shared credit goes straight to that reclamation: the capture search could not produce a plan for it, yet spent its whole 4,096-target budget on the engine thread first (80 ms to 0.5 s before the first token of every new conversation once the private catalog filled) | `resource_manager.h` |
| engine | store exhaustion during placement (`ContextCacheExhausted`, a `std::bad_alloc` naming the store) fails only the request with HTTP 429 and is counted, instead of failing the worker | the reservation sites in `models/qwen3_5/program/{program_impl,graphs,storage/context,transactions/materialization}.cpp`, the catches in `progress_materialization_transaction`, `runtime/engine/engine_core.h` |
| engine | a Paged KV exhaustion names its page numbers; three consecutive exhaustions without a successful admission mark the Engine unavailable for the healthcheck | `core/paged_kv_cache.cpp`, `engine_core.h` |
| serve | a named or single-tool forced `tool_choice` is executed by opening the call in the generation prompt (v0.11 rejects it with `tool_choice_not_supported`) | `models/qwen3_5/frontend/chat_template.cpp` (opener after the generation prompt), `tool_call_parser.{h,cpp}` (seeded decoder), `serve/translate.cpp` (reasoning rule), the three request parsers, `serve/request_log.*` (schema v22) |

The line also carries ternary checkpoints, ported from the earlier v0.10-based ternary work:

| area | behaviour | where it lives in the v0.11 tree |
|---|---|---|
| format | `t2_g128_fp16`: ternary codes as 2-bit two's complement with one binary16 scale per 128 columns, row-split only | `core/weight.h`, `core/weight_view.cpp`, `artifact/formats.cpp`, `tools/artifact/` |
| ops | T2 linear routes (small-T tensor-core kernel, narrow MMA tiles for prefill), composed T2 attention/GDN input projections, `linear_add`/`linear_swiglu`, the T2 full head and the indexed T2 proposal head in `linear_topk` (keys carry the shortlist's global token ids) | `ops/linear/t2/`, `ops/wrapper/`, `ops/linear_topk/t2.cu`, `ops/linear_topk/grouped_ksplit_topk.cuh` |
| ops | T2 int8 activations at every width (`--prefill-a8`): a small-T s8 kernel for decode, speculative verify and prompts up to 192 tokens (one launch over both parents of an attention or GDN pair), and above it the int8-activation GEMM of the dense formats, padding T to its cheapest tile width, whose ternary codes decode straight to s8 in the shared mainloop | `ops/linear/t2/t2_small_t_i8.cuh`, `ops/common/rowsplit_a8_mma.cuh` (`T2Codec`), `ops/linear/t2/t2_a8.{h,cu}`, `ops/linear/linear.cpp`, the T2 paths of `ops/wrapper/{linear_add,attn_input_proj,gdn_input_proj}.cpp` |
| ops | the T2 pair's GDN record and snapshot projections take the pair's policy (their two-parent forms were A16 only); the record kernel stages its window in shared memory before the recurrence | `ops/wrapper/gdn_input_proj.cpp`, `models/qwen3_5/execution/gdn.cpp`, `ops/linear_attention/gated_delta_net/recurrent.cuh` |
| ops | `linear_dynamic_grouped_conv_add` takes any row-split projection Linear registers at `[5120, 4096\|17408]`: Linear writes the BF16 plane the materialised Q8 route fills and the shared finish kernel adds the convolution (Q8 keeps its fused routes); Q4 shapes `n5120_k4096`, `n5120_k17408` and `n5120_k25600` for a Q4 DFlash2 adapter | `ops/wrapper/dynamic_grouped_conv.cpp`, `ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_{plan.cpp,materialized.cu}`, `ops/linear/q4/shapes/` |
| ops | `hadamard_transform` and `silu_mul_hadamard`; producers that write their output rotated (`rmsnorm_hadamard`, `gated_rmsnorm_hadamard`, `sigmoid_mul_hadamard`, `gdn_norm_gating_proj_rotated`), bit-identical to the op followed by the transform, each running four warps per 1024-point block (`hadamard_quarter_forward_store`); `embedding_rotated` gathers a T2 token row and applies the inverse transform | `ops/kernel/hadamard_transform.cuh`, `ops/kernel/hadamard_producers.cuh`, `ops/wrapper/hadamard_transform.cpp`, `ops/gdn_gating_proj/bf16/` |
| model | Hadamard-rotated Uses (`hadamard_signs` auxiliary): the norms and gates hand rotated inputs to rotated projections (`InputBasis`), other shapes rotate in place; a T2 token table is restored at the gather with the output head's hidden-width signs; the residual stream stays primal; the ternary output, MTP, DFlash2 and proposal heads take the text projections' integer route | `models/qwen3_5/execution/rotation.h`, `parameters.cpp`, `load/prepare.cpp`, `model.cpp`, the attention/GDN/FFN/head sites |
| attention | a single row of the INT8-family small-T attention runs its splits in whole waves of the device's SMs: the host passes the partial kernel's splits per wave to the partial kernel and the reducer, whose per-window policy rounds down to whole waves within the 64 staged page IDs (the split tiers fill one wave of a 170-SM part; on 82 SMs their counts left a nearly empty last wave) | `ops/softmax_attention/dense/causal_cache/small_t.{cu,cuh}`, `small_t_i8.cuh` |
| vision | an overlay encode window may also borrow the DFlash adapter (ranked below MTP), which a ternary table and head alone cannot cover | `models/qwen3_5/load/vision_overlay.cpp`, `load.cpp` |
| convert | `bonsai2_27b_ternary` builds a Qwen3.8-27B artifact whose text tower, head and token table come from PrismML's PQ2_0 GGUF, with the DFlash2 adapter's feature, output and MLP projections in Q4 (its fused query/key/value projection stays Q8); `--source mtp` takes the MTP head from a separately trained file; `--proposal` gathers the proposal head's rows from the imported ternary head in its own encoding | `tools/convert/ternary.py`, `tools/convert/sources/gguf.py`, `tools/convert/qwen3_5.py`, `tools/convert/proposal.py` |

Deliberately not carried: the LRU catalog policy (`--context-cache-policy`), the host-state byte
budget, the context-trace diagnostics and the prefix-cache scenario battery of the previous line.
Measured in production, the branching policy did not help and sometimes hurt; the fixes above are
what actually keeps prefills from being triggered.

## Verification

- `ninfer_resource_manager_test` after every context-cache commit (it builds on a Mac without
  CUDA: Homebrew clang, `-include exception`, the Xcode SDK sysroot).
- `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test` (fixture tokenizer), the OpenAI,
  Responses and Anthropic schema tests cover the forced tool call.
- `ninfer_hadamard_transform_test`, `ninfer_linear_t2_a16_test`, `ninfer_linear_t2_a8_test` and the
  T2 full and indexed cases of `ninfer_linear_topk_test` check the ternary ops against FP64 oracles;
  the T2 cases of `ninfer_attn_input_proj_test`, `ninfer_gdn_input_proj_test` and
  `ninfer_gdn_input_proj_conv_record_test` cover the pairs under both policies;
  `ninfer_linear_dynamic_grouped_conv_add_test` runs Q8, Q4 and Q5 projections and
  `ninfer_linear_q4_a16_test` the adapter's Q4 shapes; `ninfer_softmax_attention_test` runs the
  INT8-family small-T cases at the capped split counts; `tests/convert/test_ternary.py` covers the
  GGUF mapping and `tests/convert/test_proposal.py` the exact proposal rows.
- The full build and `ctest` on an RTX 3090 (sm_86) before the branch is published.

## Deployment

Deployment files are not part of this public line. The production checkout adds them on a
private branch on top of `franken/v0.11`.
