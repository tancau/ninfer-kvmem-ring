// ninfer::ops - split-KV causal small-T launcher and unified route dispatcher. INT8 Q/K
// preparation, including their paired fixed rotation, remains private to the included kernel.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry>
std::int32_t causal_small_t_split_count(std::int32_t window, std::int32_t tokens,
                                        KvCacheStorage storage) {
    // There used to be a SmallTMaximumSplits bump here for Fp8E4M3Row256 at tokens==1 and
    // window>8198, and the device asked for that bump for *every* quantized storage while only
    // fp8 was granted it. TODO.md section 2c read the asymmetry as the host shortchanging nvfp4
    // and k8v4 -- 69 splits where their kernels asked for 85 -- and proposed extending the grant.
    //
    // Measured on this 3090, and it is the other way round: more splits at depth is worse.
    // Extending the grant to nvfp4 and k8v4 cost them 2.3-3.1% on the 35B at every depth, against
    // an fp8 control that moved 0.3%. Removing it from fp8 as well gained 0.4/0.5/1.2% at
    // 4,096/16,384/32,768 over two samples per variant, with within-variant spread well under the
    // difference at the two deeper points.
    //
    // So the bump is gone from both sides -- here, and from
    // causal_small_t_quantized_active_splits -- and every quantized storage now takes the same
    // default tier. That also retires the invariant section 2c flagged as held by coincidence:
    // the partial kernel and the reducer no longer depend on two call sites happening to agree
    // about a special case, because there is no special case.
    //
    // The 27B was not conclusive either way (its fp8 control moved +-3% between identical runs,
    // which is worth knowing before trusting any single-sample result on that model).

    // A 64-key default split just above a 32-key boundary makes the partial kernel execute a
    // nearly empty second tile. T=5 uses one 32-key tile per split; the short T>=6 profile keeps
    // all newly appended rows in one tail split while retaining a useful B=8 grid.
    const bool i8_family = storage == KvCacheStorage::Int8Group64 ||
                           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    if (i8_family && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::SmallTSplitScale);
    }
    if (i8_family && tokens >= 6 && window > 128 && window <= 160) {
        constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
        return div_up(window, kKeysPerSplit);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (i8_family && tokens >= 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return causal_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t causal_small_t_launch_capacity(CausalAttentionExecutionEnvelope envelope,
                                            std::int32_t tokens, KvCacheStorage storage) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = causal_small_t_split_count<Geometry>(static_cast<std::int32_t>(window),
                                                                    tokens, storage);
        capacity          = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

// CTAs per SM of the partial kernel launch_tc_partial_i8 selects for this width and window (its
// MinBlocksPerSm): one for the wide-CTA short-window routes, two otherwise.
template <typename Geometry>
int i8_partial_ctas_per_sm(std::int32_t tokens, std::int32_t window) {
    if (tokens >= 6) { return window > 8198 ? 2 : 1; }
    if (tokens == 5) { return window > (Geometry::GroupSize == 6 ? 1029 : 4096) ? 2 : 1; }
    if (tokens == 4) { return window > 1029 ? 2 : 1; }
    return 2;
}

int device_multiprocessors() {
    static const int count = [] {
        int device = 0;
        int sms    = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return 0;
        }
        return sms;
    }();
    return count;
}

// Splits per full wave of a single-row INT8-family launch. The split tiers were set where their
// largest count (85 splits of four KV heads at two CTAs per SM) is exactly one wave of a 170-SM
// part; on a part with fewer SMs the same counts leave a nearly empty last wave that costs a full
// CTA time. The partial kernel and the reducer take this count and hold their per-window split
// count to whole waves (causal_small_t_active_splits); zero leaves the tiers alone.
template <typename Geometry>
std::int32_t causal_small_t_wave_splits(std::int32_t tokens, std::int32_t implementation_window) {
    return device_multiprocessors() *
           i8_partial_ctas_per_sm<Geometry>(tokens, implementation_window) / Geometry::KVHeads;
}

// The launch capacity the whole-wave policy can reach over the envelope.
template <typename Geometry>
std::int32_t causal_small_t_wave_capacity(std::int32_t capacity, std::int32_t tokens,
                                          CausalAttentionExecutionEnvelope envelope) {
    const auto window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const int wave    = causal_small_t_wave_splits<Geometry>(tokens, window);
    if (wave <= 0 || capacity <= wave) { return capacity; }
    return std::min(capacity, wave * div_up(div_up(window, kCausalSmallTSplitKeyLimit), wave));
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    causal_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                    Masked, CacheInput>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__nv_bfloat16*>(cache_v.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, std::int32_t wave_splits, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // A U8 value plane is the rk8v4 packed signed int4 coding.
    const bool packed_values = cache_v.dtype == DType::U8;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        auto issue = [&]<bool PackedValues>() {
            if constexpr (DynamicArena) {
                configure_cuda_device_once([&] {
                    return cudaFuncSetAttribute(
                        causal_attention_small_t_i8_tiled_kernel<
                            Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                            MultiBatch, Masked, CacheInput, PackedValues>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
                });
            }
            causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput, PackedValues>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, wave_splits, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
        };
        if (packed_values) {
            issue.template operator()<true>();
        } else {
            issue.template operator()<false>();
        }
    };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size) {
    if (tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("causal_softmax_attention split capacity: invalid profile");
    }
    (void)paged_kv_storage_layout(cache_storage, kCausalHeadDim);
    const bool i8_family = cache_storage == KvCacheStorage::Int8Group64 ||
                           cache_storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    if (q_heads == CausalD256H24Kv4::QHeads) {
        const int capacity =
            causal_small_t_launch_capacity<CausalD256H24Kv4>(envelope, tokens, cache_storage);
        if (batch_size > 1) {
            // Keep complete grids within one or two 170-SM waves. Rounding from 160 CTAs
            // leaves room for the indivisible 4*B group, including B=3/5/6/7.
            const bool narrow = tokens <= 5;
            int target_ctas   = 160;
            if (cache_storage == KvCacheStorage::BFloat16)
                target_ctas =
                    narrow || batch_size >= 5 || envelope.max_visible_keys > 4096 ? 320 : 160;
            else if (cache_storage == KvCacheStorage::Int8Group64)
                target_ctas = narrow || envelope.max_visible_keys > 4096 ? 320 : 160;
            else if (cache_storage == KvCacheStorage::Nvfp4Group16)
                target_ctas = narrow ? 320 : 160;
            const int grid_limit = div_up(target_ctas, 4 * batch_size);
            // A split stages at most 64 physical-page IDs. Leave two 64-key pages for
            // key-tile rounding and page alignment at the 262144-key resource limit.
            const int page_limit =
                div_up(static_cast<int>(envelope.max_visible_keys), kCausalSmallTSplitKeyLimit);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return i8_family
                   ? causal_small_t_wave_capacity<CausalD256H24Kv4>(capacity, tokens, envelope)
                   : capacity;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        const int capacity =
            causal_small_t_launch_capacity<CausalD256H16Kv2>(envelope, tokens, cache_storage);
        return i8_family && batch_size == 1
                   ? causal_small_t_wave_capacity<CausalD256H16Kv2>(capacity, tokens, envelope)
                   : capacity;
    }
    throw std::invalid_argument(
        "causal_softmax_attention split capacity: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                         float scale, PagedKVBatchLayerView cache,
                                         const CausalSmallTInvocation& invocation,
                                         CausalAttentionExecutionEnvelope envelope,
                                         Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                         Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits                = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);
    const bool i8_family = cache.storage == KvCacheStorage::Int8Group64 ||
                           cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    const std::int32_t wave_splits =
        i8_family && invocation.batch_size == 1
            ? causal_small_t_wave_splits<Geometry>(invocation.width, implementation_window)
            : 0;

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_CAUSAL_SMALL_T_DISPATCH(TOKENS, WARPS)                                              \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            if (cache.storage == KvCacheStorage::Int8Group64 ||                                    \
                cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {                  \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked>(                      \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, wave_splits, partial_acc, partial_m, partial_l, \
                    stream);                                                                       \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_CAUSAL_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_CAUSAL_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_CAUSAL_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_CAUSAL_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_CAUSAL_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_CAUSAL_SMALL_T_DISPATCH(6, 4);
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            NINFER_CAUSAL_SMALL_T_DISPATCH(7, 4);
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            NINFER_CAUSAL_SMALL_T_DISPATCH(8, 4);
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_launch: unsupported T");
    }
#undef NINFER_CAUSAL_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch_reduce   = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                          invocation.width * invocation.batch_size);
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                        Offset><<<grid, kReduceBlock, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), static_cast<const std::int32_t*>(pos.data),
            invocation.valid_columns
                  ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                  : nullptr,
            invocation.width, invocation.full_width, invocation.column_begin, invocation.batch_size,
            splits, Int8 ? wave_splits : 0, static_cast<__nv_bfloat16*>(out.data));
    };
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0)
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        else
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
    };
    const auto launch_for_storage = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (invocation.valid_columns)
                launch_profile.template operator()<Int8, false, true>();
            else
                launch_profile.template operator()<Int8, false, false>();
        } else if (invocation.valid_columns)
            launch_profile.template operator()<Int8, true, true>();
        else
            launch_profile.template operator()<Int8, true, false>();
    };
    // rk8v4 is an int8-family cache on this fork and takes the same path.
    if (cache.storage == KvCacheStorage::Int8Group64 ||
        cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64)
        launch_for_storage.template operator()<true>();
    else
        launch_for_storage.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_small_t_k8v4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                             envelope, column_begin, width, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_small_t_fp8_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                            envelope, column_begin, width, partial_acc, partial_m,
                                            partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_small_t_nvfp4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                              envelope, column_begin, width, partial_acc, partial_m,
                                              partial_l, out, stream);
        return;
    }
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream);
}

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_cached_small_t_k8v4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                    partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_cached_small_t_fp8_launch(q, pos, scale, cache, envelope, partial_acc,
                                                   partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_cached_small_t_nvfp4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                     partial_m, partial_l, out, stream);
        return;
    }
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, batch_cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
}

} // namespace ninfer::ops::detail
