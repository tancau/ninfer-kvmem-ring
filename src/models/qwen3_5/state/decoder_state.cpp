#include "models/qwen3_5/state/decoder_state.h"

#include "ops/kv_cache/d256_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim,
                              KvCacheStorage storage, std::int32_t table_rows,
                              std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (head_dim != ops::kD256KVCacheHeadDim) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
    const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage);
    const bool scaled                     = profile.quant_group != 0;

    const std::uint32_t logical_pages = page_count(capacity);
    // LOCAL PROTOTYPE (KVMem-style ring): the physical pool is allowed to be SMALLER than the
    // logical capacity. The address-space allocator recycles the oldest logical pages as the
    // sequence grows, and the retrieval mask keeps attention inside the resident window.
    (void)logical_pages;

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * (scaled ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {profile.key_code_dtype, profile.key_leading_extent, kv_heads, 256});
        geometry.planes.push_back(
            {profile.value_code_dtype, profile.value_leading_extent, kv_heads, 256});
        if (scaled) {
            geometry.planes.push_back(
                {profile.key_scale_dtype, profile.scale_leading_extent, kv_heads, 256});
            geometry.planes.push_back(
                {profile.value_scale_dtype, profile.value_scale_leading_extent, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .allow_logical_reservation = logical_pages > physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers         = layers,
        .max_context    = capacity,
        .kv_heads       = kv_heads,
        .head_dim       = head_dim,
        .storage        = storage,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), storage_(layout.storage) {
    // LOCAL PROTOTYPE (KVMem-style ring): when the pool is smaller than the logical capacity the
    // retrieval mask is always active. Allocate its (all-ones) device buffer up front so the
    // pointer stays stable across CUDA Graph capture, which bakes kernel arguments.
    const std::uint32_t logical_pages = page_count(max_context_);
    if (pages_.usable_pages() < logical_pages) {
        const std::size_t words = (static_cast<std::size_t>(logical_pages) + 31U) / 32U;
        std::vector<std::uint32_t> ones(words, ~0U);
        install_selection(ones);
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = ops::d256_kv_cache_profile(storage_).quant_group != 0;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .storage       = storage_,
        .selected_blocks = selected_blocks_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor());
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
        .selected_blocks = direct.selected_blocks,
    };
}

PagedKVCache::~PagedKVCache() {
    if (selection_device_ != nullptr) {
        (void)cudaFree(selection_device_);
        selection_device_ = nullptr;
    }
}

void PagedKVCache::install_selection(std::span<const std::uint32_t> words) {
    if (words.empty()) {
        clear_selection();
        return;
    }
    if (words.size() > selection_words_) {
        if (selection_device_ != nullptr) {
            (void)cudaFree(selection_device_);
            selection_device_ = nullptr;
        }
        selection_words_ = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&selection_device_), words.size_bytes()) !=
            cudaSuccess) {
            selection_device_ = nullptr;
            throw std::runtime_error("PagedKVCache: retrieval bitmap allocation failed");
        }
        selection_words_ = words.size();
    }
    if (cudaMemcpy(selection_device_, words.data(), words.size_bytes(), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        throw std::runtime_error("PagedKVCache: retrieval bitmap upload failed");
    }
    selected_blocks_ = selection_device_;
}

void PagedKVCache::install_sink_recency_selection(std::uint32_t committed_pages,
                                                  std::uint32_t sink_pages,
                                                  std::uint32_t window_pages) {
    const std::uint32_t total = committed_pages == 0 ? 1U : committed_pages;
    if (window_pages == 0 || window_pages >= total) {
        clear_selection();
        return;
    }
    // LOCAL PROTOTYPE: span the WHOLE logical page space, not just the committed prefix. During a
    // prefill chunk the kernel also attends the chunk's own (not-yet-committed) pages, so a bitmap
    // sized to `committed_pages` would be read out of bounds.
    const std::uint32_t logical_pages = page_count(max_context_);
    const std::uint32_t span = std::max(total, logical_pages);
    const std::size_t words = (static_cast<std::size_t>(span) + 31U) / 32U;
    std::vector<std::uint32_t> bits(words, 0U);
    const auto select_page = [&](std::uint32_t page) {
        if (page < span) { bits[page >> 5] |= (1U << (page & 31U)); }
    };
    // LOCAL PROTOTYPE: when the physical pool is smaller than the logical capacity the allocator
    // recycles the oldest pages, so a preserved attention "sink" cannot be honoured. Drop it.
    const std::uint32_t pool_pages = page_pool().usable_pages();
    const std::uint32_t sink = pool_pages < logical_pages ? 0U : std::min(sink_pages, total);
    for (std::uint32_t page = 0; page < sink; ++page) { select_page(page); }
    const std::uint32_t first_recent = total > window_pages ? total - window_pages : 0U;
    for (std::uint32_t page = first_recent; page < span; ++page) { select_page(page); }
    install_selection(bits);
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
