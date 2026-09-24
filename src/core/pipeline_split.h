#pragma once

// Which device holds each layer's expert block.
//
// The point of a split here is KV capacity, not speed, so the question that matters is how much
// room is left on the card that serves attention. For these models the expert / MLP block is
// ~88% of the weights (17.3 GiB of 19.6 GiB on the 35B-A3B), and it is also the only part that can
// move without dragging state with it: attention needs its KV cache, GDN needs its recurrent
// state, and the head needs round state and the persistent prefill buffer -- all of which live on
// rank 0 and are reached directly by callers of the layer loop.
//
// So the split moves expert blocks only. Everything else stays on rank 0, which means the KV
// cache, GDN state pool, decoder state, state images and the whole context-cache machinery are
// untouched by this feature. Offloading every expert block leaves ~21 GiB free on rank 0 against
// ~3.4 GiB today -- about 2.2M int8 KV tokens against 262k -- for two crossings of the residual
// stream per offloaded layer, tens of microseconds each against a decode step of tens of
// milliseconds.
//
// A single-rank split is the identity mapping, which is what keeps every consumer a no-op on
// one GPU.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer {

struct LayerPlacement {
    std::size_t rank        = 0; // which device owns it
    std::uint32_t local     = 0; // index within that rank's layer arrays
};

class PipelineSplit {
public:
    // Identity: one rank owns every layer. local == global.
    explicit PipelineSplit(std::uint32_t layer_count)
        : layer_count_(layer_count), boundaries_{layer_count} {}

    // boundaries[i] is the exclusive end of rank i, ascending, last == layer_count.
    PipelineSplit(std::uint32_t layer_count, std::vector<std::uint32_t> boundaries)
        : layer_count_(layer_count), boundaries_(std::move(boundaries)) {
        if (boundaries_.empty()) { throw std::invalid_argument("pipeline split needs a boundary"); }
        if (boundaries_.back() != layer_count_) {
            throw std::invalid_argument("pipeline split must cover every layer");
        }
        if (layer_count_ == 0) { return; }
        // Non-descending rather than strictly ascending: a rank owning zero expert blocks is the
        // *default* here, not a bug. Putting every expert block on rank 1 leaves rank 0 with none
        // and is exactly the configuration that maximises KV room on the card serving attention.
        std::uint32_t previous = 0;
        for (const std::uint32_t end : boundaries_) {
            if (end < previous) {
                throw std::invalid_argument("pipeline split boundaries must be non-descending");
            }
            previous = end;
        }
    }

    // Balance by resident bytes rather than layer count. Layer sizes differ (MoE against
    // attention), and whatever a card does not spend on weights becomes KV -- so equalising bytes
    // is what equalises the KV each card can offer, which is the entire point of the split.
    static PipelineSplit balanced_by_bytes(std::span<const std::uint64_t> layer_bytes,
                                           std::size_t ranks) {
        const auto layer_count = static_cast<std::uint32_t>(layer_bytes.size());
        if (ranks <= 1) { return PipelineSplit(layer_count); }
        // Checked before any early return: asking for more ranks than layers is always a planning
        // bug, and silently collapsing to one rank would hide it.
        if (ranks > layer_count) { throw std::invalid_argument("more ranks than layers"); }

        // Prefix sums, then pick each boundary as the one closest to that rank's ideal cumulative
        // share. A greedy "close once the share is exceeded" rule overshoots badly when layer
        // sizes are uneven -- on a front-loaded model it produced 300 against 140 where the best
        // split is 200 against 240.
        std::vector<std::uint64_t> prefix(static_cast<std::size_t>(layer_count) + 1, 0);
        for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
            prefix[layer + 1] = prefix[layer] + layer_bytes[layer];
        }
        const long double total = static_cast<long double>(prefix[layer_count]);

        std::vector<std::uint32_t> boundaries;
        boundaries.reserve(ranks);
        std::uint32_t previous_end = 0;
        for (std::size_t rank = 0; rank + 1 < ranks; ++rank) {
            const long double ideal =
                total * static_cast<long double>(rank + 1) / static_cast<long double>(ranks);
            // Leave at least one layer for this rank and for each rank still to come.
            const std::uint32_t lowest  = previous_end + 1;
            const std::uint32_t highest = layer_count - static_cast<std::uint32_t>(ranks - rank - 1);

            std::uint32_t best         = lowest;
            long double best_distance  = -1.0L;
            for (std::uint32_t end = lowest; end <= highest; ++end) {
                long double distance = static_cast<long double>(prefix[end]) - ideal;
                if (distance < 0.0L) { distance = -distance; }
                if (best_distance < 0.0L || distance < best_distance) {
                    best_distance = distance;
                    best          = end;
                }
            }
            boundaries.push_back(best);
            previous_end = best;
        }
        boundaries.push_back(layer_count);
        return PipelineSplit(layer_count, std::move(boundaries));
    }

    // Every expert block on the last rank, none on rank 0. This is the default for two devices:
    // rank 0 then holds only embeddings, attention, GDN, norms and the head, and every byte it
    // saves becomes KV.
    //
    // `keep_on_first` moves that many layers' experts back onto rank 0, trading KV room for fewer
    // crossings; 0 is the configuration that maximises capacity.
    static PipelineSplit experts_on_last(std::uint32_t layer_count, std::size_t ranks,
                                         std::uint32_t keep_on_first = 0) {
        if (ranks <= 1) { return PipelineSplit(layer_count); }
        if (keep_on_first > layer_count) {
            throw std::invalid_argument("cannot keep more expert blocks than there are layers");
        }
        // Keeping every layer's experts on the first rank leaves the last rank with none, which
        // has no device tensors to materialize -- not a valid reduced-offload configuration.
        if (keep_on_first == layer_count) {
            throw std::invalid_argument(
                "cannot keep every expert block on the first rank: the last rank needs at least "
                "one layer to materialize");
        }
        std::vector<std::uint32_t> boundaries;
        boundaries.reserve(ranks);
        boundaries.push_back(keep_on_first);
        // Any middle ranks stay empty; only the first and last are used for a two-device split.
        for (std::size_t rank = 1; rank + 1 < ranks; ++rank) {
            boundaries.push_back(keep_on_first);
        }
        boundaries.push_back(layer_count);
        return PipelineSplit(layer_count, std::move(boundaries));
    }

    // Equal layer counts. Correct enough whenever layers are near-uniform in size -- which is the
    // case for a model whose per-layer MoE dominates and is identical across layers -- and the
    // sensible default before per-layer byte costs are available at planning time.
    static PipelineSplit even(std::uint32_t layer_count, std::size_t ranks) {
        if (ranks <= 1) { return PipelineSplit(layer_count); }
        if (ranks > layer_count) { throw std::invalid_argument("more ranks than layers"); }
        std::vector<std::uint32_t> boundaries;
        boundaries.reserve(ranks);
        for (std::size_t rank = 0; rank + 1 < ranks; ++rank) {
            boundaries.push_back(static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(layer_count) * (rank + 1)) / ranks));
        }
        boundaries.push_back(layer_count);
        return PipelineSplit(layer_count, std::move(boundaries));
    }

    [[nodiscard]] std::size_t ranks() const noexcept { return boundaries_.size(); }
    [[nodiscard]] std::uint32_t layer_count() const noexcept { return layer_count_; }
    [[nodiscard]] bool single_rank() const noexcept { return boundaries_.size() == 1; }

    [[nodiscard]] LayerPlacement placement(std::uint32_t global_layer) const {
        if (global_layer >= layer_count_) {
            throw std::out_of_range("layer index past the end of the pipeline split");
        }
        std::uint32_t begin = 0;
        for (std::size_t rank = 0; rank < boundaries_.size(); ++rank) {
            if (global_layer < boundaries_[rank]) {
                return LayerPlacement{rank, global_layer - begin};
            }
            begin = boundaries_[rank];
        }
        throw std::out_of_range("layer index past the end of the pipeline split");
    }

    [[nodiscard]] std::uint32_t rank_begin(std::size_t rank) const {
        check_rank(rank);
        return rank == 0 ? 0U : boundaries_[rank - 1];
    }

    [[nodiscard]] std::uint32_t rank_end(std::size_t rank) const {
        check_rank(rank);
        return boundaries_[rank];
    }

    [[nodiscard]] std::uint32_t rank_layers(std::size_t rank) const {
        return rank_end(rank) - rank_begin(rank);
    }

    // True when a copy of the residual stream is needed after this layer. Exactly one per rank
    // boundary, which is the property that makes a layer split tolerant of a bridgeless link.
    [[nodiscard]] bool crosses_after(std::uint32_t global_layer) const {
        if (global_layer + 1 >= layer_count_) { return false; }
        return placement(global_layer).rank != placement(global_layer + 1).rank;
    }

private:
    void check_rank(std::size_t rank) const {
        if (rank >= boundaries_.size()) { throw std::out_of_range("pipeline rank out of range"); }
    }

    std::uint32_t layer_count_ = 0;
    std::vector<std::uint32_t> boundaries_;
};

// What a single binding/materialization pass owns.
//
// A rank uploads only its own layers; everything else is bound ValidateOnly, so the artifact is
// still checked in full against the file while only this rank's bytes reach this device. That
// reuses the placement mechanism already used for disabled features rather than inventing a
// second one.
//
// Default-constructed it owns everything, which is what keeps the single-GPU path byte-identical.
struct RankOwnership {
    const PipelineSplit* split = nullptr;
    std::size_t rank           = 0;

    [[nodiscard]] bool whole_model() const noexcept {
        return split == nullptr || split->single_rank();
    }

    // The expert / MLP block of a layer -- the only thing a split actually moves.
    [[nodiscard]] bool owns_layer_experts(std::uint32_t layer) const {
        return whole_model() || split->placement(layer).rank == rank;
    }

    // Everything that is not an expert block: embeddings, attention and GDN projections, norms,
    // and the head. All of it stays on rank 0, because each piece is tied to state that cannot
    // move with it -- the KV cache, the GDN recurrent state, round state and the persistent
    // prefill buffer.
    [[nodiscard]] bool owns_core() const noexcept { return whole_model() || rank == 0; }

    // The embedding feeds layer 0, so it belongs with the first rank.
    [[nodiscard]] bool owns_embedding() const noexcept { return owns_core(); }

    // The head -- final norm, output head, draft head, MTP -- stays on **rank 0**, not on the rank
    // that produces the final hidden state.
    //
    // It looks natural to put it with the last layer, but the head does not just consume the
    // hidden state: it writes into round state and the persistent prefill-hidden buffer, and feeds
    // sampling, all of which live on rank 0 and are reached directly by callers of run_layers.
    // Moving the head would strand those on the wrong device. Instead the residual stream comes
    // back to rank 0 after the last layer, which costs one extra crossing per token -- tens of
    // microseconds against a decode step of tens of milliseconds -- and leaves everything outside
    // run_layers exactly as it was on one GPU.
    [[nodiscard]] bool owns_head() const noexcept { return owns_core(); }
};

} // namespace ninfer
