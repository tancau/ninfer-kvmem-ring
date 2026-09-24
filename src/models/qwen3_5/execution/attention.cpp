#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/rotation.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    const Tensor& signs = projection_signs(parameters.projection);
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return rotated_workspace_bytes(
            signs, weight.k, last,
            ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                          single->policy, first, last));
    }
    const auto& pair = std::get<ops::PairedProjectionWeights>(parameters.projection);
    return rotated_workspace_bytes(signs, pair.first.k, last,
                                   ops::attn_input_proj_split_workspace_capacity_bytes(
                                       pair.first.qtype, pair.first.n, pair.second.qtype,
                                       pair.second.n, pair.first.k, pair.policy, first, last));
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(x, pair->first, pair->second, query, gate, key, value, pair->policy,
                             workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(x, single.weight, query, gate, key, value, single.policy, workspace,
                             stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
