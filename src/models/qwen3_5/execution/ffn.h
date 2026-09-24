#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"

namespace ninfer::models::qwen3_5::execution {

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false, bool verify = false, InputBasis basis = InputBasis::Primal);
// The sign vector of a rotated Dense FFN's input projection, or null.
[[nodiscard]] const Tensor* ffn_input_signs(const FfnParameters& parameters);

} // namespace ninfer::models::qwen3_5::execution
