#include "models/qwen3_5/load/bindings.h"

#include "artifact/transcode.h"
#include "artifact/views.h"

#include <cmath>

namespace ninfer::models::qwen3_5::loading {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format,
                             artifact::Residency residency) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference = binder.parameter(name, std::move(shape), residency, exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
#if defined(NINFER_SM8X_COMPAT)
        // sm_86/sm_89 have no FP8 or FP4 tensor cores. A stored permission for A8/A4 activations
        // is an upper bound, not a requirement, so FP8 and NVFP4 weights run their A16 routes,
        // which dequantize the stored codes before the matmul.
        result.policy = ops::LinearPolicy::A16Only;
#endif
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role == "hadamard_signs") {
                if (pending.reference.shape.size() != 2) {
                    throw artifact::ArtifactError(name + "@" + input +
                                                  ": Hadamard signs require a matrix");
                }
                result.hadamard_signs =
                    hadamard_signs(binding, pending.reference.shape[1], name + "@" + input);
                continue;
            }
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::hadamard_signs(const artifact::Binding& binding, std::uint64_t width,
                                  const std::string& use) {
    if (width == 0 || width % 1024 != 0) {
        throw artifact::ArtifactError(use + ": Hadamard rotation needs a multiple of 1024 columns");
    }
    std::string key = std::to_string(width);
    for (const auto& part : binding.parts) {
        const auto& object = binder.reader().directory().object(part.object);
        key += "|" + artifact::object_id(object) + ":" + std::to_string(part.begin) + "-" +
               std::to_string(part.end);
    }
    if (const auto found = signs_.find(key); found != signs_.end()) { return found->second; }
    PendingWeight pending;
    pending.reference = binder.binding("hadamard_signs/" + std::to_string(signs_.size()), binding,
                                       {width}, artifact::Residency::Device, QType::BF16);
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    weights.push_back(std::move(pending));
    signs_.emplace(std::move(key), id);
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format,
                          artifact::Residency residency) {
    return parameter(std::move(name), std::move(shape), {}, format, residency);
}

void Bindings::evict(WeightId id, std::uint32_t rank) {
    const auto& parameter = at(id);
    if (parameter.reference.residency != artifact::Residency::Device) {
        throw artifact::ArtifactError(parameter.reference.name +
                                      ": only a device parameter can be evictable");
    }
    for (const auto& part : parameter.reference.binding.parts) {
        binder.evict_device(part.object, rank);
    }
}

void Bindings::place(WeightId id, std::size_t rank) {
    if (rank == 0) { return; }
    const auto& parameter = at(id);
    if (parameter.reference.residency != artifact::Residency::Device) {
        throw artifact::ArtifactError(parameter.reference.name +
                                      ": only a device parameter can be offloaded to another GPU");
    }
    for (const auto& part : parameter.reference.binding.parts) {
        binder.device_rank(part.object, rank);
    }
}

bool Bindings::transcode(WeightId id, QType target, std::string_view option) {
    const auto& parameter = at(id);
    bool requested        = false;
    for (const auto& part : parameter.reference.binding.parts) {
        const auto& geometry = binder.reader().geometry(part.object);
        if (geometry.format == target) { continue; }
        if (geometry.layout != QuantLayout::RowSplit ||
            !artifact::row_split_transcode_supported(geometry.format, target)) {
            throw std::invalid_argument(std::string(option) + " requires " +
                                        parameter.reference.name +
                                        " to be stored as row-split Q8_G32");
        }
        binder.transcode_device(part.object, target);
        requested = true;
    }
    return requested;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
