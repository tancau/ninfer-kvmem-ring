#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "core/evictable_weight_pool.h"
#include "models/qwen3_5/load/bindings.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text = out->config.text;
    if (options.ranks > 1 && options.overlay_vision()) {
        // Overlay borrows weight memory from the primary device's evictable tail; an offloaded rank
        // holds expert blocks and nothing a Vision window could take. The two residency schemes are
        // answers to the same question -- where the bytes for something else come from -- and no
        // sound combination of them exists today, so say so rather than half-apply one.
        throw std::invalid_argument(
            "--vision-residency overlay and a multi-device --devices split cannot be combined");
    }
    out->weights.text =
        loading::bind_text(bindings, text, options,
                           loading::plan_pipeline_split(text.num_hidden_layers, options));
    if (options.overlay_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency overlay requires a Vision artifact");
    }
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(
            bindings, *out->config.vision, text,
            options.overlay_vision() ? artifact::Residency::Pinned : artifact::Residency::Device);
    }
    std::pair<std::size_t, std::size_t> mtp_parameters{bindings.weights.size(),
                                                       bindings.weights.size()};
    if (out->config.mtp) {
        out->weights.mtp      = loading::bind_mtp(bindings, text, out->weights.text);
        mtp_parameters.second = bindings.weights.size();
    }
    std::pair<std::size_t, std::size_t> draft_parameters{bindings.weights.size(),
                                                         bindings.weights.size()};
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
        draft_parameters.second = bindings.weights.size();
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    loading::apply_storage_trades(bindings, out->config, out->weights, options);
    if (options.overlay_vision()) {
        loading::apply_vision_overlay_placement(bindings, out->weights, mtp_parameters,
                                                draft_parameters);
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish(
        options.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1);
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    if (data->options.overlay_vision()) {
        // The tail is sized against the encode window once execution planning knows it; the
        // pool's mirror is captured then (see Model::weight_pool).
        const auto& materialization = data->materialization;
        if (materialization.evictable_tail_bytes == 0 || materialization.pinned_objects.empty()) {
            throw std::logic_error("overlay Vision load plan has no evictable tail or pinned tower");
        }
        if (!EvictableWeightPool::supported(device)) {
            throw std::invalid_argument(
                "--vision-residency overlay requires CUDA virtual memory management support");
        }
        pool = std::make_unique<EvictableWeightPool>(
            device, EvictableWeightPool::Config{
                        .arena_bytes =
                            static_cast<std::size_t>(materialization.device_capacity(0)),
                        .evictable_tail_bytes =
                            static_cast<std::size_t>(materialization.evictable_tail_bytes),
                    });
    }
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer,
                                         std::move(pool));
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    std::optional<VisionOverlayLayout> vision_overlay;
    if (data->options.overlay_vision()) {
        vision_overlay =
            loading::vision_overlay_layout(*data->weights.vision, bound, backing.pinned_block());
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing),
        std::move(vision_overlay)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
