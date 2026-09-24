#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// CUDA function attributes are scoped to a device context. A process-wide `static` result
// therefore leaves the same kernel unconfigured the first time it launches on a second GPU, which
// shows up as a launch failure or silently wrong output rather than as a clear error. Give each
// launcher specialization a cheap, device-keyed cache instead.
//
// Single-GPU behaviour is unchanged: the map holds exactly one entry.
template <typename Configure>
void configure_cuda_device_once(Configure&& configure) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaError_t> results;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaError_t result = cudaSuccess;
    {
        const std::scoped_lock lock(mutex);
        const auto existing = results.find(device);
        if (existing != results.end()) {
            result = existing->second;
        } else {
            result = std::forward<Configure>(configure)();
            results.emplace(device, result);
        }
    }
    CUDA_CHECK(result);
}

// How many pieces one cross-rank transfer is split into.
//
// A crossing is a serial D2H then H2D, so the two halves never overlap: a 4 MB residual stream
// costs ~0.765 ms measured, against the ~0.33 ms its bandwidth alone implies. Splitting the byte
// range lets piece i+1 stream out of the source while piece i streams into the destination. It is
// a pure byte-level pipeline -- the same bytes in the same order -- so it cannot affect which
// kernels run or what they compute.
inline constexpr std::size_t kCrossingPipelineDepth = 4;

// Default pinned staging capacity: covers a 1024-token residual crossing (hidden 5120, BF16)
// with generous headroom. A caller that configures a larger prefill chunk than this covers
// should size the crossing staging buffer explicitly at construction instead of relying on this.
inline constexpr std::size_t kDefaultCrossingStagingBytes = 64ULL << 20;

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    // One entry keeps the single-device route. Two entries hold a second endpoint open for
    // model-parallel execution: matching compute capability is required, before any weight is
    // uploaded. Bidirectional peer access is only probed and recorded as a capability -- it is not
    // required, since crossings stage through pinned host memory when it is unavailable.
    //
    // `min_crossing_staging_bytes` sizes the pinned cross-rank staging buffer (see
    // `crossing_staging()`); the default covers only a modest residual crossing. A caller that
    // configures a larger prefill chunk must size this explicitly, since one crossing has to fit
    // in a single staged transfer.
    explicit DeviceContext(std::span<const int> device_ids,
                           std::size_t min_crossing_staging_bytes = kDefaultCrossingStagingBytes);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    // Streaming-multiprocessor count of the attached device. Distinct from compute_capability():
    // every sm_86 part shares capability 86 but not this count (RTX 3090 has 82, RTX 3090 Ti has
    // 84), so any device-wide residency budget must read this, not compute_capability().
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool model_parallel() const noexcept;
    // True when the devices can DMA directly to each other. False is not an error: cudaMemcpyPeer
    // still works, staging through host memory at roughly 13us per hop instead of a couple. Only
    // consult this to pick between schedules -- a design crossing once per token does not care,
    // one crossing twice per layer does.
    [[nodiscard]] bool peer_access() const noexcept;
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept;
    [[nodiscard]] cudaStream_t stream_for_rank(std::size_t rank) const;
    // Weight upload goes to the arena of the rank that owns the object, and a host-to-device copy
    // has to be issued on a stream belonging to the destination device.
    [[nodiscard]] cudaStream_t transfer_stream_for_rank(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t fence_for_rank(std::size_t rank) const;
    // Pinned host staging for cross-rank copies. Allocated only for a model-parallel context.
    //
    // cudaMemcpyPeerAsync is the obvious way to move a tensor between ranks and is the wrong one
    // here on two counts, both measured on a bridgeless 2x 3090: it runs at roughly half the rate
    // of an explicit D2H/H2D pair through pinned host (0.69 ms against 0.33 ms for 4 MB), and it
    // cannot be captured into a CUDA graph at all -- capture fails with
    // cudaErrorStreamCaptureUnsupported, whereas a memcpy to or from pinned host is an ordinary
    // graph node. Losing capture costs prefill a factor of 3.4, which dwarfs the transfer itself,
    // so being capturable matters far more than the copy rate.
    [[nodiscard]] void* crossing_staging() const noexcept;
    // Fence for one piece of a pipelined cross-rank transfer.
    [[nodiscard]] cudaEvent_t piece_fence(std::size_t rank, std::size_t piece) const;
    // Recorded on the destination stream once a staged piece has been read out of the shared
    // crossing buffer. The next crossing waits on it before overwriting that piece, which is what
    // keeps one pinned buffer safe across back-to-back crossings.
    [[nodiscard]] cudaEvent_t piece_consumed_fence(std::size_t rank, std::size_t piece) const;
    // Whether that fence may be waited on from work whose capture id is `capture_id` (0 for work
    // outside any capture). During capture, a wait on an event whose last record was not part of
    // the same capture fails with cudaErrorStreamCaptureIsolation -- and the crossing path exists
    // to be captured -- so the first crossing inside a graph has nothing to wait for and says so
    // here. Its safety comes from the graph instead: the captured crossings join back into the
    // origin stream, so one launch's H2Ds all complete before the next launch's D2Hs begin.
    //
    // The one thing this cannot express is an eager crossing still in flight when a graph holding
    // crossings is launched, since a launch cannot wait on an eager fence without breaking the
    // capture. Callers capture while the decoder is idle and launch afterwards, which is the only
    // order this path is used in.
    [[nodiscard]] bool piece_consumed_visible(std::size_t rank, std::size_t piece,
                                              unsigned long long capture_id) const;
    // Records that the fence for this piece has just been recorded by work with that capture id.
    void note_piece_consumed(std::size_t rank, std::size_t piece, unsigned long long capture_id);
    [[nodiscard]] std::size_t crossing_staging_bytes() const noexcept;
    void activate_rank(std::size_t rank);
    void synchronize_rank(std::size_t rank) const;
    void synchronize() const;
    int sm() const noexcept;

private:
    struct Endpoint {
        int device                   = 0;
        cudaStream_t stream          = nullptr;
        cudaStream_t transfer_stream = nullptr;
        cudaStream_t vision_stream   = nullptr;
        cudaEvent_t fence            = nullptr;
        // Fences for pipelining one cross-rank transfer in pieces. Pre-allocated because events
        // cannot be created during CUDA graph capture.
        std::array<cudaEvent_t, kCrossingPipelineDepth> piece_fences{};
        std::array<cudaEvent_t, kCrossingPipelineDepth> piece_consumed{};
        // Where each consumed fence was last recorded: outside any capture, and the capture it
        // belonged to. Both are kept because a graph's event-record node only takes effect when
        // the graph is launched, so an earlier eager record remains the event's real state.
        std::array<bool, kCrossingPipelineDepth> piece_consumed_eager{};
        std::array<unsigned long long, kCrossingPipelineDepth> piece_consumed_capture{};
        cudaDeviceProp props{};
    };

    void refresh_active_aliases() noexcept;
    void release() noexcept;

    void* crossing_staging_             = nullptr;
    std::size_t crossing_staging_bytes_ = 0;
    std::vector<Endpoint> endpoints_;
    std::vector<int> device_ids_;
    std::size_t active_rank_ = 0;
    bool peer_access_        = false;
};

// Binds a rank for the duration of a scope and restores the previous one, so a caller that has to
// touch the secondary device cannot leave the thread bound to it.
class ScopedDeviceRank {
public:
    ScopedDeviceRank(DeviceContext& context, std::size_t rank);
    ~ScopedDeviceRank() noexcept;

    ScopedDeviceRank(const ScopedDeviceRank&)            = delete;
    ScopedDeviceRank& operator=(const ScopedDeviceRank&) = delete;

private:
    DeviceContext& context_;
    std::size_t previous_rank_ = 0;
};

// Moves `bytes` from `source` on `from_rank` to `destination` on `to_rank` through the shared
// pinned crossing buffer, pipelined over `kCrossingPipelineDepth` pieces and ordered entirely by
// events, so the whole transfer stays capturable into a CUDA graph.
//
// Callers with both ranks on one physical device should copy device-to-device instead; this path
// exists for a genuine two-card crossing. It lives here rather than in the decoder so a test can
// drive it with two ranks pinned to one device and check the fence protocol without a second card.
void stage_cross_rank_copy(DeviceContext& context, const void* source, std::size_t from_rank,
                           void* destination, std::size_t to_rank, std::size_t bytes);

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
