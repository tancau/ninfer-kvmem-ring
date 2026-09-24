#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id)
    : DeviceContext(std::span<const int>(&device_id, 1)) {}

DeviceContext::DeviceContext(std::span<const int> device_ids,
                             std::size_t min_crossing_staging_bytes) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_ids.empty() || device_ids.size() > 2) {
        throw std::invalid_argument("DeviceContext requires one or two CUDA devices");
    }
    device_ids_.assign(device_ids.begin(), device_ids.end());
    for (std::size_t i = 0; i < device_ids_.size(); ++i) {
        const int id = device_ids_[i];
        if (id < 0 || id >= count) {
            throw std::runtime_error("CUDA device " + std::to_string(id) + " does not exist: " +
                                     std::to_string(count) +
                                     (count == 1 ? " device is visible" : " devices are visible"));
        }
        // A repeated id is allowed on purpose. Two ranks on one card is not a useful deployment --
        // it frees no memory -- but it exercises the entire split path (per-rank binding and
        // materialization, per-rank workspaces, and the cross-rank copies) on a single-GPU
        // machine. Every bug caught that way is one not caught by renting two cards.
    }

    endpoints_.resize(device_ids_.size());
    try {
        for (std::size_t rank = 0; rank < device_ids_.size(); ++rank) {
            Endpoint& endpoint = endpoints_[rank];
            endpoint.device    = device_ids_[rank];
            err                = cudaSetDevice(endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
            }
            err = cudaGetDeviceProperties(&endpoint.props, endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaGetDeviceProperties failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.transfer_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.vision_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(vision_stream) failed", err));
            }
            err = cudaEventCreateWithFlags(&endpoint.fence, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaEventCreateWithFlags(fence) failed", err));
            }
            for (cudaEvent_t& piece : endpoint.piece_fences) {
                err = cudaEventCreateWithFlags(&piece, cudaEventDisableTiming);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        cuda_error_message("cudaEventCreateWithFlags(piece fence) failed", err));
                }
            }
            for (cudaEvent_t& piece : endpoint.piece_consumed) {
                err = cudaEventCreateWithFlags(&piece, cudaEventDisableTiming);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        cuda_error_message("cudaEventCreateWithFlags(piece consumed) failed", err));
                }
            }
        }

        // Two ranks on the same card need no peer setup: copies between them are ordinary
        // device-to-device, and enabling peer access to self is an error.
        if (endpoints_.size() == 2 && endpoints_[0].device == endpoints_[1].device) {
            peer_access_ = true;
        } else if (endpoints_.size() == 2) {
            const Endpoint& first  = endpoints_[0];
            const Endpoint& second = endpoints_[1];
            if (first.props.major != second.props.major ||
                first.props.minor != second.props.minor) {
                throw std::invalid_argument(
                    "model-parallel CUDA devices must have the same compute capability");
            }
            // Peer access is a performance capability, not a requirement. NVIDIA disables P2P over
            // PCIe on GeForce, so a pair of 3090s without an NVLink bridge reports can_access == 0
            // -- but cudaMemcpyPeer still works there, staging through host memory. Refusing to
            // start would rule out the most common consumer dual-GPU box for no reason, so record
            // the capability and let callers choose a schedule that suits it.
            //
            // Measured on this fork's reference 3090 (PCIe 4.0 x16), a staged hop costs ~13us and
            // is latency-bound: flat from 4KiB to 40KiB, which spans both models' per-token
            // activation (35B-A3B hidden 2048 -> 4KiB, 27B hidden 5120 -> 10KiB). That is
            // negligible for a schedule crossing once per token, and a few percent of a decode
            // step for one crossing twice per layer.
            peer_access_ = true;
            for (std::size_t source = 0; source < 2; ++source) {
                const std::size_t destination = 1 - source;
                int can_access                = 0;
                err = cudaDeviceCanAccessPeer(&can_access, endpoints_[source].device,
                                              endpoints_[destination].device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        cuda_error_message("cudaDeviceCanAccessPeer failed", err));
                }
                if (can_access == 0) {
                    peer_access_ = false;
                    continue;
                }
                err = cudaSetDevice(endpoints_[source].device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
                }
                err = cudaDeviceEnablePeerAccess(endpoints_[destination].device, 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else if (err != cudaSuccess) {
                    // Capability without a usable mapping: fall back rather than fail outright.
                    (void)cudaGetLastError();
                    peer_access_ = false;
                }
            }
        }
    } catch (...) {
        release();
        throw;
    }

    if (endpoints_.size() > 1) {
        // Sized for the largest thing a crossing carries: one prefill chunk of the residual
        // stream. The caller (which knows the configured prefill chunk and the model's hidden
        // size) is responsible for passing a capacity that covers it; pinned host memory is cheap
        // next to the weights either card holds.
        void* staging                 = nullptr;
        const cudaError_t staging_err =
            cudaHostAlloc(&staging, min_crossing_staging_bytes, cudaHostAllocPortable);
        if (staging_err != cudaSuccess) {
            release();
            throw std::runtime_error(
                cuda_error_message("cudaHostAlloc(cross-rank staging) failed", staging_err));
        }
        crossing_staging_       = staging;
        crossing_staging_bytes_ = min_crossing_staging_bytes;
    }

    active_rank_ = 0;
    err          = cudaSetDevice(endpoints_[0].device);
    if (err != cudaSuccess) {
        release();
        throw std::runtime_error(cuda_error_message("cudaSetDevice(primary) failed", err));
    }
    refresh_active_aliases();
}

DeviceContext::~DeviceContext() { release(); }

void DeviceContext::release() noexcept {
    if (crossing_staging_ != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(crossing_staging_));
        crossing_staging_       = nullptr;
        crossing_staging_bytes_ = 0;
    }
    for (Endpoint& endpoint : endpoints_) {
        if (endpoint.stream != nullptr || endpoint.transfer_stream != nullptr ||
            endpoint.vision_stream != nullptr || endpoint.fence != nullptr) {
            log_cuda_error("cudaSetDevice", cudaSetDevice(endpoint.device));
        }
        for (cudaEvent_t& piece : endpoint.piece_fences) { destroy_event(piece); }
        for (cudaEvent_t& piece : endpoint.piece_consumed) { destroy_event(piece); }
        destroy_event(endpoint.fence);
        destroy_stream(endpoint.vision_stream);
        destroy_stream(endpoint.transfer_stream);
        destroy_stream(endpoint.stream);
    }
    endpoints_.clear();
    device_ids_.clear();
    device          = 0;
    stream          = nullptr;
    transfer_stream = nullptr;
    vision_stream   = nullptr;
    props           = {};
    active_rank_    = 0;
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : crossing_staging_(other.crossing_staging_),
      crossing_staging_bytes_(other.crossing_staging_bytes_),
      endpoints_(std::move(other.endpoints_)), device_ids_(std::move(other.device_ids_)),
      active_rank_(other.active_rank_), peer_access_(other.peer_access_) {
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
    other.crossing_staging_       = nullptr;
    other.crossing_staging_bytes_ = 0;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    release();
    endpoints_              = std::move(other.endpoints_);
    device_ids_             = std::move(other.device_ids_);
    active_rank_            = other.active_rank_;
    peer_access_            = other.peer_access_;
    crossing_staging_       = other.crossing_staging_;
    crossing_staging_bytes_ = other.crossing_staging_bytes_;
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
    other.crossing_staging_       = nullptr;
    other.crossing_staging_bytes_ = 0;
    return *this;
}

void DeviceContext::refresh_active_aliases() noexcept {
    if (endpoints_.empty()) {
        device          = 0;
        stream          = nullptr;
        transfer_stream = nullptr;
        vision_stream   = nullptr;
        props           = {};
        return;
    }
    const Endpoint& endpoint = endpoints_[active_rank_];
    device                   = endpoint.device;
    stream                   = endpoint.stream;
    transfer_stream          = endpoint.transfer_stream;
    vision_stream            = endpoint.vision_stream;
    props                    = endpoint.props;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

// Streaming-multiprocessor count of the active device. Distinct from compute_capability(): every
// sm_86 part shares capability 86 but not this count, so residency budgets must read this.
int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

std::size_t DeviceContext::size() const noexcept { return endpoints_.size(); }

bool DeviceContext::model_parallel() const noexcept { return endpoints_.size() == 2; }

bool DeviceContext::peer_access() const noexcept { return peer_access_; }

void* DeviceContext::crossing_staging() const noexcept { return crossing_staging_; }

cudaEvent_t DeviceContext::piece_consumed_fence(std::size_t rank, std::size_t piece) const {
    if (rank >= endpoints_.size() || piece >= kCrossingPipelineDepth) {
        throw std::out_of_range("cross-rank piece fence is out of range");
    }
    return endpoints_[rank].piece_consumed[piece];
}

bool DeviceContext::piece_consumed_visible(std::size_t rank, std::size_t piece,
                                           unsigned long long capture_id) const {
    if (rank >= endpoints_.size() || piece >= kCrossingPipelineDepth) {
        throw std::out_of_range("cross-rank piece fence is out of range");
    }
    const Endpoint& endpoint = endpoints_[rank];
    return capture_id == 0 ? endpoint.piece_consumed_eager[piece]
                           : endpoint.piece_consumed_capture[piece] == capture_id;
}

void DeviceContext::note_piece_consumed(std::size_t rank, std::size_t piece,
                                        unsigned long long capture_id) {
    if (rank >= endpoints_.size() || piece >= kCrossingPipelineDepth) {
        throw std::out_of_range("cross-rank piece fence is out of range");
    }
    Endpoint& endpoint = endpoints_[rank];
    if (capture_id == 0) {
        endpoint.piece_consumed_eager[piece] = true;
    } else {
        endpoint.piece_consumed_capture[piece] = capture_id;
    }
}

cudaEvent_t DeviceContext::piece_fence(std::size_t rank, std::size_t piece) const {
    if (rank >= endpoints_.size() || piece >= kCrossingPipelineDepth) {
        throw std::out_of_range("cross-rank piece fence is out of range");
    }
    return endpoints_[rank].piece_fences[piece];
}

std::size_t DeviceContext::crossing_staging_bytes() const noexcept {
    return crossing_staging_bytes_;
}

std::size_t DeviceContext::active_rank() const noexcept { return active_rank_; }

const std::vector<int>& DeviceContext::device_ids() const noexcept { return device_ids_; }

cudaStream_t DeviceContext::stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].stream;
}

cudaStream_t DeviceContext::transfer_stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].transfer_stream;
}

cudaEvent_t DeviceContext::fence_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].fence;
}

void DeviceContext::activate_rank(std::size_t rank) {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    active_rank_ = rank;
    refresh_active_aliases();
}

void DeviceContext::synchronize_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    CUDA_CHECK(cudaStreamSynchronize(endpoints_[rank].stream));
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

void DeviceContext::synchronize() const {
    for (const Endpoint& endpoint : endpoints_) {
        CUDA_CHECK(cudaSetDevice(endpoint.device));
        CUDA_CHECK(cudaStreamSynchronize(endpoint.stream));
    }
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

ScopedDeviceRank::ScopedDeviceRank(DeviceContext& context, std::size_t rank)
    : context_(context), previous_rank_(context.active_rank()) {
    context_.activate_rank(rank);
}

ScopedDeviceRank::~ScopedDeviceRank() noexcept {
    if (context_.active_rank() != previous_rank_) { context_.activate_rank(previous_rank_); }
}

void stage_cross_rank_copy(DeviceContext& context, const void* source, std::size_t from_rank,
                           void* destination, std::size_t to_rank, std::size_t bytes) {
    const cudaStream_t from_stream = context.stream_for_rank(from_rank);
    const cudaStream_t to_stream   = context.stream_for_rank(to_rank);

    // Which capture this crossing belongs to, or 0 outside one. It decides which consumed fences
    // may be waited on: see DeviceContext::piece_consumed_visible.
    unsigned long long capture_id        = 0;
    cudaStreamCaptureStatus capture      = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamGetCaptureInfo(from_stream, &capture, &capture_id));
    if (capture != cudaStreamCaptureStatusActive) { capture_id = 0; }

    void* staging = context.crossing_staging();
    if (staging == nullptr || bytes > context.crossing_staging_bytes()) {
        throw std::runtime_error("cross-rank staging buffer is too small: need " +
                                 std::to_string(bytes) + " bytes, have " +
                                 std::to_string(context.crossing_staging_bytes()));
    }

    // Split the byte range so the two halves of the crossing overlap. Done as one copy, D2H must
    // finish before H2D starts and a 4 MB residual stream costs ~0.765 ms -- roughly twice what its
    // bandwidth implies. In pieces, piece i+1 streams out of the source while piece i streams into
    // the destination.
    //
    // Small transfers skip this: a decode crossing is ~4 KiB, where the per-piece launch and fence
    // overhead would cost more than the overlap saves.
    constexpr std::size_t kMinimumPipelinedBytes = 256U << 10;
    const std::size_t pieces =
        bytes >= kMinimumPipelinedBytes ? kCrossingPipelineDepth : std::size_t{1};
    const std::size_t piece_bytes = (bytes + pieces - 1) / pieces;

    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const std::size_t offset = piece * piece_bytes;
        const std::size_t length = std::min(piece_bytes, bytes - offset);
        if (length == 0) { break; }
        auto* staged            = static_cast<std::byte*>(staging) + offset;
        const auto* src         = static_cast<const std::byte*>(source) + offset;
        auto* dst               = static_cast<std::byte*>(destination) + offset;
        const cudaEvent_t fence = context.piece_fence(from_rank, piece);
        {
            ScopedDeviceRank guard(context, from_rank);
            // Every crossing stages through the same pinned buffer, and this D2H overwrites the
            // piece the *previous* crossing's H2D may still be reading -- that copy is only
            // enqueued here, never waited for. Wait on each rank's consumed fence for this piece
            // (an unrecorded event is satisfied, so the first crossing is free) rather than
            // tracking which rank consumed it last; with two ranks this is two no-op waits.
            for (std::size_t rank = 0; rank < context.size(); ++rank) {
                if (!context.piece_consumed_visible(rank, piece, capture_id)) { continue; }
                CUDA_CHECK(
                    cudaStreamWaitEvent(from_stream, context.piece_consumed_fence(rank, piece), 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(staged, src, length, cudaMemcpyDeviceToHost, from_stream));
            CUDA_CHECK(cudaEventRecord(fence, from_stream));
        }
        {
            ScopedDeviceRank guard(context, to_rank);
            CUDA_CHECK(cudaStreamWaitEvent(to_stream, fence, 0));
            CUDA_CHECK(cudaMemcpyAsync(dst, staged, length, cudaMemcpyHostToDevice, to_stream));
            // Publishes "this piece has been read out of staging"; the event belongs to the
            // destination device, which is the one recording it.
            CUDA_CHECK(cudaEventRecord(context.piece_consumed_fence(to_rank, piece), to_stream));
            context.note_piece_consumed(to_rank, piece, capture_id);
        }
    }
}



CudaEventTimer::CudaEventTimer(const DeviceContext& ctx)
    : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

} // namespace ninfer
