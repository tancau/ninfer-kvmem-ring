// Fence qualification for the shared pinned crossing buffer used by `stage_cross_rank_copy`.
//
// Every crossing stages through one pinned buffer, and each piece's D2H overwrites the bytes the
// previous crossing's H2D may still be reading -- that copy is enqueued, never waited for. What
// makes reuse safe is a per-piece "consumed" event recorded on the destination stream, which the
// next crossing's D2H waits on.
//
// Two physical cards are not needed for that: the hazard is an ordering question between two
// streams. Both ranks therefore sit on device 0 and the test calls the staged path directly, since
// `TextContext::cross_rank_copy` would take its device-to-device shortcut for a shared device.
//
// The ordering is asserted on the captured graph rather than by racing two streams. A race harness
// has no power on Windows: holding the destination stream -- with a spinning kernel or with a
// blocking host function -- also stops the driver submitting the source stream's copies, so the
// source stream stalls whether or not the fence is there. The graph states the dependency
// outright, and capturing is what this path is written for in the first place, so the same test
// also proves the added waits did not cost capturability.

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kCrossingBytes = 1U << 20; // pipelines into kCrossingPipelineDepth pieces

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

// Says how a payload differs, so a failure names the mechanism instead of only the round.
int expect_payload(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& want,
                   const char* label) {
    if (actual == want) { return 0; }
    std::size_t first = 0;
    while (first < want.size() && actual[first] == want[first]) { ++first; }
    const bool zeroed = actual[first] == 0;
    std::cerr << "expectation failed: " << label << " (first difference at byte " << first
              << ": got " << static_cast<int>(actual[first]) << ", wanted "
              << static_cast<int>(want[first]) << (zeroed ? ", destination left unwritten" : "")
              << ")\n";
    return 1;
}

std::vector<std::uint8_t> pattern(std::uint8_t seed) {
    std::vector<std::uint8_t> bytes(kCrossingBytes);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((index * 31U + seed * 97U + 1U) & 0xFFU);
    }
    return bytes;
}

// Device buffer that frees itself, so an early return cannot leak one.
class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&data_, bytes)); }
    ~DeviceBytes() {
        if (data_ != nullptr) { (void)cudaFree(data_); }
    }
    DeviceBytes(const DeviceBytes&)            = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;

    [[nodiscard]] void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

void upload(const DeviceBytes& target, const std::vector<std::uint8_t>& bytes) {
    CUDA_CHECK(cudaMemcpy(target.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}

// cudaMemset is asynchronous with respect to the host, and the crossing streams are non-blocking,
// so a bare memset here can land *after* the crossing has written the destination and zero the
// bytes under test. Settle it before the crossing is enqueued.
void clear(const DeviceBytes& target) {
    CUDA_CHECK(cudaMemset(target.get(), 0, kCrossingBytes));
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<std::uint8_t> download(const DeviceBytes& source) {
    std::vector<std::uint8_t> bytes(kCrossingBytes);
    CUDA_CHECK(cudaMemcpy(bytes.data(), source.get(), bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}

std::vector<cudaGraphNode_t> graph_nodes(cudaGraph_t graph) {
    std::size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    if (count != 0) { CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count)); }
    return nodes;
}

// Finds the one memcpy node of `kind` that moves bytes out of `source` or into `destination`.
// Piece 0 of a crossing starts at the buffer's base address, which is what the caller passes.
cudaGraphNode_t find_memcpy(cudaGraph_t graph, cudaMemcpyKind kind, const void* source,
                            const void* destination) {
    cudaGraphNode_t found = nullptr;
    for (const cudaGraphNode_t node : graph_nodes(graph)) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) { continue; }
        cudaMemcpy3DParms params{};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &params));
        if (params.kind != kind) { continue; }
        if (source != nullptr && params.srcPtr.ptr != source) { continue; }
        if (destination != nullptr && params.dstPtr.ptr != destination) { continue; }
        if (found != nullptr) { return nullptr; } // ambiguous: the caller's identity is too weak
        found = node;
    }
    return found;
}

// Whether `to` can only run after `from`, following the graph's dependency edges.
bool depends_on(cudaGraph_t graph, cudaGraphNode_t from, cudaGraphNode_t to) {
    std::size_t edges = 0;
    // CUDA 13 dropped the four-argument form: the edge-data array is passed explicitly.
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
#else
    CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edges));
#endif
    std::vector<cudaGraphNode_t> sources(edges);
    std::vector<cudaGraphNode_t> destinations(edges);
    if (edges != 0) {
#if CUDART_VERSION >= 13000
        CUDA_CHECK(cudaGraphGetEdges(graph, sources.data(), destinations.data(), nullptr, &edges));
#else
        CUDA_CHECK(cudaGraphGetEdges(graph, sources.data(), destinations.data(), &edges));
#endif
    }

    std::deque<cudaGraphNode_t> pending{from};
    std::vector<cudaGraphNode_t> seen{from};
    while (!pending.empty()) {
        const cudaGraphNode_t node = pending.front();
        pending.pop_front();
        if (node == to) { return true; }
        for (std::size_t edge = 0; edge < edges; ++edge) {
            if (sources[edge] != node) { continue; }
            const cudaGraphNode_t next = destinations[edge];
            if (std::find(seen.begin(), seen.end(), next) != seen.end()) { continue; }
            seen.push_back(next);
            pending.push_back(next);
        }
    }
    return false;
}

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));

    const int device_ids[2] = {0, 0};
    ninfer::DeviceContext ctx(std::span<const int>(device_ids, 2), 2U << 20);
    if (ctx.crossing_staging() == nullptr || ctx.crossing_staging_bytes() < kCrossingBytes) {
        std::cout << "SKIP: no pinned crossing staging buffer\n";
        return 77;
    }

    int failures = 0;

    // 1. Byte integrity across back-to-back crossings that all reuse the same staging pieces.
    {
        DeviceBytes source(kCrossingBytes);
        DeviceBytes destination(kCrossingBytes);
        for (std::uint8_t round = 0; round < 8; ++round) {
            const std::vector<std::uint8_t> sent = pattern(round);
            upload(source, sent);
            clear(destination);
            ninfer::stage_cross_rank_copy(ctx, source.get(), 0, destination.get(), 1,
                                          kCrossingBytes);
            ctx.synchronize();
            failures +=
                expect_payload(download(destination), sent, "crossing delivers its own bytes");
        }
    }

    // 2. Two crossings captured into one graph: it captures at all, it replays the right bytes,
    //    and the second crossing's first D2H waits for the first crossing's first H2D -- the
    //    dependency that keeps one crossing from overwriting staging the previous one is reading.
    {
        DeviceBytes first_source(kCrossingBytes);
        DeviceBytes second_source(kCrossingBytes);
        DeviceBytes first_destination(kCrossingBytes);
        DeviceBytes second_destination(kCrossingBytes);
        const std::vector<std::uint8_t> first  = pattern(200);
        const std::vector<std::uint8_t> second = pattern(201);
        upload(first_source, first);
        upload(second_source, second);
        clear(first_destination);
        clear(second_destination);

        // Captured by hand rather than through DecodeGraphDefinition: the test needs the
        // cudaGraph_t itself to read the dependency it is asserting.
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream_for_rank(0), cudaStreamCaptureModeThreadLocal));
        ninfer::stage_cross_rank_copy(ctx, first_source.get(), 0, first_destination.get(), 1,
                                      kCrossingBytes);
        ninfer::stage_cross_rank_copy(ctx, second_source.get(), 0, second_destination.get(), 1,
                                      kCrossingBytes);
        // Capture ends on the origin stream, so the rank the crossings forked onto has to be
        // joined back into it before the capture closes.
        CUDA_CHECK(cudaEventRecord(ctx.fence_for_rank(1), ctx.stream_for_rank(1)));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream_for_rank(0), ctx.fence_for_rank(1), 0));
        CUDA_CHECK(cudaStreamEndCapture(ctx.stream_for_rank(0), &graph));
        failures += expect(graph != nullptr, "two staged crossings capture into a graph");

        const cudaGraphNode_t first_upload =
            find_memcpy(graph, cudaMemcpyHostToDevice, nullptr, first_destination.get());
        const cudaGraphNode_t second_download =
            find_memcpy(graph, cudaMemcpyDeviceToHost, second_source.get(), nullptr);
        if (first_upload == nullptr || second_download == nullptr) {
            failures += expect(false, "the graph carries both crossings' first pieces");
        } else {
            failures += expect(depends_on(graph, first_upload, second_download),
                               "the next crossing's D2H waits for the staged piece to be consumed");
        }

        cudaGraphExec_t executable = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, ctx.stream_for_rank(0)));
        ctx.synchronize();
        failures += expect_payload(download(first_destination), first,
                                   "the captured crossing delivers its payload");
        failures += expect_payload(download(second_destination), second,
                                   "the following captured crossing delivers its payload");
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    if (failures != 0) { return 1; }
    std::cout << "cross-rank staging fences hold\n";
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
