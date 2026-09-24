// ---------------------------------------------------------------------------
// Media decoding for this build: images are decoded with stb_image, which covers
// PNG / JPEG / BMP / GIF / WEBP. That is the format set coding agents actually
// paste (screenshots, diagrams), and it removes the FFmpeg dev-library dependency
// the upstream tree otherwise needs on Windows.
//
// Video decoding is NOT available here: it needs the libav* stack. decode_video /
// inspect_video reject the request with a clear error, and the Vision frontend
// only calls them for video inputs.
//
// The upstream FFmpeg implementation is kept next to this file as `decode.cpp.orig`.
// ---------------------------------------------------------------------------
#include "media/decode/decode.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA
#include "stb/stb_image.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ninfer::media::decode {

namespace {

void enforce_byte_budget(std::size_t bytes, const Policy& policy) {
    if (policy.max_bytes != 0 && bytes > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media payload exceeds the byte budget");
    }
    if (policy.checkpoint) { policy.checkpoint(); }
}

[[noreturn]] void fail_decode(const char* what) {
    const char* reason = stbi_failure_reason();
    throw Error(ErrorKind::InvalidInput,
                std::string(what) + ": " + (reason != nullptr ? reason : "decode failed"));
}

void probe(std::span<const std::uint8_t> bytes, const char* what, int& width, int& height) {
    if (bytes.empty()) { throw Error(ErrorKind::InvalidInput, "empty media payload"); }
    int channels = 0;
    if (stbi_info_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height,
                              &channels) == 0) {
        fail_decode(what);
    }
    if (width <= 0 || height <= 0) {
        throw Error(ErrorKind::InvalidInput, std::string(what) + ": image has no pixels");
    }
}

void enforce_pixel_budget(int width, int height, const Policy& policy) {
    if (policy.max_decoded_pixels == 0) { return; }
    const std::uint64_t pixels =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > policy.max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "image exceeds the decoded pixel budget");
    }
}

[[noreturn]] void video_unsupported() {
    throw Error(ErrorKind::InvalidInput,
                "video decoding is not available in this build (no FFmpeg); images are supported");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    enforce_byte_budget(bytes.size(), policy);
    int width = 0;
    int height = 0;
    probe(bytes, "inspect_image", width, height);
    enforce_pixel_budget(width, height, policy);
    return ImageInfo{width, height};
}

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    enforce_byte_budget(bytes.size(), policy);
    int width = 0;
    int height = 0;
    probe(bytes, "decode_image", width, height);
    enforce_pixel_budget(width, height, policy);
    if (policy.checkpoint) { policy.checkpoint(); }

    int source_channels = 0;
    // 3 desired channels: the Vision frontend consumes RGB24 (see Image.rgb).
    stbi_uc* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width,
                                            &height, &source_channels, 3);
    if (pixels == nullptr) { fail_decode("decode_image"); }

    Image image;
    image.width  = width;
    image.height = height;
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    image.rgb.assign(pixels, pixels + count);
    stbi_image_free(pixels);
    return image;
}

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    video_unsupported();
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    video_unsupported();
}

} // namespace ninfer::media::decode
