find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Windows links the static CUDA runtime so shipped binaries do not depend on a cudart DLL.
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()

# --- LOCAL PATCH (text-only build) -------------------------------------------
# Upstream requires FFmpeg dev libraries here (supplied by vcpkg on Windows).
# This build targets text / MTP / DFlash2 inference only and ships a stubbed
# src/media/decode/decode.cpp, so the dependency is an empty interface target.
add_library(ninfer_ffmpeg_dependencies INTERFACE)
set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  # --- LOCAL PATCH (text-only build): see the FFmpeg note above. The stubbed
  # src/product/media_acquire/acquire.cpp removes the libcurl dependency.
  add_library(ninfer_curl_dependencies INTERFACE)
  set(NINFER_CURL_TARGET ninfer_curl_dependencies)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
