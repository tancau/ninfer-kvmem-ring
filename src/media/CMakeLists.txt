add_library(ninfer_media_decode STATIC
  decode/decode.cpp
)
ninfer_internal_includes(ninfer_media_decode)
# LOCAL BUILD: decode.cpp decodes images with the vendored stb_image single header instead of
# FFmpeg, so it needs third_party on its include path (stb/stb_image.h).
target_include_directories(ninfer_media_decode PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET})
