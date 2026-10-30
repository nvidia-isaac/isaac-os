# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cuda)
include(isaac_deps/foxglove-sdk)
include(isaac_deps/holoscan)
include(isaac_deps/mcap)

set(mcap_logger_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET tensor_frame_generated
  SCHEMAS tensor_frame.fbs
  OUTPUT_DIR "${mcap_logger_generated_dir}"
  BFBS_EMBED
)

add_library(mcap_logger SHARED
  frame_encoder.cc
  mcap_sink.cc
)
add_library(isaac::common::mcap_logger ALIAS mcap_logger)
add_dependencies(mcap_logger tensor_frame_generated)

target_include_directories(mcap_logger PUBLIC ${PROJECT_SOURCE_DIR} "${mcap_logger_generated_dir}")
target_link_libraries(mcap_logger
  PUBLIC
    # PUBLIC, not PRIVATE: tensor_frame_builder.h includes <cuda_runtime.h> and calls cudaMemcpy
    # directly, and that header is pulled in by mcap_logger.h through tensor_topic_poller.h, so
    # every consumer of this target compiles CUDA runtime calls of its own.
    CUDA::cudart
    # PUBLIC: mcap_logger.h and tensor_topic_poller.h both include <fmt/format.h> and call
    # fmt::format() in inline code, so every consumer of this target compiles fmt calls of its
    # own -- must not rely on holoscan::core happening to expose fmt::fmt transitively.
    fmt::fmt
    holoscan::core
    holoscan::schema
    holoscan::sensor_io
    isaac::foxglove_sdk
)

# Hardware video encoding is only compiled in when the NVIDIA Video Codec SDK is enabled.
# PUBLIC on the definition, not PRIVATE: it decides whether frame_encoder.h declares a member,
# so every consumer of this header must agree with this library on the class layout. The
# encoder and NPP themselves stay PRIVATE -- frame_encoder.h only forward declares the session
# type, so they never reach a consumer's compile line.
if(ISAAC_ENABLE_NV_VIDEO_SDK)
  target_compile_definitions(mcap_logger PUBLIC ISAAC_ENABLE_NV_VIDEO_SDK)
  target_link_libraries(mcap_logger
    PRIVATE
      CUDA::nppidei
      isaac::common::image_schema
      isaac::common::nv_video_encoder
  )
endif()

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
