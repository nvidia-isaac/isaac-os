# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVIDIA Video Codec SDK dependency setup.

# Already exists?
if(TARGET nv::encoder AND TARGET nv::decoder)
  return()
endif()

# Dependencies
include(isaac_deps/cuda)
include(isaac_deps/nvidia-encode-driver)
include(isaac_deps/nvidia-decode-driver)

find_path(NVC_SDK_DIR REQUIRED
  NAMES Samples/NvCodec/NvEncoder/NvEncoder.cpp
  HINTS "${NVC_SDK_ROOT}" ENV NVC_SDK_ROOT
  PATHS /opt/nvidia/video-codec-sdk/13.0.37
)
set(NVC_CODEC_DIR "${NVC_SDK_DIR}/Samples/NvCodec")

# Common interface library
add_library(nv_encoder SHARED
  ${NVC_CODEC_DIR}/NvEncoder/NvEncoder.cpp
  ${NVC_CODEC_DIR}/NvEncoder/NvEncoderCuda.cpp
)
add_library(nv::encoder ALIAS nv_encoder)
target_include_directories(nv_encoder
  PUBLIC ${NVC_SDK_DIR}/Interface
  PUBLIC ${NVC_CODEC_DIR}
  PUBLIC ${NVC_SDK_DIR}/Samples/Utils
)
target_compile_definitions(nv_encoder
  PUBLIC
    NV_WINDOWS=0
    NV_LINUX=1
)
target_compile_options(nv_encoder
  PRIVATE
    -Wno-missing-field-initializers
    -Wno-reorder
    -Wno-unused-parameter
)
target_link_libraries(nv_encoder
  PUBLIC
    CUDA::cuda_driver
    nv::encode_driver
)

# Only the decoder uses the SDK logger in 13.0.37.
add_library(nv_decoder SHARED
  ${CMAKE_CURRENT_LIST_DIR}/nvidia-video-codec/logger.cpp
  ${NVC_CODEC_DIR}/NvDecoder/NvDecoder.cpp
)
add_library(nv::decoder ALIAS nv_decoder)
target_include_directories(nv_decoder
  PUBLIC ${NVC_SDK_DIR}/Interface
  PUBLIC ${NVC_CODEC_DIR}
  PUBLIC ${NVC_SDK_DIR}/Samples/Utils
)
target_compile_definitions(nv_decoder
  PUBLIC
    NV_WINDOWS=0
    NV_LINUX=1
)
target_compile_options(nv_decoder
  PRIVATE
    -Wno-missing-field-initializers
    -Wno-reorder
    -Wno-unused-parameter
)
target_link_libraries(nv_decoder
  PUBLIC
    CUDA::cuda_driver
    nv::decode_driver
)

isaac_install(
  COMPONENT DEP_nvidia_video_codec
  TARGETS
    nv_decoder
    nv_encoder
)

message(STATUS "NVIDIA Video Codec SDK configured at: ${NVC_SDK_DIR}")
