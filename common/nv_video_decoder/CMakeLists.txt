# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_library(nv_video_decoder SHARED nv_video_decoder.cpp)
add_library(isaac::common::nv_video_decoder ALIAS nv_video_decoder)

target_include_directories(nv_video_decoder PUBLIC ${PROJECT_SOURCE_DIR})

include(isaac_deps/cuda)
include(isaac_deps/nvidia-video-codec)
target_link_libraries(nv_video_decoder
    PUBLIC
        CUDA::cuda_driver
        holoscan::core
        nv::decoder
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
