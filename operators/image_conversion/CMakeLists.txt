# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_library(image_conversion_op SHARED
  nv12_to_rgb_op.cc
)
add_library(isaac::operators::image_conversion ALIAS image_conversion_op)

target_include_directories(image_conversion_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(image_conversion_op
  PUBLIC
    holoscan::core
    holoscan::sensor_io
  PRIVATE
    CUDA::cudart
    CUDA::nppicc
    CUDA::nppidei
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
