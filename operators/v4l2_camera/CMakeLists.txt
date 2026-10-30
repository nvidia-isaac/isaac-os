# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_library(v4l2_camera_op SHARED v4l2_camera_op.cc)
add_library(isaac::operators::v4l2_camera ALIAS v4l2_camera_op)

target_include_directories(v4l2_camera_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(v4l2_camera_op
  PUBLIC
    CUDA::cudart
    holoscan::core
  PRIVATE
    CUDA::nppicc
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
