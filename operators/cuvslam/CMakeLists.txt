# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cuda)
include(isaac_deps/cuvslam)
include(isaac_deps/holoscan)

add_library(cuvslam_op SHARED cuvslam_op.cc)
add_library(isaac::operators::cuvslam ALIAS cuvslam_op)
target_include_directories(cuvslam_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(cuvslam_op
  PRIVATE
    CUDA::cudart
  PUBLIC
    cuvslam
    holoscan::core
    holoscan::sensor_io
    isaac::common::camera_info_cpu_messages
    isaac::common::geometry_cpu_messages
    isaac::common::pose_tree
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
