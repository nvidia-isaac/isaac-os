# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/so101_reach_policy)

add_executable(so101_reach_real main.cc)
target_include_directories(so101_reach_real PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_reach_real
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::common::so101_driver
    isaac::operators::so101
    isaac::operators::so101_reach
)
add_dependencies(so101_reach_real grip_reach_delta_policy_onnx)
