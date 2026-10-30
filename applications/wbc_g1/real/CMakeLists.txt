# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_executable(unitree_g1_wbc main.cc)
target_include_directories(unitree_g1_wbc PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(unitree_g1_wbc
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::operators::agile_wbc
    isaac::operators::g1_low_level::joint_command_publisher
    isaac::operators::g1_low_level::joint_state_subscriber
    isaac::operators::start_up
    isaac::operators::teleop_twist_command
)
add_dependencies(unitree_g1_wbc nvidia_wbc_agile_velocity_unitree_g1)
