# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_executable(real_gr00t_n1_7_wbc_g1 main.cc)
target_include_directories(real_gr00t_n1_7_wbc_g1 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(real_gr00t_n1_7_wbc_g1
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::applications::gr00t_n1_7_wbc_g1::green_context
    isaac::applications::gr00t_n1_7_wbc_g1::shutdown_signals
    isaac::operators::agile_wbc
    isaac::operators::g1_low_level::joint_command_publisher
    isaac::operators::g1_low_level::joint_state_subscriber
    isaac::operators::gr00t_action_handler
    isaac::operators::gr00t_n1_7
    isaac::operators::joint_command_combiner
    isaac::operators::start_up
    isaac::operators::teleop_twist_command
    isaac::operators::v4l2_camera
)

add_dependencies(real_gr00t_n1_7_wbc_g1
  gr00t_n1_7_g1_apple_to_plate_models
  nvidia_wbc_agile_velocity_unitree_g1
)
