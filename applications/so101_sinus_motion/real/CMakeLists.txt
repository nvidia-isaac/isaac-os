# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/holoscan)

add_executable(so101_sinus_motion_real main.cc)
target_include_directories(so101_sinus_motion_real PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_sinus_motion_real
  PRIVATE
    CLI11::CLI11
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::common::so101_driver
    isaac::operators::simple_motion_planner
    isaac::operators::sinus_motion
    isaac::operators::so101
)

# Isolated target writes make the servo-side motion profile measurable on real hardware.
add_executable(so101_motion_profile motion_profile.cc)
target_include_directories(so101_motion_profile PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_motion_profile
  PRIVATE
    CLI11::CLI11
    holoscan::config_io
    isaac::common::so101_driver
)
