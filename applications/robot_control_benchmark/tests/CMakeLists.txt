# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)
include(isaac_deps/holoscan)



# Runs a real OnClock operator to verify its scheduled and observed physical coordinates.
# TIMEOUT covers three runs of 200 ticks at 1 kHz plus startup.
add_executable(robot_control_activation_coordinates_test activation_coordinates_test.cc)
target_compile_features(robot_control_activation_coordinates_test PRIVATE cxx_std_20)
target_include_directories(robot_control_activation_coordinates_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(robot_control_activation_coordinates_test
  PRIVATE
    GTest::gtest_main
    holoscan::core
)

add_test(NAME robot_control_activation_coordinates_test
  COMMAND robot_control_activation_coordinates_test)
set_tests_properties(robot_control_activation_coordinates_test PROPERTIES
  LABELS "cpp;applications"
  TIMEOUT 60
)
