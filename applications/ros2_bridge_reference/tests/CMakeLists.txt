# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

# Runs the reference app, sends it SIGINT mid-run and requires a clean exit. It does not link the
# app; it only launches the binary, which it is told the path of.
add_executable(test_ros2_bridge_reference_sigint test_ros2_bridge_reference_sigint.cc)
target_compile_definitions(test_ros2_bridge_reference_sigint PRIVATE
  "ISAAC_ROS2_BRIDGE_REFERENCE_BIN=\"$<TARGET_FILE:ros2_bridge_reference>\""
)
target_link_libraries(test_ros2_bridge_reference_sigint PRIVATE GTest::gtest_main)
add_dependencies(test_ros2_bridge_reference_sigint ros2_bridge_reference)
add_test(NAME test_ros2_bridge_reference_sigint COMMAND test_ros2_bridge_reference_sigint)
set_tests_properties(test_ros2_bridge_reference_sigint PROPERTIES
  LABELS "applications;cpp;integration"
  TIMEOUT 90
)
