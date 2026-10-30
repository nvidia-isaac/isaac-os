# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_teleop_twist_command test_teleop_twist_command_op.cc)

include(isaac_deps/gtest)
target_link_libraries(test_teleop_twist_command
  PRIVATE
    GTest::gtest_main
    isaac::operators::teleop_twist_command
)

add_test(NAME test_teleop_twist_command COMMAND test_teleop_twist_command)
set_tests_properties(test_teleop_twist_command PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
