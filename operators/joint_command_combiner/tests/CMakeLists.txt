# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_joint_command_combiner test_joint_command_combiner_op.cc)

include(isaac_deps/gtest)
target_link_libraries(test_joint_command_combiner
  PRIVATE
    GTest::gtest_main
    isaac::operators::joint_command_combiner
)

add_test(
  NAME test_joint_command_combiner
  COMMAND test_joint_command_combiner
)
set_tests_properties(test_joint_command_combiner PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
