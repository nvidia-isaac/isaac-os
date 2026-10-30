# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_joint_command_chunk_cpu test_joint_command_chunk_cpu.cc)
target_link_libraries(test_joint_command_chunk_cpu
  PRIVATE
    GTest::gtest_main
    isaac::common::joint_command_chunk_cpu
)

add_test(NAME test_joint_command_chunk_cpu COMMAND test_joint_command_chunk_cpu)
set_tests_properties(test_joint_command_chunk_cpu PROPERTIES
  LABELS "cpp;common"
  TIMEOUT 30
)
