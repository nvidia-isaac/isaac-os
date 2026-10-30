# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_gr00t_n1_7_fused_op test_gr00t_n1_7_fused_op.cc)

include(isaac_deps/gtest)
target_include_directories(test_gr00t_n1_7_fused_op PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_gr00t_n1_7_fused_op
  PRIVATE
    GTest::gtest_main
    holoscan::core
    isaac::common::joint_command_chunk_cpu
    isaac::common::joint_cpu_messages
    isaac::operators::gr00t_n1_7
)

add_test(NAME test_gr00t_n1_7_fused_op COMMAND test_gr00t_n1_7_fused_op)
set_tests_properties(test_gr00t_n1_7_fused_op PROPERTIES
  LABELS "cpp;gpu;operators"
  TIMEOUT 120
)
