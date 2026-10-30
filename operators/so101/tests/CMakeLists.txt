# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_so101_delta_motion_control_op test_so101_delta_motion_control_op.cc)
target_link_libraries(test_so101_delta_motion_control_op
  PRIVATE
    GTest::gtest_main
    isaac::operators::so101
)
add_test(NAME test_so101_delta_motion_control_op COMMAND test_so101_delta_motion_control_op)
set_tests_properties(test_so101_delta_motion_control_op PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)

add_executable(test_so101_state_source_op test_so101_state_source_op.cc)
target_link_libraries(test_so101_state_source_op
  PRIVATE
    GTest::gtest_main
    holoscan::core
    isaac::common::so101_cpu_messages
    isaac::common::so101_driver
    isaac::operators::so101
)
add_test(NAME test_so101_state_source_op COMMAND test_so101_state_source_op)
set_tests_properties(test_so101_state_source_op PROPERTIES LABELS "cpp;operators" TIMEOUT 15)
