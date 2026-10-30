# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_simple_motion_planner_op test_simple_motion_planner_op.cc)
target_link_libraries(test_simple_motion_planner_op
  PRIVATE
    GTest::gtest_main
    isaac::operators::simple_motion_planner
)
add_test(NAME test_simple_motion_planner_op COMMAND test_simple_motion_planner_op)
set_tests_properties(test_simple_motion_planner_op PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
