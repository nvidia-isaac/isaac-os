# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_start_up
  test_start_up_handover.cc
  test_start_up_op.cc
  test_waypoint_sequencer.cc
)

include(isaac_deps/gtest)
target_link_libraries(test_start_up
  PRIVATE
    GTest::gtest_main
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::operators::start_up
)

add_test(NAME test_start_up COMMAND test_start_up)
set_tests_properties(test_start_up PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 60
)
