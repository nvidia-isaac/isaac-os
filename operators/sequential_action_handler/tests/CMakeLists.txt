# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_sequential_action_handler test_sequential_action_handler_op.cc)

include(isaac_deps/gtest)
target_link_libraries(test_sequential_action_handler
  PRIVATE
    GTest::gtest_main
    isaac::operators::sequential_action_handler
)

add_test(NAME test_sequential_action_handler COMMAND test_sequential_action_handler)
set_tests_properties(test_sequential_action_handler PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
