# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_sinus_motion_op test_sinus_motion_op.cc)
target_link_libraries(test_sinus_motion_op
  PRIVATE
    GTest::gtest_main
    isaac::operators::sinus_motion
)
add_test(NAME test_sinus_motion_op COMMAND test_sinus_motion_op)
set_tests_properties(test_sinus_motion_op PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
